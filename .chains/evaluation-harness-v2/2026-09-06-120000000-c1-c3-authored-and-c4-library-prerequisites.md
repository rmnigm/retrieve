---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-05-120000000-plan-h-harness-v2-protocol-and-architecture.md"
nextStep: "Re-derive A1's golden against the deterministic k-means before the C4 1e-6 gate can mean anything; then rerun C4."
created: "2026-09-06T12:00:00Z"
---

# §9: C1-C3 authored (CPU) and the three C4 library prerequisites (A100)

## C1-C3, 2026-09-06, Mac / CPU, `dev/c1-harness-v2`
`CUDA_VISIBLE_DEVICES="" uv run pytest retrieval/tests/ -q`: 82 passed (8 files); ruff clean; links 0. End-to-end tests drive `run.run`, `bench run`, `bench campaign` on a 24-item, 8-query pre-encoded fixture on `backend="torch"`, eager, shrunken windows: record schema, resume by key with `code_version`, a failed cell recorded and the loop continuing, the exact-algo gate, the parity spill across two backends, one real child process per group.

In the tree: `bench.py` 349, `metrics.py` 111, `algos.py` 328, `config.py` 365, `data.py` 291, `oracle.py` 264, `run.py` 582, `cli.py` 190, `upload.py` 60, `encode.py` 117 = 2,657 code; tests 1,699. Budget was 1,450 + 350; overrun is docstrings (~1/3), record assembly, the two loaders, and tests locking the old harness's numbers. Deleted 30 old files / 4,339 lines (`f021179`).

Deviations (also in the system doc): parity spill `.npz` with ids + scores, hash of the key block minus `backend`, cleaned per `(dataset, dim, algo)`; samples as a JSONL sidecar; resume re-runs `failed` / `partial`; campaign child is `python -m ... run` on the same interpreter; `EXACT_ALGOS = (linr_v1_filter_mask, linr_v2)`; no `--flush-l2`; `training/evaluate.py` moved to the new metrics API; `bench campaign` forwards `--skip-quality / --skip-perf / --profile`.

## C4 library prerequisites, 2026-09-06, A100, `dev/c4-library-fixes`
A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, nvcc 12.8, Python 3.11, `/venvs/c4` with `--extra official`; clocks unlocked; tests and probes only, nothing citable.

1. Compaction kernels as opaque custom ops (`dd8b7b5`, verifying `5815275`): under `triton_op` inductor's TTIR mutation analysis follows `compact_store`'s data-dependent address back through the `tt.call` arguments and reports the index buffers (graph inputs) as mutated, so cudagraph trees skip the whole forward. New gate `tests/compile/test_linr_compile.py` (`reduce-overhead`, `dynamic=False`, `fullgraph=True`, 5 warm-ups, reads `cudagraph_skips`).

| cell | skips on `custom_op` | skips on `triton_op` (control) |
|---|---|---|
| linr_v1 clause / bloom | 0 / 0 | 0 / 0 |
| linr_v2 clause / bloom | 0 / 0 | 1 / 0 |
| linr_v3 clause / bloom | 0 / 0 | 1 / 0 |
Scores `torch.equal` to eager, ids up to ties. Only `clause_compact` tripped at these shapes; `bloom_compact` converted too (same store, silent failure mode). Export still works (graph holds the op node; bit-exact round trip). 67 tests passed.

2. Deterministic k-means (`d5d824b`): `bincount` counts + a float64 one-hot GEMM accumulated panel by panel with `addmm_`, no float atomics; float64 costs 8 % over float32 and cannot be demoted to TF32 by a caller, so nothing global is toggled. N=200k, D=128, n_lists=1024, n_iter=10: old `index_add_` 142.3 ms (two seed-0 fits not equal in general), new 173.7 ms, 1.22x, centroids and assignments `torch.equal`. One update from a fixed assignment differs by max abs 3.3e-6 (the atomic side's fp32 error); full fits are chaotic (6.0e-8 and 2.1e-2 apart on different runs); the objective differs by 5.4e-9 relative. `test_kmeans.py` 5 passed; `test_official.py` 43 passed.

3. `-1` id sentinel in the Triton SilverTorch epilogue (`8df7e9a`, plan O §14.7): `torch.where(isfinite(topk_scores), topk_ids, -1)` in `_cps_finish` / `_cpse_finish`. New `TestFewSurvivorsSentinel` (4 cells fail without the change). 167 passed.

Suites: library 516 passed, 3 failed (the pre-existing B4 `SimHashKNN` `k_bits` lambda bug); harness CPU 103 passed, 1 skipped; ruff clean; links 0.

Notes for the coordinator:
1. A1's golden must be re-derived: every SilverTorch golden cell came from the atomic k-means (one arbitrary draw), and A1's `linr_v2` / `linr_v3` graph cells were compiled-eager, not graph.
2. The sentinel can move SilverTorch triton quality on rows with < k survivors: `metrics._hits` masked on `ids != -1`, not score finiteness, so a filtered-out id in a `-inf` slot could count as a hit (prediction later shown wrong: see the 2026-09-15 golden re-derive note).

Operational finding: inductor's on-disk FX cache (`/tmp/torchinductor_root`) does not invalidate when a `@triton_op` body's Python source changes; after editing `_cps_finish` compiled cells kept running the old epilogue. Clear it or set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
