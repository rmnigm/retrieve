---
chain: "refactor-validation-handoff"
branch: "main"
nextStep: "Run the runbook top to bottom on an A100: step 2 (K2 tracing caveat) first, then the full library suite, then the harness steps 1, 4, 6, 7 and the per-kernel perf gates of step 5."
created: "2026-07-06T12:00:00Z"
---

# GPU validation runbook for `refactor/kernels-eval`

Source: `docs/plans/refactor-validation-handoff.md` (library half) and `docs/plans/archive/refactor-validation-handoff-harness.md` (harness half, split out 2026-09-06). Written 2026-07-06 against branch `refactor/kernels-eval` on `main` @ `2b1ff80`: 12 phase commits, one per work package (K1 to K8 of kernels-layers-design, E1 to E8.1 of evaluation-refactor, plus the rename).

## 1. What landed (library, K1 to K8)
- K1: fixed the broken `tune-kernels codesigned-probe-score` subcommand (called the public op with kwargs removed in Stage 2b) + CUDA-gated tuner smoke test.
- K2: host-wrapper dedup in all seven kernel files: `_<name>_prep` (validation, contiguity, buffers, launch kwargs in a frozen `_<Name>Launch`) and `_<name>_finish`; every entry point is prep, one launch line, finish. The last four `@custom_op` kernels (`clause_mask`, `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`) moved to `@triton_op`; their shape-branching (P-bucketing, `p == 0` early return, pad tails) lives only in the eager `_impl`s behind `bucket=` / `pad_to_k=`. All ten ops `@triton_op` with inline `wrap_triton`.
- K3: shared `@triton.jit` helpers in `common.py` (`popcount_int64`, `bloom_subset_pass`, `clause_pass`, `compact_store`, `or_combine`); bloom kernels on the `qb & ~sig` OR-reduce form.
- K4: `masked_topk` / `counts_to_valid` replacing six inlined epilogues; `_PackedBitsKNN` base folding `SimHashKNN`'s ~95 % copy of `OneBitKNN` plus the `k_bits` re-registration fix; SilverTorch `register_index` split into validate / IVF / quantize / filter-buffer phases with frozen buffer order; the `candidate_ids` + `query_clause_attrs` guard.
- K5: bloom hash math behind public names in `bloom_hash.py`; hash output bit-identical.
- K6: `FilterModule.register_index(item_clause_attrs, *, clause_is_reverse=None)` kw-only on ABC and both filters; minimal `RetrievalModule` ABC.
- K7: `tune.py` as a `KernelTuneSpec` registry: seven subcommands incl. the new `codesigned-probe-score-exact`.
- K8: new tests `test_topk_util.py`, `test_bit_knn_base.py`, `test_bloom_hash.py`, `test_tune_smoke.py`, `tests/compile/test_export_kernel_ref.py`.
- Rename `SilverTorch(filter=...)` to `filter_mode=`, no shim.

Invariants through validation: parity with unchanged tolerances (strict equality on OPORP / SimHash), `retrieve::*` op schemas frozen, buffer names and order frozen, `wrap_triton` textually inline in every `@triton_op` body.

## 2. The steps
- Step 1 (harness, CPU): `uv run pytest retrieval/tests/`: 5 files, 24 functions.
- Step 2 (first, pattern-setter): `test_codesigned_probe_score.py`, `test_silvertorch_compile.py` (zero graph breaks on all three `filter_mode`s), `test_export_kernel_ref.py`; then `test_clause_mask.py` / `test_clause_compact.py` for keyword args into `@triton.jit` helpers.
- Step 3: full `retrieve` suite, strict parity.
- Step 4 (harness): `TORCH_LOGS=graph_breaks uv run evaluate --config config/goodreads/d128-filter.yaml --algo linr_v3 --filter-kind clause --sweep c0_genre --skip-quality`: no new breaks.
- Step 5: per-kernel `tune-kernels --json-out` medians within +-5 % of `main` (on `main`, `codesigned-probe-score` is broken by the K1 bug: baseline from the K2 commit instead). Then `tune-kernels codesigned-probe-score-exact` end to end.
- Step 6 (harness): golden diff `main` vs branch on goodreads d128 `c0_genre` `linr_v3`, joined on `(filter_kind, sweep, impl, backend, batch_size, k)`: quality byte-identical, additive columns only (`precision@k`, `mrr@k`, `extra.{gpu,torch,commit}`), identical cell keys, latency within ~5 %, one oracle rebuild then cache hits.
- Step 7 (harness): `uv run run-evaluation config/goodreads/d128-filter.yaml --resume -- --skip-quality --sweep c0_genre` exits 0 with resume skips and run logs.

## 3. Fallbacks
- F1 (K2 tracing fails on the frozen-dataclass + kwargs splat): prep returns a plain tuple, positional splat; prove on `codesigned_probe_score.py`, roll to the other six (`bloom_match.py` has no prep).
- F2 (keyword args into `@triton.jit` helpers rejected): positional call sites.
- F3 (a kernel regresses > 5 %, most plausibly `clause_pass` register pressure): revert that kernel's body to the inlined predicate with a `keep in sync with common.py::clause_pass` breadcrumb.
- F4 (golden quality drift): never acceptable; bisect by phase commit. Suspects in order: `masked_topk` call-site flags, stale oracle caches on both sides, TF32 pins, kernel numerics.
- F5 (export loses the kernel reference): exactly one textual `wrap_triton(` per `@triton_op` body.
- F6 (orchestrator smoke fails): `output:` unset is correct `SystemExit`; resume skip = target JSON parses non-empty.

## 4. Intentional behaviour deltas
- Library: `PrefilterKNN` torch path gets the `isfinite -> -1` sentinel; `masked_topk` pads in the scores dtype (fp16 `-inf` for PostfilterKNN paths); `PostfilterKNN*` with `k > N` pad instead of raising; SilverTorch torch-backend epilogue tombstones `-inf` winners to `-1` (the Triton tail did not yet); `forward(query_clause_attrs=..., candidate_ids=...)` raises `ValueError`; `_forward_candidates` with `P < k` returns `min(k, P)` columns; `OneBitKNN` re-resolves the `k_bits=0` sentinel on every `register_index`; fmkt public op gained an identity `clamp_max(p-1)` and now validates inputs; bloom kernels on the OR-reduce subset form.
- Harness: `cell` key format unchanged; loud failures (unknown `filter_kind`, algo construction errors, missing attrs paths); `datasets` renamed `eval_datasets` (it shadowed HuggingFace `datasets`); `upload-results` requires `--repo-id`; oracle cache `gt_topk_v3_` keyed by content fingerprint, `users_limit` in it; `torch_knn` deleted.

## 5. Known latent bug recorded here
On the checkpoint path `queries_cache` trimmed to `users_limit` before caching, while `load_query_attrs` compared `eval_split.parquet` rows against the trimmed count: goodreads d128 filter (`users_limit: 10000`) could raise `eval_split rows != queries`. Fixed by roadmap A1 on the old harness; harness v2 applies `users_limit` once after the row check.
