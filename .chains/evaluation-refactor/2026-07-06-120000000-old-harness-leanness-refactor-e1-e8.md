---
chain: "evaluation-refactor"
branch: "main"
nextStep: "None. Implemented 2026-07-06 on refactor/kernels-eval; archived 2026-09-06 with roadmap C3, which replaced this harness with harness v2 (see the evaluation-harness-v2 chain)."
created: "2026-07-06T12:00:00Z"
---

# Evaluation harness refactor E1-E8: readability and leanness (old harness)

Source: `docs/plans/archive/evaluation-refactor.md`. Implemented E1-E8 on `refactor/kernels-eval` 2026-07-06 (line refs against `2b1ff80`); archived 2026-09-06 with roadmap C3. Harness v2 deleted every file this plan touched except `encode.py`, `metrics.py`, `oracle.py`, `config.py`. Companion: kernels-layers-design (library side).

## 1. Intent
Three kinds of debt in the old harness: dead weight (a broken unused algo, an unreachable CPU timing path, identity indirection, three unused deps, a one-shot upload script); parameter threading (15-20 kwargs through five levels; the reverse-clause fix touched five `sweep.py` signatures for one field); doc drift in `docs/system/evaluation.md`.

## 2. As-built map (2026-07-03)
- Entry points: `evaluate` (one algo x one config -> one JSON), `run-evaluation` (orchestrator, one `uv run evaluate` subprocess per (config, algo), `--eval-type {filter,quality,param-sweeps}`), `stage-results`, `upload-results`, ETL CLIs `yambda` / `arxiv` / `goodreads`, `eval-fetch` / `eval-publish` / `eval-publish-checkpoint`, `upload-checkpoints`.
- Driver: `cli/evaluate.py` -> seeds, TF32 off, `recompile_limit=64` -> `queries_cache.load_or_cache_queries` (keyed `(ckpt_mtime, max_seq_length, users_limit)`) -> `loaders.load_query_attrs` -> `sweep.run_sweep` -> JSON.
- Row schema: `suite, cell, filter_kind, sweep, impl, backend, device, seed, batch_size, k, n_users_kept, median_ms, p20_ms, p80_ms, peak_mem_mib, index_mem_mib, fwd_scratch_mib, recall@K, ndcg@K, extra.params`.
- Measurement invariants: TF32 off; one-shot warm-up outside windows; warm-up before peak-memory reset; memory window without do_bench's L2 buster; timing extends `rep_ms` to >= `MIN_SAMPLES=30` (cap `MAX_REP_MS=3000`); pool of 4096 fixed-seed batches; quality streamed at `QUALITY_BATCH_SIZE=16`; per-sweep `torch._dynamo.reset()` + `empty_cache()`; per-algo process isolation.

## 3. Phases
- E1 dead weight: delete `TorchKnnAlgo` (a plain class without `__call__`, `TypeError` on first use, in no YAML); delete the unreachable `measure_forward_cpu` and every `is_cpu` branch (keep the `device` column hard-coded `"cuda"`); inline `expand_param_combos` (keep `is_valid_combo`, the `n_probe > n_lists` skip); drop `torchvision`, `matplotlib`, `einops`; quarantine `upload_results.py` (`--repo-id` required, `--notes-file`, `--private/--public`); rename `datasets` -> `eval_datasets` (it shadowed HuggingFace `datasets`, which `sentence-transformers` imports).
- E2 context objects: frozen `SweepContext` (once per run, post `users_limit`) and `FilterAssets` (per filter kind, per-sweep fields stamped by `dataclasses.replace`).
- E3: `RetrievalAlgo` Protocol; `SUPPORTED_FILTER_KINDS` table instead of `ValueError` as control flow (the old `except ValueError` silently dropped cells on real bugs such as `k > n_probe x max_cluster_size`); `FilterKind` literal.
- E4: `AlgoBase._finalize` holding the one `torch.compile(dynamic=True, mode="reduce-overhead")` call; forwards stay in each file as the readable spec of each cascade.
- E5: split `bench_tools.py` into `measure.py` / `encode.py` / `passes.py`; `PerfStats` / `QualityStats`; emit `precision@k`, `mrr@k` (already computed, previously discarded).
- E6: oracle cache keyed by content fingerprint (sha256 over shapes, dtypes, 64 linspace rows, `k_gt`), filename `gt_topk_v3_`; fixes the goodreads stale-cache incident that the roadmap called a publication blocker.
- E7: `resolve_path` no longer falls back to the basename; one `_`-key YAML strip; `output: null` guard; docstring sweep.
- E8: CPU unit tests first (`test_metrics.py`, `test_sweep_helpers.py`, `test_oracle.py`, `test_config.py`), then the `evaluation.md` rewrite.

## 4. Conventions and gates
Golden run before phase 1 and after every phase (goodreads d128 `c0_genre` `linr_v3`): quality byte-identical, same `cell` keys, latency within ~5 %. Never rename an existing JSON row field (thesis plotting consumes them). Grep gates: zero hits for `expand_param_combos`, `is_cpu`, `measure_forward_cpu`, `TorchKnnAlgo`, `BACKEND_CAPABLE_ALGOS`.

## 5. Deferred small items, later closed by harness v2
JSONL streaming writes, tqdm control characters in tee'd logs, `weights_only=True`, `EVAL_TYPES` hand-sync, provenance columns in rows, keeping `triton_knn` / `linr_v1_filter_mask` names for YAML lineage.
