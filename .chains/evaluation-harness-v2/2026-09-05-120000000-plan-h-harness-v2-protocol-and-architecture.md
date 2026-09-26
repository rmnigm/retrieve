---
chain: "evaluation-harness-v2"
branch: "main"
nextStep: "WP-0 (roadmap A1): commit the users_limit row-count fix and run the golden cells on the old harness; then WP-1..WP-3 (C1-C3) on the CPU."
created: "2026-09-05T12:00:00Z"
---

# Plan H: evaluation harness v2 (protocol, architecture, work packages)

Source: `docs/plans/evaluation-harness-v2.md` §1-§8 (plan "H"), planned 2026-09-05 on `feat/cute-dsl-scorer`, authored on the Mac. Supersedes the harness half of refactor-validation-handoff and roadmap §4a/§4b. Raw survey artifacts: `docs/artifacts/evaluation-harness-v2/` (`survey-ann-ir-benchmarks.md`, `survey-bench-infrastructure.md`). The as-built reference is the wiki page `docs/system/evaluation.md`; the §2 protocol is reproduced verbatim there.

Amendment the same day: official ops become the reference backend and CUDA / CuTe are deleted (plan O): read every `cuda` / `cute` below as `official`, whose `graph` mode records `null` with reason `not_capturable`.

Question: what must the harness measure to be comparable with SilverTorch ("Evaluation") and LiNR ("Model Inference Benchmarking"), and what is the smallest code that measures it correctly for every dataset x dim x backend without babysitting?

## 1. Verdicts on the old harness (3,884 lines, 30 files, 19 YAMLs)
1. `do_bench` with a ~256 MiB L2 flush per call, p20 / p80 only, no p99 / QPS / mean / vector / build time, `MIN_SAMPLES = 30`: real. Both papers report warm replayed traffic (SilverTorch: 5,000 replayed requests, 50 warm-up + 100 test batches, P99 + QPS; LiNR: mean + p95 by batch size).
2. What was timed was cudagraph replay with `dynamic=True` (every algo compiled in its constructor, quality too) and `cudagraph_skips` never asserted. Neither paper serves through graph replay; the cute WP-6 table measured eager 0.31-1.29 ms vs graph 0.06-0.29 ms. Fix: quality eager; perf as labelled `eager` and `graph` variants.
3. Memory columns: `index_mem_mib` was an allocator delta excluding the mask algos' filter but including SilverTorch's attrs; `fwd_scratch_mib` was 0.0 on 6,066 of 7,104 rows. Fix: `index_mib = Σ buffers` incl. the filter submodule, `filter_mib`, `peak_fwd_mib` from the eager window.
4. Wide repetitive rows; the `cell` string collided on 90 / 162 deep-sweep rows.
5. Global backend axis: `PostfilterKNN*` ignore the flag, so `torch` rows duplicated `triton`; filters built per algo backend. Fix: backends per algo, `PATHS` maps to the code that runs, duplicates collapse.
6. Held-out recall missing on filter cells: add `target_in_filter`.
7. Robustness: hard-coded device, `--filter-kind` typo -> empty JSON, all-or-nothing resume, zero-latency rows when all users skipped. Blocking bug: `users_limit` on a checkpoint dataset made `load_query_attrs` raise on the row count, so the golden gate could not run.
8. One seed everywhere, no repeated windows, no spread.
9. 19 near-identical YAMLs.
10. TF32 pinned twice; `_dynamo.reset()` only between sweeps; no clocks or rich provenance; per-chunk metric syncs; eager bloom paid a pageable H2D per call (the salt); the process boundary was (config, algo), not the thesis's (dataset, dim, bs, filter, impl).

## 2. Protocol (now the canonical text in `docs/system/evaluation.md`)
Cell = `(dataset, dim, filter_kind, sweep, algo, backend, params, seed)`. Environment once per process; inputs once per `(dataset, dim)` with one `users_limit` site (prefix); build (`build_s`, `index_mib`, `filter_mib`); quality eager once per cell at `k_max`, oracle on filter cells and held-out always, exact algos assert `recall_oracle@k >= 0.99`; perf per `(k, bs, mode)` with a 4,096-batch fixed-seed pool, no L2 flush, `graph` = `reduce-overhead`, `dynamic=False`, `fullgraph=True`, skips asserted 0; 50 warm-ups, 3 windows of `clamp(2 s / median, 1000, 5000)` calls, median of window medians, `spread > 0.05` -> `unstable`; seeds only on `deep` and headline filter cells; comparability paragraph (eager is comparable, graph is deployed best case); cell cost ~2 min (estimate; D1-a measured otherwise).

## 3. Target architecture
Files budget 1,450 code + 350 tests: `config.py`, `data.py`, `oracle.py`, `metrics.py`, `algos.py`, `bench.py`, `run.py`, `cli.py`, `report.py`, tests. Surviving classes: `Job` (its `asdict` is the record key) and the algo modules (for `buffers()`, `.k`, compile, readable forwards). Output: one JSONL record per cell + a samples file. Config: one YAML per dataset + `suites.yaml`. CLI `bench run | campaign | report`; the campaign is a subprocess loop per group with per-child logs.

## 4. Not to abstract
No context bags, no stats dataclasses, no algo framework, no config object model, no logging framework in the orchestrator, no results-IO layer or composite `cell` string, no timing strategy classes, no exception swallowing / retries / device option / `--eval-type` presets.

## 5. Deleted
`context.py`, `passes.py`, `results_io.py`, the algos package, `stage_results`, `run_evaluation`, `evaluate`; `do_bench`, L2 flush, `rep_ms` loop; constructor-time `torch.compile`, `recompile_limit`, `_autotune_prewarm`; the old memory / percentile / `cell` columns (schema break, old JSONs to `results/archive/`); 19 YAMLs; `triton_knn` alias; staging layout and tee.

## 6. Work packages
- WP-0 golden baseline on the old harness (A1).
- WP-1 `bench.py`, `metrics.py`, `algos.py` + tests (C1). WP-2 `config.py`, `data.py`, `oracle.py` + tests (C2). WP-3 `run.py`, `cli.py`, deletions, docs (C3).
- WP-4 GPU gate (C4): goodreads d128 clause `c0_genre`, all algos and backends. Gates: (1) quality within 1e-6 of golden; (2) graph median within 5 % of golden; (3) `cudagraph_skips == 0`; (4) `jaccard_vs_first@100 == 1.0` torch vs triton on exact algos and official vs triton on SilverTorch; (5) no `unstable` cell at locked clocks; (6) kill mid-run and `--resume` continues.
- WP-5 campaign (D1). Gate: `bench report` with no missing cells; `median_ms(bs=16) < 16·median_ms(bs=1)`; ids identical across `mode`; a rerun byte-identical in quality.
- WP-6 `report.py` (D4).

## 7. Risks as written
`module.k` baked into buffers (none found); eager bloom inflated by the salt (fixed at B5); per-bs static capture cost; torch at bs=16 materialising `[B, P, D]` (let OOM kill loudly); `users_limit` a prefix not a sample (kept for golden comparability); clock locking needs root (fallback: sampled `sm_mhz` + `unstable`, and the paper says so); held-out recall thin on tight sweeps; schema break; no precomputed-oracle input (YFCC's shipped GT is validated out of band by `yfcc_check_gt.py`; WP-2 should add a load-this-blob path with a per-dataset metric: YFCC is squared L2).

## 8. External practice review (two literature surveys)
- Confirmed: one appended record per finest unit as the resume mechanism; provenance inline; sweeps as data, Cartesian-producted; closed loop by default with a named second mode; cutoff-qualified metric keys (`recall@100`). Correctly not done: Docker per algorithm, a separate results repo, a power axis.
- Accepted: A build vs query params sweep separately (the `deep` suite is 2 builds, not 12; `set_query_params`); B `code_version` = `git rev-parse HEAD:retrieve/src/retrieve` in the resume key; C `schema_version`; D `status`; E IQR + outlier counts, never drop outliers; F `git_branch`, `python`, `clocks_locked` recorded, `report.py` enforces `dirty`; G `flat.csv` before any plot; H QPS-vs-recall Pareto + recall-at-budget as default views; I the oracle as a shippable artifact with its hash in the filename; J `disabled: true`; K one process per `(dataset, dim, algo, backend)` (user decision 2026-09-06), parity via spill file.
- Rejected: open-loop load generation (no retrieval harness does it for queries, neither paper reports it; record `load: "closed_loop"`); Hydra / W&B Sweeps / MLflow / Sacred / DVC.
