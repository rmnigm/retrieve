---
title: evaluation
created: 2026-09-26
updated: 2026-10-07
type: entity
tags: [harness]
sources: [evaluation/bench/, evaluation/config/, evaluation/tests/]
---

# `evaluation/` — the benchmark harness (`bench`)

The retrieval benchmark that produces the thesis and paper numbers: the
measurement protocol, the module map, the config matrix, the `bench` CLI,
the JSONL record, resume and the campaign process model. Why it is built
this way is in [decisions](../decisions.md#harness); which of its gates
pass is in [validation](../validation.md#harness-gates).
**Nothing this harness produces is citable until its gate is green in
[validation](../validation.md)** (CLAUDE.md rule 2).

`evaluation/` is three packages with one dependency direction, `bench` →
`training` → `eval_datasets` (`tests/test_dependency_direction.py` enforces
it): `bench` measures algorithms (this document), `training` makes the
embeddings the sequential datasets need, `eval_datasets` owns what is on
disk. For the other two see [datasets.md](datasets.md) and
[checkpoints.md](checkpoints.md); for the algorithm internals
[kernels.md](kernels.md) and [architecture.md](architecture.md); for the
library's correctness suite, [testing.md](testing.md).

## Measurement protocol

One **cell** = `(dataset, dim, filter_kind, sweep, algo, backend, params, seed)`.
One process runs one `(dataset, dim, algo, backend)` group; per cell, in
this order (everything is `eager` unless labelled `graph`). The harness
docstrings cite these steps as `§2.1`-`§2.8`.

1. **Environment, once per process.** Seed torch/CUDA; TF32 off and
   `float32_matmul_precision("highest")` (`measure.setup`); one 3-matmul GPU
   warm-up (`warm_gpu_once`); record provenance: GPU name, driver, CUDA
   runtime, torch/triton versions, the installed official (`silvertorch`)
   commit, git commit + dirty flags, branch,
   `code_version`, hostname, python, UTC start, and one `nvidia-smi` clock
   sample (`env.sm_mhz_idle`, `mem_mhz`, `sm_max_mhz`, `power_limit_w`).
   Clocks cannot be locked on the pods
   ([decisions](../decisions.md#harness)), so every perf entry carries its
   own under-load `sm_mhz` (see [Reading the numbers](#reading-the-numbers)).
2. **Inputs, once per (dataset, dim).** Item embeddings, queries, held-out
   targets, query attrs; `users_limit` applied once, as a prefix. Filter
   modules: one per *filter backend* (`triton` for triton and official
   cells, `torch` for torch cells). Per sweep: synthesised query attrs,
   skip mask, exact oracle top-`max(ks)` (the [blob v4](#oracle-blob-v4)
   cache) **plus** `target_in_filter[u]` (the held-out item passes the exact
   mask) and `pass_rate` (exact; bloom cells also record the bloom rate as
   `bloom_fp_rate`), the axis both papers organise their tables around
   (LiNR high/low pass-rate; SilverTorch FP rate against bits).
3. **Build.** `build_s` = wall time of algo construction + `register_index`
   with a sync on each side (`measure.timed_build`), recorded on every cell
   of the build (a `deep` job with six `n_probe` values repeats one
   `build_s` six times). `index_mib = Σ numel·itemsize` over
   `algo.buffers()` (the filter is a submodule, so mask algos include it,
   `silvertorch` includes its attrs); `filter_mib` for the filter alone. No
   allocator deltas. `SilverTorch.build_timings` holds the per-phase build
   seconds (`kmeans_s`, `assemble_s`, `quantize_s`, `filter_s`); the harness
   does not record it yet.
4. **Quality, eager, once per cell at `k_max = max(ks)`.** Stream all kept
   users in chunks of 16 (`QUALITY_CHUNK`, the OOM bound of `[B, P, D]` on
   loose filters; the per-row queries, attrs, targets and oracle top-k /
   `targets_in_filter` go to the device once per cell and each chunk gathers
   its rows there, torch's CPU threads held at `QUALITY_CPU_THREADS` = 1 for
   the loop: H-QLOOP, the host gathers and pageable copies were ~80 % of the
   pass), accumulate `recall/ndcg/precision/mrr` at every `k` in
   `ks` from the one top-`k_max` list (exact for every library algo: same
   candidate set, same scores, `torch.topk` sorted). `postfilter` is the
   exception (`run.PER_K_QUALITY`): its candidate pool is `alpha * k`, so
   the prefix of its `k_max` run is not its top-`k` run, and its quality
   pass runs once per `k`. Targets: **oracle** on
   filter cells, **held-out** always (on filter cells restricted to targets
   the exact mask admits, with `n_queries_heldout` and
   `n_targets_in_filter` recorded). Metrics accumulate as running sums on
   device, one sync at the end. The exact algos (`EXACT_ALGOS`:
   `linr_v1_filter_mask`, `linr_v2`) must
   reach `recall_oracle@k_max ≥ 0.99` (fp16 tolerance); a failure is
   recorded as `failed` and then raises `QualityGateError`, which ends the
   run. Cross-backend correctness belongs to the library's parity suite
   ([testing](testing.md)); the harness keeps only a *wiring* check, the
   parity spill: the first backend to run a cell writes its top-`k_max` ids
   (int32) and scores (float32) to
   `results/_parity/<dataset>-d<dim>_<algo>/<hash>.npz` (`run.parity_group`
   names the directory; the hash is over the key block minus `backend` and
   minus `params.score_path`, the official kernel's epilogue, so `h2h`'s
   official arms compare against the triton spill; it includes `suite`,
   `filter_kind` and `sweep`), and later backends
   record `jaccard_vs_first@k` and `score_max_abs_diff` against it. There is
   no "missing reference" state: whichever backend runs a cell first writes
   the spill (`parity: "reference"`), so with `--resume` after the triton
   cells are done, `torch` becomes the reference and `official` records
   `vs_torch`. A spill written by the *same* backend (killed after the spill,
   before the record) is rewritten as the reference, never compared against.
   `bench run` never deletes the directory (the next backend's run needs
   it); `bench campaign` drops every other group's spill before each child,
   so a campaign restarted mid-group still compares against the spill of
   the backends the killed process finished.
5. **Perf, per `(bs, k, mode)`** with `mode ∈ {eager, graph}` and
   `module.k = k` set before each variant. Inputs: the fixed-seed pool of
   query batches (and attr batches) rotated round-robin, identical across
   backends and modes. No L2 flush (`--flush-l2` does not exist): the pool
   rotation makes index reads cold while the structures serving keeps hot
   stay hot. `graph` = `torch.compile(mode="reduce-overhead",
   dynamic=False, fullgraph=True)` of the module per bs; after warm-up
   `measure.graph_callable` asserts `cudagraph_skips == 0` and exactly one
   `cudaGraphLaunch` per call, else the entry is null with a `reason`, never
   a mislabelled number. `official` is not capturable and records `reason:
   not_capturable`. Its bloom forward parses each query batch into
   expression plans on the CPU (about 59 µs per call at B=16,
   [kernels](kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend))
   and memoises them by default; the pool replays batches, so a cached cell
   would not pay what serving fresh queries costs. Quality runs with the
   library default (cache on); `run.perf` then swaps in
   `OfficialConfig(cache_plans=False)` in place (read per forward, no
   rebuild, bit-identical results) before the first timed variant, and every
   entry records `cache_plans`. An entry with `cache_plans: true` is not a
   timing number. Per variant (`measure.latency`): 50 warm-up calls → sync →
   **3 windows** of `N = clamp(2 s / median_est, 1000, 5000)` calls, each
   call bracketed by CUDA events on the current stream, wall clock around
   the window with one sync at the end. From the window with the median
   median: `median_ms, mean_ms, trimmed_mean_ms, p95_ms, p99_ms, min_ms,
   iqr_ms, n`, two
   outlier counts (`outliers_std` beyond 3σ, `outliers_tukey`), `qps =
   N·bs / wall_s` (closed-loop single client, `load: "closed_loop"`, the
   in-process analogue of the papers' client-side QPS), `host_gap_ms =
   wall/N − mean_gpu_ms` (diagnostic), `window_medians_ms` and `spread =
   (max − min) / median` of the three window medians; `spread > 0.05` sets
   `unstable: true`. After each window's sync one `nvidia-smi` SM clock
   sample goes to `window_sm_mhz` (outside the timed calls). The per-call vector of the chosen window goes to the
   samples sidecar. `peak_fwd_mib` = `max_memory_allocated −
   allocated_before` over the first eager window only (graph mode
   allocates nothing). The first eager call runs under
   `torch.cuda.set_sync_debug_mode("warn")` to catch hidden host syncs.
   `--profile` (off by default) wraps one eager call in `torch.profiler`
   and stores per-kernel CUDA µs (top 8 kernels) as `kernels`, and the device
   time and launches summed over every kernel as `kernels_us` / `kernels_calls`
   (`measure.kernel_summary`; H-KSUM: the top-8 sum is a lower bound that favours
   the arm with more launches), and the same sums per named kernel scope as
   `kernel_scopes` (H-SCOPE, scorer / topk / epilogue / other, so T3's scorer
   column compares like with like). The sums skip `## …` events: the profiler books a
   compiled call's `## Call CompiledFxGraph <hash> ##` range as a device event
   spanning that graph's kernels, so counting it would count them twice (the top-8
   list keeps it). In a
   long-lived process the profiler drops the device activities at the end of
   a short window, one more kernel per session as the process ages, so H2H-FINAL
   stored empty or truncated lists
   ([validation](../validation.md#harness-gates)). `measure.profile_once` therefore
   brackets the call with two `torch.cuda._sleep` sentinel kernels
   (`spin_kernel`, left out of `kernels`). It retries with an idle pad of
   0.01, 0.1, 1 and 5 s on both sides of the window until both sentinels are
   recorded, and raises (the cell fails at the perf stage) if they never are.
6. **Seeds and repeats.** Timing repeats are the 3 windows above (no
   rebuild). Seeds change the IVF (k-means), the OPORP projection
   (`v3_seed`) and the pool; V1/V2 and the postfilter are seed-invariant in
   quality, so their later seeds copy seed 0's ([quality cache](#quality-cache)).
   Every suite runs seeds `[0, 1, 2]` on every dataset and sweep (user,
   2026-10-08), `codesign`'s as timing repeats; `h2h` runs five. The report takes the
   median across seeds with min-max whiskers.
7. **Comparability, stated once in the paper.** Same as SilverTorch: A100,
   D=128, INT8 IVF, an `n_probe` grid including **24** (their production
   setting), bloom bits and FP rate, mean latency at bs=16 against recall
   (their Fig. 5), QPS and P99 from the same vector. Same as LiNR: V1-V3 by
   batch size (1, 16), mean + p95, label recall on filtered sets, high/low
   pass-rate bucketing by `pass_rate`. Necessary differences: in-process
   (no RPC, no user tower, no 5,000-request replay), smaller catalogs than
   the papers' 10-80 M / 15.5 M, an 80 GB card, k ≤ 1000 against
   1024/2000, closed-loop QPS rather than open-loop load, no live updates.
   `eager` is the comparable number; `graph` is the deployed-best-case
   number and is reported alongside, never instead.
8. **Cell cost.** Quality once per cell (not per k); perf 3 k × 3 bs × 2
   modes × 3 windows. Build params sweep separately from query params (one
   build, many query configs). Measured per-cell costs (D1's arxiv `deep`
   groups, pubmed's D=768 filter cells) are in roadmap H3
   ([roadmap](../roadmap.md#phase-h-harness-prerequisites)).

The record carries `schema_version`, `status`, `code_version` in the
resume key, `git_branch` / `python` in `env`, `disabled: true` sweeps and
the oracle fingerprint in the blob's file name; see
[Output](#output-one-jsonl-record-per-cell).

### Reading the numbers

- **Per-call `*_ms` is closed-loop latency, not kernel time.** The CUDA
  events bracket the GPU timeline between the two `record`s; when the
  GPU idles waiting for the host (eager, `bs = 1`) the interval includes
  launch latency and `host_gap_ms ≈ 0`. That is the latency the papers
  report; `--profile` (`kernels`) is the kernel-time view.
- **Clocks are compared load against load, never against idle.**
  `env.sm_mhz_idle` is the process-start sample, provenance only. Every
  perf entry samples the SM clock right after each timing window's sync,
  under load, into `window_sm_mhz`; its `sm_mhz` is the last of these (the
  same sample as before `window_sm_mhz` existed). A throttle inside one
  window shows up as that window's clock sample, next to its entry in
  `window_medians_ms`; neither `unstable` nor `clocks_drift` reads
  `window_sm_mhz`. `env.sm_mhz_load` is the median of a cell's under-load
  samples, and `env.clocks_drift` fires (and sets the record's `unstable`)
  when any of them is more than 5 % (`run.CLOCK_DRIFT`) from the process's
  *first* under-load sample. An idle sample reads low and would flag the GPU
  boosting, and there is no `clocks_locked` field because the pods cannot
  lock clocks. Compare latencies across runs against `perf[].sm_mhz`.
- **The job's clock log.** `env.frac_windows_below_max` is the share of a
  cell's window samples below the device's max SM clock, `env.sm_max_mhz`
  (`nvidia-smi` `clocks.max.sm`, sampled at process start; 1410 MHz on the
  A100), so it does not depend on which cells ran before; `null` without a
  device max. The record reuse rule reads it (under 10 %,
  [decisions](../decisions.md#campaign-v2-user-2026-10-08)). The full
  view is in the job's log: `bench campaign` writes `nvidia-smi -q -d CLOCK`
  before the child starts and after it ends, and `run` logs the histogram
  of every window sample of the process at its end
  (`sm_mhz under load (n=…): 1410×…, 1395×…`, `measure.clock_histogram`).
  [c4_gate.py](../artifacts/evaluation-harness-v2/c4_gate.py) reads the
  schema-1 `env.sm_mhz` field.
- **No L2 flush between calls.** `measure.latency` does not flush L2
  (`triton.testing.do_bench` does by default). Consecutive calls take
  consecutive batches of the query pool, so what depends on the query
  (which index rows a batch reads) varies call to call, while what every
  call reads stays in L2 as it would in serving. Cache state is therefore
  the serving steady state, not a cold-cache worst case; a flushed number
  could be higher on bandwidth-bound variants, and it is not measured. The
  kernel microbenchmarks under `docs/artifacts/` (`q3/roofline.py`,
  `kernel-opt/bench_kernels.py`) do not flush either.
- **One sync per window, not per call.** `_time_calls` records a CUDA
  event pair around each call and calls `torch.cuda.synchronize()` once
  before the window and once after it, never inside the loop, so a sync
  does not inflate the per-call numbers. The host is free to run ahead
  unless the module itself syncs (the first eager call runs under
  `set_sync_debug_mode("warn")` to catch that).
- **No sub-launch-floor flag.** Per-call numbers under about 10 µs
  would be mostly timer and launch overhead. The smallest `min_ms` in
  the D1-a records (Hub subtree `d1-a`, 2106 perf entries) is 0.196 ms (`silvertorch`
  triton graph, bs = 1), 20× above that, so no entry carries a flag for it.
- **`trimmed_mean_ms`** is the mean after dropping `n // 10` calls from
  each end of the chosen window: a central value that one-sided
  interference on the shared GPU moves less than `mean_ms`. `mean_ms`
  stays the papers' comparable number.
- **Samples go to a JSONL sidecar** (`<name>.samples.jsonl`, one line per
  perf entry with the key block, `k`, `bs`, `mode`, `ms: [...]`), because
  parquet cannot be appended per cell. `bench report` reads it for the
  latency violins.

## Architecture

Twelve modules under [`evaluation/bench/`](../../evaluation/bench/), plus
the two the harness takes from its sibling packages
(`eval_datasets.layout` for the on-disk readers, `training.encode` for the
SASRec encode). No base class, no context bags, no stats dataclasses, no
results-IO layer; two things are classes on purpose — `Job`, the
resolved build spec whose `key(params)` *is* the record's key block, and
the algo `nn.Module`s (the library's, and the harness's one baseline),
which exist for `buffers()`, `.k`, `torch.compile` and because their
`forward` is the readable spec of each cascade.

| module | owns |
|---|---|
| [`measure.py`](../../evaluation/bench/measure.py) | `setup`, `warm_gpu_once`, `provenance` (GPU, driver, CUDA, torch, triton, `official_commit` — the installed `silvertorch`'s PEP 610 `direct_url.json` `vcs_info.commit_id`, `None` when it is not installed from git — commit, `dirty` = `subtree_dirty()` over `LIB` (the imported `retrieve` package's directory, in whichever checkout holds it), `repo_dirty`, branch, `code_version` = `LIB`'s tree hash, or `files:<sha256>` of the sources on disk when it is dirty, `lib_dir` = `LIB`, host, python, started; [Resume](#resume)), `clocks()` (one `nvidia-smi` sample), `timed_build`, `index_bytes` (Σ buffers, submodules included, deduplicated), `stats`, `latency_group(fns, bs=, mode=)` (§2.5 windows of each arm, round-robin across arms, IQR + outlier counts, `load: closed_loop`, `peak_fwd_mib`, the under-load `sm_mhz`) and `latency(fn, …)`, its one-arm case, `graph_callable` (raises `NotCapturable` with the record's `reason`; no dynamo reset, so interleaved arms stay captured side by side), `profile_once` |
| [`records.py`](../../evaluation/bench/records.py) | what a record *is*: `SCHEMA_VERSION`, `KEY_FIELDS`, `resume_key`, `record_path`, `samples_path`, `append_record` (one `write` + `fsync`), `read_records` / `read_keys` (one torn trailing line tolerated), `record_files`, `latest` (last record per key), `aggregate(results_dir) → results.parquet` (one row per perf entry — what `report.py` reads), `read_table` |
| [`metrics.py`](../../evaluation/bench/metrics.py) | `accumulator(ks, device)` / `accumulate(acc, ids, targets, num_targets=None, ranked=False)` (returns the chunk's per-row recall, the [sidecar](#per-query-sidecar)'s values) / `finalize(acc)` (`null` metrics when `n == 0`, `null_if_empty`) — recall, ndcg, precision, mrr at every `k` from one top-`k_max` list as float64 running sums on device; `ranked=True` scores against the oracle's own top-`k` prefix; `per_row`, `jaccard_at_k`. `training/evaluate.py` keeps its own frozen copy, pinned to agree (`training/test_encode.py`) |
| [`algos.py`](../../evaluation/bench/algos.py) | the algorithm table: `ALGOS` name → class (`LiNRV1`–`LiNRV3`, `SilverTorch`; `Postfilter`, the harness's own), `FILTER_KINDS`, `BACKENDS`, `FILTER_MODE` (`clause` → `exact`), `DISPATCH` (the library's table plus the `Postfilter` row), `PATHS` **derived from it**, `filter_backend` (`official` → `triton`), `build(algo, item_embs, k=, backend=, …)` (construct + `register_index`, ≈ 25 lines), `build_filter`, `is_valid_combo`, `official_config` (`bloom_path` / `score_path` → `OfficialConfig`) |
| [`postfilter.py`](../../evaluation/bench/postfilter.py) | `Postfilter`, the generic-torch baseline ([below](#the-postfilter-baseline)) |
| [`config.py`](../../evaluation/bench/config.py) | `Dataset`, `Job`, `load_dataset`, `load_matrix` — the config matrix below; `interleave_units`, `shared_key` (the [interleave groups](#interleaved-groups)) |
| [`inputs.py`](../../evaluation/bench/inputs.py) | `load_inputs` (dispatch to `training.encode.encode_split` or the `eval_datasets.layout` text readers; `users_limit` once, as a prefix), `sweep_qa`, `build_filters` (keyed by filter backend), `exact_filter`, `query_pool` |
| [`oracle.py`](../../evaluation/bench/oracle.py) | the exact filtered oracle as blob v4, `attrs_digest`, `pass_counts`, `pass_rate`, `bloom_fp_rate` |
| [`run.py`](../../evaluation/bench/run.py) | `run(jobs, out_dir=...)` — the cell loop; `MODES`, `QUALITY_CHUNK = 16`, `QUALITY_CPU_THREADS = 1`, `EXACT_ALGOS`, `SEED_FREE_QUALITY`, `PER_K_QUALITY`, `PERF_STAT_KEYS`, `IDS_PROBE_BATCHES`, `CLOCK_DRIFT` |
| [`cli.py`](../../evaluation/bench/cli.py) | `bench run` / `campaign` / `check` / `upload` / `fetch` / `report` / `env` |
| [`report.py`](../../evaluation/bench/report.py) | `bench report`: `results.parquet`, the paper tables as LaTeX, the figures, the methodology paragraph and `report.md`; the `ARTIFACTS` dispatch table and the citability verdict. See [Report](#report-reportpy) |
| [`upload.py`](../../evaluation/bench/upload.py) | `bench upload`: publish a results tree (records, samples, a freshly aggregated `results.parquet`) to the HF results repo with a `MANIFEST.json` (provenance + a sha256 per file) and a generated README; `bench fetch`: one subtree back, checked against its manifest. See [Results storage](#results-storage) |

### Algorithms and the `PATHS` table

`PATHS[(algo, filter_kind, backend)]` names the code path that actually
runs, or `None` when there is no such cell. It is what a record's `path`
column carries and what `load_matrix` collapses on: backends that run the
same code become one job (logged once), `None` triples are skipped. It is
*derived* from `retrieve.interfaces.DISPATCH` (the library's dispatch
table as data — [architecture.md](architecture.md#backend-dispatch)):
`DISPATCH[class][backend]` is the label (`cublas` where the flag is a
no-op, `None` where the constructor raises), the cuBLAS algos' filter
cells get `+<backend>` for the filter's kernel, and `linr_v2 / none` is
`None` because its candidate source is the filter. `algos.DISPATCH` adds
one row the library does not have, the harness's `Postfilter` (torch only;
`postfilter / none` is `None`, there being nothing to filter).
`tests/bench/test_paths.py` pins the derivation.

| algo | `none` | `clause` / `bloom` | `official` |
|---|---|---|---|
| `linr_v1_filter_mask` (`PostfilterKNN`, fp16 cuBLAS + mask) | `cublas` (triton and torch collapse) | `cublas+triton` / `cublas+torch` (the filter's kernel) | — |
| `linr_v2` (`PrefilterKNN` over the filter's candidate list) | — (the candidate source is the filter) | `triton` / `torch` | — |
| `linr_v3` (`OneBitKNN` top-`candidate_pool` → `PrefilterKNN`) | `triton` / `torch` | `triton` / `torch` | — |
| `silvertorch` (IVF + INT8, predicate fused: `filter_mode` none / exact / bloom) | `triton` / `torch` | `triton` / `torch` | `official` |
| `postfilter` (the harness's baseline: fp16 cuBLAS, top-`alpha*k`, then the filter) | — | `cublas+torch` (no triton cell) | — |

`official` is Meta's reference backend and exists for
`silvertorch` only; its standalone filter modules are Triton
(`algos.filter_backend`), and it is eager-only (`SilverTorch.capturable`
is `False` there), so its `graph` perf entries are `null` with `reason:
not_capturable`.

Every algo module but `postfilter` is the library's: `forward(query,
query_clause_attrs=None) -> (ids [B, k], scores [B, k])`, `torch.topk`-sorted
rows, `-1` ids where a row has fewer than `k` survivors, `k` settable after
`register_index`, `set_query_params` (`n_probe` on `SilverTorch`,
`candidate_pool` on `LiNRV3`, `alpha` on `Postfilter`) re-validating without a rebuild, `capturable`
a class attribute, the filter a submodule (`self.filter` on the LiNR
variants; `SilverTorch` fuses the predicate and carries the attribute
buffers inside `index_mib`). `build(algo, item_embs, k=, backend=,
filter_kind=, filter_mod=, item_attrs=, clause_is_reverse=, params=,
seed=)` is the one factory; it refuses `None`-path cells and `n_probe >
n_lists`. The build param `bloom_path` (`partial` | `full`, the S9
co-design ablation) exists on `silvertorch / bloom` on two backends
(anywhere else `load_matrix` raises `ConfigError`):
- **official:** `SilverTorch(official=OfficialConfig(bloom_path=…))`,
  through `algos.official_config`;
- **triton:** the layer's own `SilverTorch(bloom_path=…)` (C5-OURS,
  [kernels](kernels.md#bloom_full_mask--the-full-n-mask-bloom_pathfull)).

Without the key the library default runs, which is `bloom_path="partial"`
on both. `params` are the merged build + query params over
`algos.SILVERTORCH_DEFAULTS` (`n_lists 1024, n_probe 24, n_iter 10`); on
`silvertorch` bloom cells `run.py` merges the suite's `bloom` defaults
(`m_bits`, `k_hash`) in as well. The probe-pool check is the library's,
in `set_query_params` and at forward.

### The postfilter baseline

`postfilter` ([`bench/postfilter.py`](../../evaluation/bench/postfilter.py))
is the study's baseline ([decisions](../decisions.md#harness)): what a
practitioner writes without a retrieval library. It lives in the harness,
not the library, because it is by definition not a retrieval library's
algorithm. Per batch: one dense matmul over the whole item table (LiNR V1's
scoring — fp16 table, fp32 scores, cuBLAS — so the two differ only in where
the filter runs), `torch.topk(min(alpha * k, N))`, the standalone filter's
`evaluate_subset` on those candidates only, the first `k` survivors kept in
rank order (a stable sort on the keep flags), and the rest of the row
padded with the `-1` / `-inf` sentinel. A row whose top-`alpha*k` held
fewer than `k` items passing the filter comes back short, and the missing
ids count as a recall loss against the exact oracle. Unlike
`linr_v1_filter_mask` (mask before top-k) it is not exact and is not in
`EXACT_ALGOS`.

`alpha` (a positive int, default 1) is a query param: `set_query_params(alpha=)`,
swept over `{1, 2, 4, 8}` in the `filter` suite against one build, so the
report can show what a naive system pays in latency to recover recall;
`alpha = 1` is the headline baseline. The pool is recomputed from the
current `k` at every forward, so perf at `k` and quality at `k` (the
per-`k` pass, `run.PER_K_QUALITY`) measure the same pool. Torch backend
only, clause and bloom filter kinds; capturable.

### The router arm

`router` ([`bench/router.py`](../../evaluation/bench/router.py), roadmap V-ROUTER) routes each query
by its **local pass rate** `l_q`: the share of its unfiltered top-100 that its filter admits (EXHIBITS
idea #2: IVF recall follows `l_q`, not the global pass rate). `algos.build("router", …)` assembles
three parts with the ordinary builders and one seed, so both SilverTorch indexes share one k-means:
an unfiltered SilverTorch pre-probe (`pre_n_probe`, k 100), a filtered SilverTorch (`n_lists`,
`n_probe`, the IVF branch) and exact LiNR V2 on the standalone filter. `forward` runs the pre-probe,
takes `l_q` from the filter's `evaluate_subset` on its 100 ids, sends the rows with `l_q <
lq_threshold` to V2 and the others to the IVF branch (each on its own sub-batch), and scatters the
results back; the pre-probe is inside `forward`, so the arm's timing includes it. The split is
data-dependent: `capturable = False`, so `graph` records `not_capturable`. Build params
`pre_n_probe`, `lq_threshold` (router only, both required, a `ConfigError` otherwise); query param
`n_probe` (the IVF branch). The quality pass writes `router_lq` and `router_exact` (1.0 = V2) per
kept query to the per-query sidecar and `quality.router_exact_share` to the record. Triton and
torch backends, clause and bloom.

## Config: one YAML per dataset + `suites.yaml`

Twelve files under [`evaluation/config/`](../../evaluation/config/):
[`goodreads.yaml`](../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../evaluation/config/arxiv.yaml),
[`yambda-500m.yaml`](../../evaluation/config/yambda-500m.yaml),
[`yambda-5b.yaml`](../../evaluation/config/yambda-5b.yaml),
[`yfcc10m.yaml`](../../evaluation/config/yfcc10m.yaml),
[`pubmed.yaml`](../../evaluation/config/pubmed.yaml) and
[`openalex.yaml`](../../evaluation/config/openalex.yaml) and
[`kuairand.yaml`](../../evaluation/config/kuairand.yaml), the three
synthetic-selectivity siblings `goodreads-synth.yaml`, `arxiv-synth.yaml`
and `yfcc10m-synth.yaml` plus arXiv's cluster-correlated `arxiv-corr-synth.yaml`
([datasets](datasets.md#synthetic-selectivity-attrs)) (goodreads, arxiv,
yfcc10m, pubmed and openalex in the `filter` suite, pubmed and openalex at
768; yambda and kuairand are out of the study), and
[`suites.yaml`](../../evaluation/config/suites.yaml). `users_limit:
10000` and the goodreads/arXiv sweeps, ks and batch sizes match the
[golden cells](../../evaluation/golden/README.md), so the two stay
comparable. The grid is one backend per algo (`triton`; `silvertorch` also
runs `official`), without LiNR V4 (it was ours, not LiNR's, and is
out of the harness) and without a dim ablation
([decisions](../decisions.md#harness)).
[`tests/bench/test_config.py`](../../evaluation/tests/bench/test_config.py)
pins the grid: every (suite, dataset)'s job and cell counts and the
rules above on every cell (G-grid), and the resume keys of cells the
redesign kept, as `b96e1f2` computed them (G-key). It also runs
every `config/*.yaml` against every suite through `load_matrix`: a listed
dataset expands to jobs, an unlisted one is refused by name and still
resolves at each of its dims.

`suites.yaml` is the campaign-v2 grid (user decisions 2026-10-08). Every
suite runs seeds `{0, 1, 2}` on every dataset and sweep, except `h2h`'s five
repeats `{0 … 4}`. Batch sizes are
`{1, 16}` and ks are `{100, 1000}` or a subset of those. No suite has
`n_probe` 4 or 256, LiNR V4, or openalex (`config/openalex.yaml` and its
ETL are backlog). Official SilverTorch runs only `bloom` cells: its clause
cells timed our `pack_mask` adapter. The official clause path stays in
`PATHS` and the library, so the old records still read. There is no
unfiltered `quality` suite ([decisions](../decisions.md#harness)); synth's
`p1` sweep is the unfiltered point.

| suite | datasets | arms | feeds |
|---|---|---|---|
| `filter` | goodreads, arxiv, yfcc10m, pubmed, three kept sweeps each (yfcc10m: `tags_and`) | V1, V2, V3 (pool 5000) triton; `silvertorch` triton clause + bloom and official bloom at the dataset's tuned `n_lists` and `n_probe` {24, n95} ([IVF tuning](#ivf-tuning)); `silvertorch` torch (#12) and torch compiled (#13) at 24 on the same `n_lists`, goodreads + arxiv only; `postfilter` alpha {1, 8} | T2, C3, C6 |
| `deep` | goodreads, arxiv, yfcc10m, kept sweeps | `silvertorch` triton clause + bloom, official bloom: `n_lists` per dataset (goodreads {1024, 4096}, arxiv {1664, 8192}, yfcc10m {4096, 16384}) × `n_probe` {8, 16, 32, 64, 128}; V3 `candidate_pool_frac` {0.005, 0.01, 0.02, 0.05, 0.1} | F3, C6 |
| `synth` | goodreads-, arxiv-, yfcc10m-synth (uniform) and arxiv-corr-synth (cluster-correlated, `c001` `c003` `c01`, d128, arxiv-synth's `n_lists` so the two curves pair) ([datasets](datasets.md#synthetic-selectivity-attrs)); sized by claim (SYNTH-TRIM, 2026-10-10): seed 0 everywhere but goodreads-synth's first seven rates (seeds 0-2, the pilot's cells) | **arxiv-synth** (`p0001 p001 p005 p01 p02 p05 p1`, 46 cells): V1, V2 triton clause; `silvertorch` triton clause `n_probe` {24, 64, 256, 1024}; triton and official bloom `n_probe` 24 at `p001`, `p1`. **yfcc10m-synth** (`p001 p01 p02 p05 p1`, k 100, 25 cells): V1, V2 triton clause; `silvertorch` triton clause {24, 256, 1024}. **arxiv-corr-synth** (42 cells): V3 `candidate_pool_frac` {0.01, 0.05}; `silvertorch` triton clause {24, 64, 128, 256, 512, 1024}; `postfilter` {1, 8}. **goodreads-synth** (all ten rates, 558 cells): on the first seven, V1, V2 triton clause + bloom, V3, `silvertorch` triton clause {24, 64, 128, 256, 512, 1024} and triton + official bloom {24, 256}, `postfilter`, V1 / V2 torch eager and compiled on `p001, p01, p1`; on `p005 p02 p05` V1, V2 clause and the SilverTorch clause sweep. Every SilverTorch arm at its real dataset's tuned `n_lists`. k 1000 is dropped where N·p < 4000 (goodreads `p0001`, `p0003`; arxiv `p0001`) | F1, F2, C1, C6 (C2 on corr) |
| `codesign` | arxiv (kept sweeps), goodreads (`c0_genre, c2_format, c3_year`) | official and triton bloom, `bloom_path` {partial, full} × `n_probe` {8, 32, 128}, `n_lists` arxiv 1664 / goodreads 1024, k 100 | F4b, C5 |
| `bloomwidth` | goodreads `c0_genre`, arxiv kept, pubmed `c0_mesh` | `silvertorch` triton bloom `m_bits` {64 … 2048} × `k_hash` {3, 5}; official bloom `k_hash` {3, 5} (its width is `OfficialConfig.b_multiplier`, not `m_bits`); bs 16; quality only (`perf: false`) | F4a, C4 |
| `bloomwidth-timed` | the same | the same widths at `k_hash` 5 (official: its one width), k 100, bs 16, timed | F4a |
| `v3bits` | goodreads-synth (its first 7 rates), goodreads (`filter`'s kept sweeps); pubmed d768 (`filter`'s kept sweeps, clause only) | V3 triton only, `candidate_pool_frac` {0.01, 0.05}, seeds 0-2, bs {1, 16}, k {100, 1000} (synth's `ks_by_sweep`); goodreads `k_bits` {64, 128}, clause + bloom; pubmed `k_bits` {256, 768} (256 divides 768 and sits below LiNR's 512; 768 is the default, the comparison), clause only (bloom adds false-positive noise to a bits question), one arm per side so the goodreads keys are unchanged. LiNR's 512 bits at d128 would need a library change (declined) | C2 (V-V3BITS, V3-BITS-PUBMED) |
| `router` | goodreads (kept sweeps; arXiv gets the fitted threshold afterwards); pubmed d768 (V-ROUTER PubMed, the keep/kill Pareto test: clause kept sweeps, k 100 via `ks_by_sweep`, `lq_threshold` {0.05 (goodreads' fit), 0.2}, and in the same leg the IVF curve SilverTorch 4096 at `n_probe` {24, 64, 256, 1024} plus exact V1 and V2; its own arms, so goodreads' keys are unchanged) | `router` triton, `pre_n_probe` 8, `lq_threshold` {0.02, 0.05, 0.1, 0.2}, IVF branch `n_lists` 4096 / `n_probe` 24; its branches V2 triton and SilverTorch triton (4096 / 24) beside it; clause + bloom, seeds 0-2, bs {1, 16}, k {100, 1000} | F2 / T2 practical take (V-ROUTER) |
| `h2h` | goodreads `c0_genre`, arxiv `c0_maincat`, `none` + `bloom`, d128 | `silvertorch` triton and official with `score_path` {fp16, int32}, `n_probe` 24, bs {1, 16}, k {100, 1000}, seeds {0 … 4} (the repeats); one interleave group of the three arms; run with `--interleave --profile` | T3, C7 (H2H-final) |

### IVF tuning

SilverTorch's `n_lists` and its matched-recall `n_probe` depend on the dataset and are tuned, not
the library default 1024 ([decisions](../decisions.md#campaign-v2-user-2026-10-08), *IVF tuned
per dataset size*). **n95** is the smallest `n_probe` reaching `recall_oracle@100 ≥ 0.95` on a
dataset's median kept sweep. IVF-TUNE measures it quality-only (seed 0, bs 16, `n_probe`
doubling, capped at `n_lists` / 4 = 25 % of the items scanned, user) on goodreads and PubMed, per
`n_lists`, and picks the point that scans the fewest items (`n_probe` · N / `n_lists` + `n_lists`). The scripts are in
[`docs/artifacts/campaign-v2/ivf-tune/`](../artifacts/campaign-v2/ivf-tune/README.md). The tuning
records are artifacts in their own results tree, never campaign cells. `n_lists` starts at ≈ 4 √N to a power of two
(goodreads 4096, YFCC and PubMed 16384, then 4096 by the cap below); arXiv was measured at 2048, 4096 and 8192, which reach
0.95 at the same ≈ 12.7 % scanned (n95 = `n_lists` / 8), so it takes the fewest-items 2048 / 256.
YFCC (`tags_and`, pass 0.0185; 0.72 at the cap at 16384) and PubMed (`all5`, 0.018; 0.873 at
the cap at 4096) do not reach 0.95 within the 25 % cap. Their slot is the cap at `n_lists` 4096,
`n_probe` 1024: both list counts scan the same 25 % there (ties go to the smaller). The choice
predates ST-IDS, which lifted the probe scorers' `n_probe` ≤ 1024 at k 1000 limit ([kernels](kernels.md)). yfcc10m-synth follows at 4096. An n95 is written only where it was measured. Each value is one
`datasets:` slot per arm: `filter`'s SilverTorch arms get `{build: {n_lists: [L]}, query:
{n_probe: [24, n95]}}`, and `synth`'s get `n_lists` only, because it sweeps `n_probe`
explicitly (user, 2026-10-08). A slot left `{}` runs at the library default `n_lists` with
`n_probe` 24 only. n95 / `n_lists` depends on the pass rate as well as N: goodreads `c0_genre`
(0.33) needs `n_lists` / 64, and PubMed `all5` (0.018) needs more than `n_lists` / 16.

`codesign` is the S9 ablation of the bloom path, on both backends.
- **official:** `partial` (fused partial masks over the probed clusters,
  the library default, which every `filter` / `deep` official cell runs
  with the key absent) against `full` (a full-N bool mask, then IVF).
- **triton** (C5-OURS): our fused scorer against `bloom_full_mask`, then
  the same scorer. This separates the co-design idea from Meta's
  implementation of it.

`--interleave` pairs partial and full of one backend per group. Every
cell carries `bloom_path` in `params`, so no key collides with the
`filter` / `deep` cells; latency and `peak_fwd_mib` are the compared fields.
`bloomwidth` declares `perf: false`: `load_matrix` sets
`Job.timed = False`, `run` skips perf on their jobs without being asked, and
their records are `ok` (a `--skip-perf` record of a timed suite is `partial`,
because it lacks what the suite asked for; `--skip-perf` or `--mode` on an
untimed suite drops nothing).

```yaml
# config/<dataset>.yaml — one per dataset; every string may carry {dim}
data_dir: data/goodreads-work-id
checkpoint: data/goodreads-work-id/checkpoints/sasrec-ssm-logq-d{dim}/best_model.pt
#   or, for pre-encoded text datasets, a per-dim mapping instead of `checkpoint`:
#   content_dir: {64: content_d64, 128: content_d128, 256: content}   # relative to data_dir
dims: [64, 128, 256]
encode: {batch_size: 512, num_workers: 8, max_seq_length: 200}       # SASRec datasets only
users_limit: 10000                                                    # or null
filters:                                                              # optional
  attrs: item_attrs_narrow.pt                                         # relative to data_dir
  reverse: clause_is_reverse_narrow.pt                                # optional
  # query_attrs: query_attrs_synth.pt  # optional .pt [U_full, C]; default eval_split.parquet
  clause: {c0_genre: [0], c1_lang_reverse: [1], all4: [0, 1, 2, 3]}   # name: active clauses
  bloom: {c0_genre: [0], old_sweep: {clauses: [2], disabled: true}}   # long form: disabled
```

`gt_dir` is derived (`<data_dir>/gt_d{dim}`). Unknown keys
raise `ConfigError` naming the file.

```yaml
# config/suites.yaml — a suite = cells run on every listed dataset × the dims both list
filter:
  datasets: [goodreads, arxiv, yfcc10m, pubmed]
  dims: [128, 192, 768]             # optional; default: the dataset's dims
  filter_kinds: [clause, bloom]     # none | clause | bloom
  ks: [100, 1000]
  batch_sizes: [1, 16]
  seeds: [0, 1, 2]                  # optional; default [0]
  perf: true                        # optional; false = a quality-only suite (bloomwidth)
  interleave:                       # optional: comparison groups timed round-robin (--interleave)
    - {by: algo, values: [linr_v1_filter_mask, linr_v2]}   # `values` limits who joins
    - {by: [backend, score_path]}   # algo, backend or build params
  sweeps: {goodreads: [c0_genre, c1_lang_reverse, all4], ...}   # optional per-dataset selector
  ks_by_sweep: {goodreads-synth: {p0001: [100]}}                # optional per-(dataset, sweep) ks
  seeds_by_sweep: {arxiv-synth: {p001: [0, 1, 2]}}              # optional per-(dataset, sweep) seeds, replacing `seeds`
  arms:
    - {algo: linr_v1_filter_mask, backends: [triton]}
    - algo: silvertorch
      backends: [official]
      filter_kinds: [bloom]          # optional: a subset of the suite's
      sweeps: [c0_genre]             # optional: a subset of the dataset's (and the selector's)
      build: {n_lists: [1024]}       # dict-of-lists = grid, list-of-dicts = explicit combos
      query: {n_probe: [24]}
      datasets:                      # optional: only these datasets, each with overrides
        goodreads: {query: {n_probe: [24, 37]}}   # merged key by key over the arm's grids
        arxiv: {}
    - {algo: silvertorch, backends: [torch], build: {compile: [max-autotune]}}
  # bloom: {...}                    # optional per-suite override of the top-level default
bloom: {m_bits: 1024, k_hash: 5}
```

An arm is one algo on one or more backends. Its `build:` params rebuild
the index. Its `query:` params (`n_probe`, `candidate_pool`,
`candidate_pool_frac`, `alpha`, the `QUERY_PARAMS` set) are applied with
`set_query_params` to the built index, so `deep` is two k-means per
`(dataset, sweep, seed)`, not ten. Putting a query param under `build:`
(or the reverse) is a `ConfigError`. So is a param on an arm where it has
no meaning: `bloom_path` off silvertorch / bloom / official, `score_path` off silvertorch / official, `m_bits` /
`k_hash` off silvertorch / bloom ([bloom widths](#bloom-widths-as-build-params)),
`compile` off a torch arm ([compiled arms](#compiled-arms)),
`candidate_pool_frac` off `linr_v3` ([pool fractions](#pool-fractions)), and `k_bits` off
`linr_v3`. `k_bits` is a harness build param: `LiNRV3` takes none, so `algos.build`
replaces its stage 1 with `OneBitKNN(k_bits=...)` before `register_index`; absent, stage 1
keeps the library default `k_bits = D` (V-V3BITS). Two arms that expand to the same
cell are also an error. When two backends of one algo run the same code
path (`PATHS`) over the same cells, they collapse to the first, logged
once. A collapse that would cover only part of a job's cells is a
`ConfigError`. CLI narrows (`dims`, `algos`, `backends`, `filter_kinds`,
`sweeps`, `seeds`, `ks`, `batch_sizes`) filter the lists *before* the
`PATHS` collapse. `ks` / `batch_sizes` are replacements, not selections:
they set `Job.narrowed` when they differ from the job's own lists (the
suite's, or its `ks_by_sweep` entry), and `run` then records `partial`.

`load_matrix(dataset_yaml, suites_yaml, suite, **narrows)` returns `Job`s
grouped by `Job.group == (dataset, dim, algo, backend)` — the campaign's
process boundary — in the order `dim → algo → backend → arm →
filter_kind → sweep → build → seed`. A `Job` is one build, with its own
`ks` (the suite's or its `ks_by_sweep` entry) and `bloom` (the suite's
default, or the arm's gridded widths): `dataset, dim, suite,
filter_kind, sweep, clauses, algo, backend, path, build, query, ks,
batch_sizes, seed, bloom, data: Dataset, narrowed, timed`; `job.cells()` lists the
`params = build | query` of each cell and `job.key(params)` is the
record's key block. `none` cells have `sweep == "full_scan"` and
`clauses is None`.

## CLI

```
bench run      --dataset D --suite S [--dim N]* [--algo A]* [--backend B]* [--filter-kind K]*
               [--sweep W]* [--k N]* [--bs N]* [--seed N]* [--mode eager|graph]*
               [--skip-quality] [--skip-perf] [--profile] [--interleave] [--out results] [--output FILE]
               [--resume|--force] [--config-dir config] [--checkpoint PATH]
bench campaign --suite filter|deep|codesign|…|all [--dataset D]* [--dim N]* [--mode M]*
               [--skip-quality] [--skip-perf] [--profile] [--interleave] [--out results] [--resume|--force]
               [--config-dir config] [--timeout 48.0]
bench oracle   --dataset D --suite S [--dim N]* [--sweep W]* [--config-dir config]   # prebuild blobs
bench check    --dataset D [--dim N]* [--config-dir config]   # eval_datasets.layout.validate_layout
bench upload   [--repo-id user/repo] [--results DIR] [--path-in-repo PREFIX] [--gate STEP]
               [--private|--public] [--verify] [--dry-run]
bench fetch    --path-in-repo PREFIX [--repo-id user/repo] [--results DIR]
bench report   [results] [--out DIR] [--gate STEP] [--manifest FILE] [--only NAME]*
               [--dim 128] [--k 100] [--bs 1] [--mode eager|graph] [--backend triton]
bench env                                                     # provenance | clocks, as JSON
```

`*` = repeatable. `bench run` expands one `(dataset, suite)` through
`load_matrix` (every repeatable option is a narrow) and runs the cells in
*this* process; `--resume` (the default) skips cells already `ok` at the
current `code_version`, `--force` re-runs them (the earlier records stay in
the file). `--mode` defaults to both; `--output` overrides the per-`(suite,
dataset, dim)` file. `--k`, `--bs` and `--mode` *replace* the suite's
lists rather than select cells, so a run with any of them (or a
`--skip-*` flag) writes `status: partial` records — resume re-runs them,
and a later full run never counts a narrowed cell as done. The exit
code is 1 when any cell failed, and 1 with a message when the narrows
select zero cells (a `--sweep` typo is an error, not an empty success).
`--checkpoint PATH` (`{dim}` templated) replaces a sequential dataset's
`checkpoint` for this run and is refused on a `content_dir` dataset; the
key's `inputs` follows it, so the pinned cells key and report apart from
the config's encoder. It exists for the golden cells, which stay on
`gsasrec-d128-drop0.5-id` ([How to run](#how-to-run)).

`bench campaign` is the process loop: for every suite (in the
order of `SUITES = ("filter", "deep", "codesign")` for `all`), every listed dataset and every
`(dataset, dim, algo, backend)` group of `load_matrix`, one child process
`python -m bench.cli run --dataset … --dim … --suite … --algo …
--backend … --resume` (same interpreter, `cwd = evaluation/`), sequential.
With `--interleave` the groups one [interleave unit](#interleaved-groups)
spans share a child (`cli._children`: connected groups of `config.interleave_units`),
run as `--algo a … --backend b … --interleave`; the set has to be exactly one
`--algo` × `--backend` product, else the campaign stops with the groups named,
and the log name joins them with `+` (`codesign_arxiv-d128_silvertorch_official.log`,
`filter_arxiv-d128_linr_v1_filter_mask+linr_v2_triton.log`).
The child's stdout + stderr go to
`results/_logs/<suite>_<dataset>-d<dim>_<algo>_<backend>.log` (appended, the
command line first, then `nvidia-smi -q -d CLOCK` under `=== clocks at start`,
the child's output with its closing under-load clock histogram, and the clock
block again under `=== clocks at end`); one summary line per child (`time suite dataset dim
algo backend rc seconds log`) goes to `results/_logs/campaign.log` and the
terminal; a non-zero rc is recorded and the loop continues; a child
still running after `--timeout` hours (default 48, `cli.TIMEOUT_H`: sized
for the largest group, arxiv `deep` `silvertorch` at ≈ 10 h, so it catches
hangs rather than budgeting work) is killed and recorded
as `rc=timeout` (exit code 124, noted in its log); the exit code
is the worst child rc, or 1 when a listed dataset expands to no groups or
no child was launched at all (`--dataset` / `--dim` selecting nothing).
After the last child the campaign aggregates the tree into
`results/results.parquet` (`records.aggregate`, logged as `=== aggregated`).
Before each child the campaign deletes every entry of `results/_parity/`
but the child's own `(dataset, dim, algo)` subdirectory — decided from
what is on disk, not from the loop, so a restart mid-group keeps the
killed process's spill — and deletes the whole directory after each
dataset. A backend is the thing under test, so it gets
the process: no dynamo cache, allocator arena or CUDA-graph pool outlives
it, at the cost of a dataset reload and a CUDA context init per group.
`bench oracle --dataset D --suite S` builds (or finds) the [oracle
blob](#oracle-blob-v4) of every filter sweep the suite's jobs read, one per
`(dim, sweep, k_max)`, in its own process, with no module and no timing
(`run.prebuild_oracles`); it exits 1 when the narrows select no filter sweep.
Run it before a campaign: `bench run` still builds a missing blob, but warns
that it should have been prebuilt.
`bench check --dataset D` runs `eval_datasets.layout.validate_layout` on
the dataset's directory at every dim (files, prefix sidecars, the row
alignments the readers enforce) and exits 1 on any problem — run it before
a campaign on a freshly staged dataset.
`bench env` prints `measure.provenance() | measure.clocks()` as one JSON
object — the provenance fields of the record's `env` block plus one
`nvidia-smi` sample under `clocks()`'s names (`sm_mhz`, `mem_mhz`,
`sm_max_mhz`, `power_limit_w`) — to paste into a
[validation](../validation.md) record next to a number measured outside the
harness.

### Inductor cache

`bench run` sets
`TORCHINDUCTOR_CACHE_DIR` to `<base>/<code_version>` (`:` → `-`;
`measure.inductor_cache_dir`) and prints it, so a `graph`-mode run
after a library edit compiles afresh instead of replaying a stale
`@triton_op` body ([testing](testing.md#running)). `<base>` is the
`TORCHINDUCTOR_CACHE_DIR` the caller set (the pods' per-job
`/scratch/inductor/<job>`), else `<tmp>/bench-inductor`: a caller's dir
reused across library edits gets one subdirectory per tree and never
serves another tree's graphs (H-INDCACHE). The caller's value is read in
`bench/__init__.py`, before any `bench` module imports `torch._inductor`,
whose import writes torch's default into the environment; for the same
reason `bench campaign` gives its children the caller's value (or none),
not its own environment's; each child appends its own `code_version`.

## The cell loop (`run.py`)

`run(jobs, *, out_dir, out_path=None, resume=True, modes=MODES,
skip_quality=False, skip_perf=False, profile=False, env_extra=None,
latency_kw=None, device=None) -> Counter` runs the jobs in order:

1. once: `measure.setup(seed)`, `warm_gpu_once`, `provenance()` (+
   `env_extra`, the CLI's `config_sha`), the idle `clocks()` sample
   (`env.sm_mhz_idle` and the memory clock / power limit);
2. per `(dataset, dim)`: `inputs.load_inputs` (with attrs when any job of
   the group is a filter cell); the previous inputs are freed first;
3. per `(filter_kind, sweep, filter backend, k_max, bloom params)`:
   `sweep_qa` → `build_filters` → on filter cells `exact_filter` +
   `oracle.load_or_build` (blob v4 at `k_max = max(ks)`) and on bloom cells
   `pass_counts` → `bloom_fp_rate`; the row masks: `keep` (not
   skip-masked), `oracle_rows` (kept with ≥ 1 survivor), `heldout_rows` (kept, ≥ 1 target, and on filter cells
   `target_in_filter`);
4. per job: `measure.setup(job.seed)`, one `timed_build` of
   `algos.build(...)` with the first cell's params; `index_mib`,
   `filter_mib` (the module's `filter` submodule, 0 without one); a build
   failure writes a `failed` record per cell of the job;
5. per cell: `set_query_params` for the query part of `params`; quality
   (`module.k = k_max`, chunks of 16 kept rows, `accumulate(...,
   ranked=True)` against the oracle prefix for `oracle_rows`,
   `accumulate(..., targets, n_targets)` for `heldout_rows` — on filter
   cells the targets the exact mask excludes are set to `-1` first and
   `n_targets` counts only the reachable ones (`blob["targets_in_filter"]`),
   since no algo can retrieve a masked-out item; row selection
   by CPU masks + `index_select`, so no per-chunk sync; `PER_K_QUALITY`
   algos rerun the pass at each other `k`, `module.k = k`, and their
   metrics at that `k` replace the prefix's). A side that scored
   no row (`n == 0`: pubmed's `c3_journal_reverse`, where no kept query has
   an in-filter held-out target) has no mean: every metric is `null`, not
   0.0 (`metrics.null_if_empty`, applied by `finalize`; the exact-algo
   oracle gate skips a `null` recall); the parity spill;
   the exact-algo gate; perf per `(bs, k, mode)` (`query_pool` per bs,
   `module.k = k`, `graph_callable` or a null entry with `reason`,
   `latency`, optional `profile_once`); `clocks_drift` over the perf
   entries' under-load `sm_mhz` samples; `memory_reserved_mib`; one
   `records.append_record`; the samples sidecar; `torch._dynamo.reset()`
   per bs;
6. after each job: drop the module, `gc.collect()`, `torch._dynamo.reset()`,
   `empty_cache()`.

Any exception inside a cell (an OOM on the torch path included) becomes a
`status: failed` record with the traceback and `stage` (`build`,
`query_params`, `quality`, `perf`) and the loop continues. Three things
stop the process: `KeyboardInterrupt`; `QualityGateError` — an exact
algo (`linr_v1_filter_mask`, `linr_v2`) below `recall_oracle@k_max ≥ 0.99`
— which is recorded first; and a sticky CUDA error (`run.STICKY_CUDA`:
`CUDA error`, `illegal memory access`, `device-side assert` in the
message), also recorded first and then re-raised, because the context is
dead and every later cell would fail in seconds with the same traceback.
In a campaign either of the last two ends the *child*, i.e. the
`(dataset, dim, algo, backend)` group; the loop continues with the next
group and `--resume` re-runs the un-run cells later.

Crash safety: the oracle blob, the encode cache and the parity file are
written through `eval_datasets.layout.atomic_write` (`<path>.tmp` + fsync +
`os.replace`), so a crash mid-save leaves the previous file or nothing, never
a torn one, and an unreadable file at the oracle's fingerprint path is
rebuilt with a warning. Each JSONL line is one `write` + `fsync`; the
samples line is appended *before* its record, so a crash between the two
cannot leave a resumable record without its vector; `read_keys` ignores
(and logs) one torn trailing line — the cell in flight when the process
died — and raises on a malformed line anywhere else.

### Interleaved groups

A suite's `interleave:` list names comparison groups: `by` is the field the
arms differ in (`algo`, `backend` or build params such as `bloom_path`,
`score_path`; one name or a list), `values` (one `by` field only) limits which
arms join. `config.interleave_units(jobs)` puts the jobs whose key is equal but
for the `by` fields (query params aside, the seed included, so a group is per
seed) in one unit, in matrix order; every other job is a unit of its own. A
job in two groups of two or more is a `ConfigError` at `load_matrix`. The
shipped groups: `filter` and `synth` V1 vs V2 (per backend and compile mode)
and triton vs official (silvertorch bloom); `codesign` partial vs full; `h2h`
triton vs official fp16 vs official int32. `bloomwidth-timed` has none, and `by: backend` could not
pair its arms: Triton's build carries `m_bits`, official's does not, so their keys differ
beyond `backend`.

Without `--interleave` nothing changes. With it, `run` builds every arm of a
unit (assets per arm's step-3 key, usually shared), then per query combo runs
each arm's quality in order and times all arms together: per `(bs, k, mode)`
one `measure.latency_group` call warms and calibrates each arm (its own `N`)
and then runs window `i` of every arm before window `i + 1` (A, B, A, B, A, B
over the 3 windows), the method of
[h2h.py](../artifacts/kernel-opt/h2h.py) and
[interleave.sh](../artifacts/kernel-opt/interleave.sh) inside one process. In
`graph` mode dynamo is reset once before the variant, then each arm is
captured (`graph_callable` no longer resets, so captures coexist); an arm that
cannot capture (official) gets its null `not_capturable` entry and the others
run round-robin without it. Each arm writes its own record, under the key a
non-interleaved run gives it, with `interleave: {group, arms, position}`:
`group` is the first 16 hex of the sha1 of the cell's key block minus the `by`
fields (`config.shared_key`; the seed and query params included), `arms` the
labels in run order (`algo/backend` plus each `by` build param, e.g.
`silvertorch/official/score_path=int32`), `position` this arm's index. Its
measured perf entries carry `rounds` (the window count, equal across the
group), and window `i` of every arm is round `i`, which is what the report's
paired ratios read. Resume works per arm, but a group whose arms are not all
`ok` re-runs whole. A failure in one arm's quality drops that arm from the
perf round-robin; a perf failure fails every arm of that cell. `elapsed_s`
of an interleaved record is the whole group cell's wall time.

### Quality cache

The seed only moves the perf pool of `linr_v1_filter_mask`, `linr_v2` and
`postfilter` on any backend, compiled or not (`run.SEED_FREE_QUALITY`): their
quality is the same at every seed. Such a cell copies the quality of an
`ok` / `partial` record of the same JSONL whose key differs only in `seed`, at
the same `code_version` and `ks`, that computed its quality itself (the lowest
such seed other than its own; `run.cached_quality`). It takes that record's
`quality` and `per_query` and records `quality_source: {seed, code_version}`;
perf still runs at its own seed. The parity spill and the exact-algo gate are
not redone on a copied cell. SilverTorch (k-means) and `linr_v3` (OPORP
projection) always recompute. The seed changes SilverTorch's quality, but not
`linr_v3`'s at its default `k_bits = D`. There the OPORP projection is a signed
permutation, so the seed moves the bits but not the Hamming scores
([kernels](kernels.md#oporp-layout)): V3's quality is the same at every seed, and
its three seeds measure perf-pool variance only. Below `D` (the `v3bits` suite's
`k_bits` 64) OPORP bins dimensions together, so there the seed does move V3's quality. The records are the cache: `run`
indexes the file's computed records when it first opens it (`quality_sources`) and
adds each new one. Every record carries `seed_scope`, `pool` for the seed-free arms
and `pool+build` for the others. For `linr_v3` at `k_bits = D`, `pool+build`
overstates the seed's reach. It stays as recorded, because the campaign freezes
behaviour.

### Bloom widths as build params

`m_bits` / `k_hash` in an arm's `build:` grid (silvertorch / bloom only,
else a `ConfigError`) become that job's `bloom`. It feeds the module's
index, the standalone filter of step 3 and so the bloom pass counts and
`bloom_fp_rate`, the record's `bloom`, and `index_mib` / `filter_mib`. The
step-3 asset key carries the bloom, so each width builds its own filter.
The exact oracle does not depend on the width and is shared. A cell that
does not grid them keeps the suite default outside `params`, so its key is
byte-for-byte the pre-grid one. The `bloomwidth` suites grid them. On
`official`, `SilverTorch` ignores `m_bits`: the official index's width is
`OfficialConfig.b_multiplier`, and `k_hash` is its search `k`. An official
cell's `bloom_fp_rate` is the standalone Triton filter's at the job's
`m_bits`, not the official index's.

### Compiled arms

`compile: max-autotune` (a build param, torch arms only) is the #13 arm:
`algos.compile_module` calls `nn.Module.compile(mode=…)` on the built
module in place, so `k`, `set_query_params` and the buffers stay the
module's own. Compilation is lazy, so `build_s` does not include it.
`run.compile_warmup` times the first forward on one quality chunk as the
record's `compile_s`. `perf` resets dynamo per `(bs, k)`, so each variant
recompiles inside `latency`'s 50-call warm-up, never in a timed window.
`max-autotune` already replays inductor's own CUDA graphs, so the module is
marked uncapturable: `graph` is a null entry with `reason:
not_capturable`, the `eager` entry is the compiled path's latency, and an
`--mode eager` run of it is not `partial`. Its outputs live in
CUDA-graph buffers that the next call overwrites, so `quality` clones each
chunk's ids and scores before keeping them for the parity spill. Torch
raises on a read of an overwritten output, so a missed clone fails loudly.
The timed calls discard their outputs. `SilverTorch.capturable` is a
read-only property, so the override is a per-instance subclass of the same
name. A compile failure surfaces at `compile_warmup`, inside the build
step, and is recorded as a `build` failure. There is no bit-exact gate
between eager and compiled.

### Pool fractions

`candidate_pool_frac` (a `linr_v3` query param, in the key) is resolved
per sweep by `run.resolve_pool`, before the build and before
`set_query_params`, to `candidate_pool = max(POOL_MIN = 2000, round(frac
× mean pass count over the sweep's oracle rows))`. On uniform synth every
query passes the same items, so the mean is the achieved pass count (N·p in
expectation); on arxiv-corr-synth it is the mean of the queries' cluster sizes. The key keeps the fraction, the module gets the int, and
the record's `candidate_pool` carries it.

## Output: one JSONL record per cell

`results/<suite>/<dataset>-d<dim>.jsonl`, appended by the process the
moment a cell finishes (`json.dumps` of one line, `allow_nan=False` — NaN
and ±inf become `null` — then `write` + `fsync`). Nested, not wide; the flat
form is `results.parquet` (below). The tree is local working state —
gitignored, read by resume, kept on the Hub once a leg finishes
([Results storage](#results-storage)).

| field | type | value |
|---|---|---|
| `schema_version` | int | `4`; schema 3 lacks `seed_scope`, `quality_source`, `per_query`, `interleave`, the perf entries' `ids_sha256` / `ids_sha256_canon` / `rounds` and `env.frac_windows_below_max` (all read as null); schema 2 also lacks `inputs` (derived on read, [Resume](#resume)); schema 1 also lacks the under-load clock fields below (`env.sm_mhz` instead) |
| `status` | str | `ok`, `partial` (the record does not carry everything the suite asked for), `failed` |
| `partial_reasons` | list / null | why `partial`: any of `skip_quality`, `skip_perf`, `modes` (a `--mode` subset that drops a mode this job's module can run: an eager-only run stamps it on a capturable module, not on one with `capturable = False` such as `official`, whose `graph` entry would be `not_capturable` anyway; decided per job after the build), `ks_bs` (`--k` / `--bs` replaced the suite's lists — `Job.narrowed`) |
| `dataset`, `dim`, `inputs`, `suite`, `filter_kind`, `sweep`, `algo`, `backend`, `params`, `seed` | | the key block = `Job.key(params)` (`KEY_FIELDS`); `params` is the native dict of build + query params (`{}` when the algo takes none); `inputs` is the input identity below |
| `path` | str | `PATHS[(algo, filter_kind, backend)]` |
| `n_items`, `n_queries` | int | catalogue size, queries after `users_limit` |
| `n_kept` | int | queries not skip-masked (every query on `none` cells) |
| `n_queries_oracle` | int / null | kept queries with ≥ 1 survivor (the oracle metrics' `n`); `null` on `none` cells |
| `n_queries_heldout` | int | queries the held-out metrics cover (kept, ≥ 1 target, `target_in_filter` on filter cells) |
| `n_targets_in_filter` | int | held-out targets those queries are scored against: every valid target on `none` cells, only the ones the exact mask admits on filter cells |
| `pass_rate` | float | exact mask pass rate over kept queries (`1.0` on `none`) |
| `bloom_fp_rate` | float / null | mean per-query `(bloom − exact) / (N − exact)`; bloom cells only |
| `bloom` | dict / null | `{m_bits, k_hash}` on bloom cells: the suite default, or the cell's own when `params` grids them ([bloom widths](#bloom-widths-as-build-params)) |
| `k_max`, `ks`, `batch_sizes` | | the suite's, `k_max = max(ks)` |
| `build_s` | float | construction + `register_index`, sync on each side (same value on every cell of one build) |
| `compile_s` | float / null | a `compile` arm's first forward (one quality chunk), apart from `build_s` ([compiled arms](#compiled-arms)); `null` elsewhere |
| `candidate_pool` | int / null | the `candidate_pool` the module ran with when `params` sets one: the literal, or the value `candidate_pool_frac` resolved to ([pool fractions](#pool-fractions)) |
| `index_mib` | float | Σ buffers of the algo module, filter submodule included |
| `filter_mib` | float | Σ buffers of the filter submodule alone (`0.0` without one; `silvertorch` carries its attrs inside `index_mib`) |
| `seed_scope` | str | `pool` on the seed-free arms (`SEED_FREE_QUALITY`: the seed moves only the perf pool), `pool+build` on SilverTorch and `linr_v3`; for `linr_v3` at `k_bits = D` the label overstates the seed's reach, because its quality is seed-free ([Quality cache](#quality-cache)) |
| `interleave` | dict / null | `{group, arms, position}` when the cell ran in an [interleave group](#interleaved-groups); `null` otherwise |
| `quality_source` | dict / null | `null` when this record computed its quality; `{seed, code_version}` of the record the [quality cache](#quality-cache) copied it from |
| `per_query` | str / null | the [per-query sidecar](#per-query-sidecar) relative to the results root; `null` without quality |
| `quality` | dict / null | `heldout: {recall@k, ndcg@k, precision@k, mrr@k for k in ks, n}`, each metric `null` when `n == 0`; on filter cells also `oracle: {…}` (ranked-prefix targets); `jaccard_vs_first@k` per `k`, `score_max_abs_diff`, `parity` (`reference` = this record wrote the spill file, `vs_<backend>` = compared against it, `shape_mismatch:…`); `null` with `--skip-quality` |
| `perf` | list / null | one entry per `(bs, k, mode)` (table below); `null` with `--skip-perf` and on a quality-only suite (`perf: false`) |
| `unstable` | bool | any perf entry `unstable` (window spread > 5 %), or `clocks_drift` |
| `memory_reserved_mib` | float / null | `torch.cuda.memory_reserved()` after the cell — the leak detector across a group's cells |
| `elapsed_s` | float | wall time of the cell |
| `env` | dict | `gpu, driver, cuda, torch, triton, official_commit, commit, dirty, repo_dirty, git_branch, code_version, lib_dir, host, python, started, config_sha`; the process-start sample `sm_mhz_idle, mem_mhz, sm_max_mhz, power_limit_w`; the cell's `sm_mhz_load` (median of its perf entries' under-load samples; `null` without perf or CUDA), `clocks_drift` (any under-load sample > 5 % from the process's first) and `frac_windows_below_max` (the share of the cell's `window_sm_mhz` samples below `sm_max_mhz`, the device's max SM clock; `null` without samples or without a device max) |
| `stage`, `error` | str | `failed` records only: where it died and the traceback |

Perf entry:

| key | value |
|---|---|
| `k`, `bs`, `mode` | the variant; `mode ∈ {eager, graph}` |
| `n`, `median_ms`, `mean_ms`, `p95_ms`, `p99_ms`, `min_ms`, `iqr_ms` | of the chosen window (median of the three window medians); quantiles linear-interpolated |
| `trimmed_mean_ms` | mean of that window after dropping `n // 10` calls from each end |
| `qps` | `n · bs / wall_s` of that window (closed-loop, one client) |
| `host_gap_ms` | `wall / n − mean_ms` |
| `outliers_std`, `outliers_tukey` | counts beyond 3 σ / the 1.5 IQR fences, never dropped |
| `spread`, `unstable` | `(max − min) / median` of the three window medians; `> 0.05` |
| `window_medians_ms` | the three medians |
| `rounds` | interleaved records only: the window count; window `i` of every arm of the group is round `i` |
| `window_sm_mhz` | the SM clock sampled right after each window's sync, same order as `window_medians_ms`; `null` elements without CUDA; not in `results.parquet` |
| `peak_fwd_mib` | eager only, first window: `max_memory_allocated − allocated_before` |
| `sm_mhz` | `window_sm_mhz[-1]`: the SM clock sampled right after the last window's sync, with the GPU still at its load clock — the per-variant value cross-run latency comparisons read, and the only clock `clocks_drift` looks at; `null` without CUDA |
| `cache_plans` | on every entry: `false` on `silvertorch`/`official` (`run.perf` replaces `module.official` with `cache_plans=False` before the first variant, so every timed forward pays the CPU expression parse), `null` on backends without a plan cache |
| `load` | `"closed_loop"` |
| `kernels` | `--profile`, eager only: top-8 CUDA kernels `{kernel, us, calls}` |
| `kernels_us`, `kernels_calls` | `--profile`, eager only: device µs and kernel launches summed over every kernel of the call, sentinels and `## …` profiler ranges excluded (records before H-KSUM lack them; records between H-KSUM and its fix double-count compiled calls, which no eager `h2h` arm is; T3 reads these, and for such a record the top-8 sum, marked as a lower bound) |
| `kernel_scopes` | `--profile`, eager only: `{scope: {us, calls}}` for `scorer`, `topk`, `epilogue`, `other` (`measure.KERNEL_SCOPES`, by substring of the kernel name, first match wins), adding up to `kernels_us` / `kernels_calls`. `scorer` holds the like-for-like scoring scope: ours `_codesigned_probe_score*` (bloom test fused), Meta's `fused_kmean_ann::` (`process_cluster*`, payload and warp-size kernels) and `bloom_search::` kernels; `topk` the torch top-k / sort kernels (`mbtopk`, `sbtopk`, `radixSortKVInPlace`, `bitonicSortKVInPlace`); `epilogue` the index / gather / scatter kernels. In `results.parquet` as `perf_kernels_<scope>_us` / `_calls` (H-SCOPE; records before it lack them) |
| `ids_sha256` | sha256 of the ids the timed callee (the eager module or the graph replay) returns on the first 8 batches of that `(bs, seed)` pool (`run.IDS_PROBE_BATCHES`), int64 row-major, batch after batch, run once after the windows; `null` when the variant did not run. Equal eager and graph hashes are the D1-G identity gate. Graph mode is inductor's code, not the eager ops replayed, so this gate catches arithmetic that inductor lowers differently. Records before library tree `5d158f20` differ in graph `quantize_int8` ([kernels](kernels.md#quantize_int8-retrieveindexing), [validation](../validation.md#campaign-v2-phase-v-not-yet-validated), *V-GRAPH-IDS*) |
| `ids_sha256_canon` | the same ids with each row re-ordered by (score desc, id asc) before hashing: two backends with bit-equal scores whose tied ids come in another order hash equal (official int32 against Triton). `bench report`'s T3 identity column reads it; eager vs graph reads `ids_sha256`. An added field, schema stays 4 |
| `reason` | present when the variant could not run (`not_capturable`, `cuda_unavailable`, `cudagraph_skips=N`, `cudaGraphLaunch per call = N, expected 1`); every stat key is then `null` |

`results/<suite>/<dataset>-d<dim>.samples.jsonl` holds the per-call vector
of the chosen window: one line per perf entry, `{key block, k, bs, mode,
ms: [...]}`, written before the cell's record.

### Per-query sidecar

Every record that computed its quality points at one compressed npz,
`per_query` = `<suite>/<dataset>-d<dim>.perquery/<sha1(resume key)[:20]>.npz`
relative to the results root (`run.per_query_path`), written through
`atomic_write` before the record (`run.write_per_query`):

| array | dtype | value |
|---|---|---|
| `rows` | int32 | the query's index in the `users_limit`-trimmed query set; one entry per kept query |
| `pass_count` | int64 | items the exact mask passes for that query (the oracle blob's `pass_counts`); `-1` on `none` cells |
| `recall_oracle@{k}` | float32 | per-query recall against the oracle prefix, for every `k` in the record's `ks`; NaN where the row has no oracle (no survivor, or a `none` cell) |
| `heldout_recall@{k}` | float32 | per-query held-out recall over the reachable targets; NaN where the row has none |

The values are the per-row recalls `metrics.accumulate` sums into the
record (it returns them), so their mean over non-NaN rows is the record's
`recall@k` up to float32 summation order; a `PER_K_QUALITY` algo's `k` comes
from its run at that `k`. A [quality-cache](#quality-cache) copy points at
its source's file; a record without quality has `per_query: null`. A rerun
of the same resume key (`--force`) rewrites the same file. The directory
name has no `_` prefix, so `bench upload` publishes it and `bench fetch`
brings it back; it does not match `*/*.jsonl`, so `records.aggregate`
ignores it.

### Resume

The resume key is `records.resume_key(job.key(params), code_version)` —
canonical JSON of the key block plus `measure.code_version()`: the tree
hash of the *imported* `retrieve` package (`git rev-parse HEAD:./` run in
`Path(retrieve.__file__).parent`, i.e. in whichever checkout the venv's
editable install points at) when that subtree is clean, else
`files:<sha256>` over its `**/*.py` sources actually on disk (also the
value outside a git checkout; the two namespaces are disjoint). It is
computed in every `bench run` child and `bench oracle` process, never taken
from the checkout that launched `bench`: on the pods the shared venv
imports `/workspace/retrieve` while `bench` runs from a worktree, and a
fast-forward there mid-leg moves the next child's stamp with the code it
runs. `env.lib_dir` records the directory.

A kernel edit — committed or not — therefore invalidates
every cell; a doc or plan edit invalidates none. (D1 reruns only the arms
a fix changes, by narrow `bench run`s: its [code_version
policy](../validation.md#campaign-roadmap-d1-in-progress-not-yet-validated).)
`records.read_keys(path)` rebuilds the key from a record as
`resume_key(records.key_block(rec), rec["env"]["code_version"])` and keeps the last status per
key; a cell is skipped when that status is `ok`. `--force` runs everything
and appends. Resume reads the local JSONL only, never the network: a
campaign resumed on a box that lost its tree first restores it with
`bench fetch --path-in-repo <leg>` and then appends to the fetched files.

**The input identity (`inputs`).** The key block names what was encoded, not
only which dataset: `Dataset.inputs` is the checkpoint's directory name
(`sasrec-ssm-logq-d128`) on a sequential dataset and the resolved
`content_dir`'s name (`content_d128`; arxiv d256 is `content`) on a text
dataset. Without it a goodreads cell on the E1c checkpoint had the resume
key of its gSASRec record at the same `code_version`, so `--resume` skipped
it and `latest` kept one of the two. A name, not a digest: checkpoints and
content directories are published under immutable names (a retrained
encoder gets a new checkpoint directory), a digest would cost a pass over
GBs per process, and a pre-schema-3 record can only be given a name — its
files are not on the box that reads it. Content drift under one name is the
oracle fingerprint's job ([Oracle blob v4](#oracle-blob-v4)), not the key's.
A schema-2 record has no `inputs`; `records.inputs_of` derives it:
goodreads → `gsasrec-d{dim}-drop0.5-id` (what every pre-H2 goodreads record
ran on); a text dataset → today's `config/<dataset>.yaml` resolved at the
record's dim, so D1's arxiv / pubmed records keep exactly the resume key a
new run computes (they were written with the same `content_dir`); any other
checkpoint dataset raises (kuairand's E4 cell and yambda's retired suite
have no identity left). The derivation reads the repository's
`evaluation/config/`, not a run's `--config-dir`: it describes what D1's
records ran on, which is the checked-in config. `SCHEMA_VERSION` went 2 → 3 because the record
layout gained a key field.

`records.aggregate(results_dir)` writes `results.parquet` — `records.latest`
(the last record per key of every `<suite>/*.jsonl`), one row per perf entry
(one row with null perf columns when the record has none) — the one table
`report.py` reads. Columns, in order: the key block (`params` as canonical
JSON); the record scalars `status, path, n_items, n_queries, n_kept,
n_queries_heldout, n_queries_oracle, n_targets_in_filter, pass_rate,
bloom_fp_rate, k_max, build_s, index_mib, filter_mib, unstable,
memory_reserved_mib, elapsed_s, schema_version, stage, error,
partial_reasons` (a list), `seed_scope, per_query`; `env_code_version,
env_commit, env_dirty, env_gpu, env_sm_mhz_load, env_clocks_drift,
env_git_branch, env_frac_windows_below_max`; `quality_source_seed`,
`interleave_group`, `interleave_position` (null on records without them); `heldout_<metric>@k`,
`oracle_<metric>@k` (through `null_if_empty`, so a record written before H7
with `n == 0` and 0.0 means reads `null` too) and the non-dict `quality_*` entries (`quality_parity`,
`quality_jaccard_vs_first@k`, `quality_score_max_abs_diff`); `perf_<key>` for
every perf-entry key except `window_sm_mhz` and `kernels`, so
`perf_window_medians_ms` (a list column), `perf_ids_sha256`, `perf_ids_sha256_canon` and `perf_rounds`
are in it: the table alone carries the bootstrap's inputs, and
`tests/bench/test_records.py` pins it equal to the nested entry `report._attach`
joins. Types are inferred per column (int, double, bool, string,
list<string>), so a column that mixes types across records fails the
aggregation instead of reaching a table; the column set is the union over
the records, so it varies with `ks` and schema version. Parquet over CSV
because the table is typed (no string round trip to re-guess `True`, `""` or
`1e-05`), smaller than the JSONL it comes from (D1-a: 1.4 MB → 0.2 MB), and read in one call by polars, pyarrow or DuckDB.

Two dirty flags, both `null` outside a git checkout: `env.dirty` is
`git status --porcelain -- .` in the imported package's directory (untracked files
included — a new kernel module is measured code too) and is the flag
`report.py` refuses to cite; `env.repo_dirty` is the tracked
files anywhere else (`--untracked-files=no`, informational: a docs or
harness edit). The results tree is gitignored, so a campaign appending to
its JSONL never flips it.

### Oracle blob v4

`oracle.load_or_build(gt_dir, sweep, k_gt, item_embs=, queries=, targets=,
qa_sweep=, skip_mask=, clauses=, filter_mod=, attrs_digest=, device=)` returns one dict,
cached at `<gt_dir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt`:

| key | value |
|---|---|
| `version` | `4` |
| `topk` | `[U, k_gt]` int64 exact filtered top-k (0-indexed, `-1` padded; skipped rows all `-1`) |
| `pass_counts` | `[U]` int64 items passing the exact mask; `-1` on skipped rows |
| `pass_rate` | mean of `pass_counts / n_items` over kept rows |
| `targets_in_filter` | `[U, T]` bool, target `t` of user `u` passes the mask |
| `target_in_filter` | `[U]` bool, any target passes |
| `n_items`, `n_queries`, `n_kept`, `k_gt`, `sweep`, `clauses` | shape of the build |
| `fingerprint` | sha256 over shapes, dtypes and a 64-row linspace sample of `item_embs`, `queries`, `targets`, `qa_sweep`; the full bytes of `item_attrs` and `clause_is_reverse` (`oracle.attrs_digest`, computed once per `(dataset, dim)` in `inputs.load_inputs` as `inputs["attrs_digest"]`); plus `clauses` and `k_gt` |
| `code_version`, `harness_commit`, `torch`, `created` | provenance |

**Item-chunked** (`oracle.compute`): per batch of 64 kept queries the loop
walks the items in chunks of `ITEM_CHUNK = 2_000_000` rows: `q @ chunk.T` in
fp32 (the function raises unless TF32 is off and the matmul precision is
`highest`, which `measure.setup` sets), `-inf` where the exact filter's
`evaluate_mask(qa, start, end)` (a `[B, end − start]` bool over that item
range) fails, a local `topk`, merged into the running `[B, k]` with ids
offset by the chunk start; equal scores go to the lower id (a stable sort by
id, then by score). `pass_counts` and `targets_in_filter` accumulate per
chunk the same way, and `oracle.pass_counts` (the bloom counts) sums
`evaluate_mask` over the same chunks. No `[B, N]` score or mask exists and
`item_embs` is never transposed into a copy (the chunk's `.t()` is a view
cuBLAS reads transposed). The blob version and fingerprint did not change:
a blob built in one shot holds the same content up to exact-score ties
(the one-shot `torch.topk` promised no order among them).

The fingerprint is in the file name, so a stale blob is never read (a
different dim off the same `data_dir`, regenerated attrs — even one
edited value, since the item side is hashed in full — a retrained
checkpoint, another `users_limit` or clause set each produce a new file),
and the blob is a portable artifact for roadmap F4. The filter passed in
is always exact (`inputs.exact_filter`: the clause module itself on `clause`
cells, a fresh `ExactAttributeFilter` on `bloom` cells) — bloom's false
positives never leak into ground truth. Bloom pass rates are not cached.

## Report (`report.py`)

`bench report <results>` writes `<results>/report/` (or `--out DIR`):
`results.parquet` first (`records.aggregate`, the artifact that ships
with the paper), then one file per artifact, then `report.md`. Every table
and figure is built from `results.parquet`; `records.latest` is read a
second time for the provenance block alone, which needs the nested `env`
the table flattens to seven columns. To report on a leg that is no longer
on disk, `bench fetch` it first. The tables select by dataset and dim,
never by encoder, so a results tree holding two `inputs` under one
`(dataset, dim)` (a gSASRec and an E1c goodreads leg) is refused with the
identities named; report each encoder from its own tree.

| artifact (`--only` name) | file | what it is |
|---|---|---|
| `pareto` | `tables/tab-pareto_<dataset>.tex` | sweep × arm (one row per parameter set): oracle recall, `median_ms`, speedup vs LiNR V1, `index_mib` (`tab:pareto_<dataset>`) |
| `memory` | `tables/tab-memory.tex` | `index_mib`, datasets × algos (`tab:memory`) |
| `parity` | `tables/tab-backend_parity.tex` | per `(dataset, sweep, algo, backend)`: `path`, `jaccard_vs_first@k`, `score_max_abs_diff`, eager vs graph median and their ratio |
| `matched` | `tables/tab-matched_recall.tex`, `matched_recall.json` | per recall curve (SilverTorch along `n_probe`, one curve per `n_lists` / filter kind / backend / suite; V3 along `candidate_pool` or `candidate_pool_frac`) and batch size: latency at `recall_oracle@k` 0.90 and 0.95 with the two bracketing points named, or why not; `n95` per SilverTorch curve, the smallest measured `n_probe` with recall ≥ 0.95 — the value the campaign writes into `suites.yaml` (`tab:matched_recall`) A curve is emitted once per *timed* batch size, so a curve with no perf entries (a `perf: false` suite: `n95`, `bloomwidth`) is not emitted and neither is its `n95`; the `n95` suite's value is read off the records with [`n95.py`](../artifacts/campaign-v2/v-pubmed/n95.py); with `--bands e1,e2,…` also `tables/tab-matched_bands.tex` / `matched_bands.json`: QPS at `recall_oracle@k` 0.95 per selectivity band (`[0, e1)`, …, `[e_last, 1]`), each band's recall the mean per-query recall of the cell's queries whose own pass rate (`pass_count / n_items`) falls in it (≥ 20 queries, else `---`) against the cell's latency, interpolated as above; exact arms (V1, V2) at their own latency, recall shown where below 0.95 (EXHIBITS idea #6) |
| `t1` | `tables/tab-t1_claims.tex` | T1, the claims table: per claim C1–C7 of [`evaluation/claims.yaml`](../../evaluation/claims.yaml) the original number and source, "ours" from the yaml's record selectors, the hand-written verdict (`---` while `null`) |
| `t2` | `tables/tab-t2_<dataset>.tex` | T2, real filters (`filter` suite, goodreads / arxiv / yfcc10m / pubmed): per sweep, every arm at the operating point (SilverTorch `n_probe` 24) and SilverTorch at matched recall 0.95, recall@100 and p50 at bs 1 and 16 |
| `t3` | `tables/tab-t3.tex` | T3, official vs Triton (`h2h` suite): per cell, `k`, `bs`, arm and mode: p50 with CI, the paired ratio over Triton eager, kernel-only µs and launches (`kernels_us` / `kernels_calls`, every kernel of the `--profile` call; a record without them shows the top-8 sum marked $^{8}$), the like-for-like scorer µs (`kernel_scopes["scorer"]`, H-SCOPE; `---` on records without scopes), `index_mib`, `peak_fwd_mib`, `ids_sha256_canon` identity with Triton eager per common seed: `=` (equal up to tied-id order), `= (ties)` (equal up to boundary ties: where the hash differs, `quality_score_max_abs_diff` is exactly 0, so only a tied id at the k-th cut differs), else `≠`; jaccard, max score difference |
| `f1` | `figures/fig-f1-latency-vs-pass-rate.png` | F1: p50 vs pass rate on the `*-synth` datasets (log x), one panel per scale × bs; V1, V2, V3 per pool, postfilter per alpha, SilverTorch Triton at matched recall 0.95 |
| `f2` | `figures/fig-f2-recall-vs-pass-rate.png` | F2: `recall_oracle@100` vs pass rate; synth SilverTorch per measured `n_probe` and V3 per pool, the real `filter` sweeps overlaid as per-query pass-rate buckets from the sidecars |
| `f3` | `figures/fig-f3-pareto.png` | F3: the `deep` suite's recall–latency curves (matched-recall curves), one panel per (dataset, sweep) × bs (one row per sweep), filter kinds as separate curves, 0.90 / 0.95 marked |
| `f4a` | `figures/fig-f4a-bloomwidth.png`, `tables/tab-f4a_bloomwidth.tex` | F4a (`bloomwidth*` suites): `bloom_fp_rate` and `index_mib` vs `m_bits` per dataset / backend / `k_hash`; the table adds `filter_mib`, recall and the bs-16 timed point |
| `f4b` | `figures/fig-f4b-codesign.png`, `tables/tab-f4b_codesign.tex` | F4b (`codesign` suite): partial vs full bloom path latency vs `n_probe` and the paired full / partial ratio with its CI |
| `fig_deep_sweep` | `figures/fig-deep-sweep-*.png` | one per swept parameter: recall and latency against its value, whiskers = min–max across seeds |
| `fig_latency_violin` | `figures/fig-latency-violin.png` | per-call distributions from the samples sidecar (`<name>.samples.jsonl`, one torn trailing line tolerated) |
| `methodology` | `methodology.tex` | the measurement-methodology itemize, its constants read live out of `measure.latency`, `inputs.query_pool` and `run` so text and code cannot drift |
| — | `report.md` | provenance, citability verdict, the selection used, a coverage table, the failed cells with their stage and error, the partial records, the unstable variants with their spread, the artifact list |

**Citability (CLAUDE.md rule 2).** Nothing is citable by default. `--gate
STEP` declares that step's roadmap gate green, and only then can an
artifact come out unmarked — but the evidence vetoes the flag: a `failed`
or `partial` record, an `env.dirty` one, or one whose `env.git_branch` is
not `staging` / `development` / `main` (rule 2's own wording: "harness numbers from a
branch are not paper material") keeps the marker on. Otherwise
every `.tex` carries a `% PROVENANCE: *** NOT CITABLE ***` banner listing
the reasons in full, its caption starts with `\textbf{[NOT CITABLE: …]}`
naming them short (`gate not green`, `N failed`, `N partial: <partial_reasons
with counts>`, `N dirty`, `branch <name>`, `no records`; `provenance()["marks"]`),
every figure gets a diagonal "NOT CITABLE" watermark over the same short
reasons, and `report.md` lists the full ones (`provenance()["blockers"]`,
also what `bench upload` puts in the manifest). The marker states the
evidence only: it says nothing about when the records were taken. Every artifact
carries the `code_version`, the commit, the branch, the GPU, the schema
version and the run window regardless.

**Failed, partial, unstable.** A `status: failed` record never reaches a
number and is listed in `report.md` with its stage and error. A `partial`
record is used and marked `*`; a perf entry with `unstable: true` (window
spread > 5 %) is used and marked `†`. Both marks are explained in every
caption.

**Clocks.** Latency artifacts print which estimator they used: the
per-variant under-load `perf[].sm_mhz`, with its observed range. No clock
normalisation is applied, and the idle `env.sm_mhz_idle` (or a schema-1
`env.sm_mhz`, which is a whole-run median dominated by idle samples) is
never compared with an under-load one, because an idle sample reads low
and makes matched latencies look like regressions.

**Selection.** One set of options narrows the grid artifacts: `--dim`,
`--k`, `--bs`, `--mode`, `--backend`. Where a table's shape allows only one
value per cell and the records hold several parameter sets (`tab:memory`),
the smallest by canonical-JSON order is shown and a caption footnote names
all of them; several seeds are reduced to their median (min–max whiskers in
the figures). A `null` value (a held-out side with no scored row) is
skipped, not averaged in as a miss. A selection that matches nothing emits
the table with a `--- no matching cells ---` row rather than failing.

**Language.** Every label, caption and note is English with a decimal
point; the thesis's Russian artifacts (decimal comma) stay in git history
and the delivered thesis, and are not regenerated.

### Statistics

[`bench/stats.py`](../../evaluation/bench/stats.py) — its own module because
it is pure numpy with known-answer tests (`tests/bench/test_stats.py`) and
no record, LaTeX or plotting in it. Every resample uses `B = 10 000` draws
from one fixed generator seed (`stats.SEED`), so regenerating a report from
the same records prints the same intervals.

| quantity | point estimate | 95 % CI (percentile bootstrap) |
|---|---|---|
| latency of one arm at one `(bs, k, mode)` | median of the per-window medians (`perf[].window_medians_ms`) pooled over seed × window | resampling seed × window units |
| `recall_oracle@k` of one arm | mean over queries of the per-query recall from the [sidecar](#output-one-jsonl-record-per-cell) (`per_query`), averaged first across the arm's distinct sidecars (its seeds, when quality depends on the seed; a quality-cache copy points at its source's sidecar) | resampling queries. A schema ≤ 3 record has no sidecar: the median across seeds, no CI |
| A over B, interleaved (every common seed ran both in one `interleave.group`) | median of the per-round ratios window *i* of A / window *i* of B | resampling seed × round |
| A over B, otherwise | median(A) / median(B) | each side resampled independently; marked $^{u}$ |

A ratio whose CI contains 1.0 prints "no difference" (the plan's noise
gate); no repeat is added to force significance. `_attach` joins each
`results.parquet` row to its nested record and perf entry and reads the
windows, the sidecar path and the interleave block there. The parquet carries
the same values as `perf_window_medians_ms`, `per_query` and
`interleave_group` / `interleave_position` (pinned equal by
`tests/bench/test_records.py`), so the shipped table alone holds the
bootstrap's inputs. The per-sweep table, the parity table and the deep-sweep
figures already use these estimators.

**Matched recall.** `stats.at_recall` interpolates latency at a target
recall on one curve: piecewise-linear between the two adjacent measured
points that bracket the target on the recall-sorted curve, the measured
point itself on an exact hit, never extrapolated ("not reached" past the
last point; "first point already above" before the first). The curve's
latencies are taken in `graph`, the headline mode, or in `eager` where no
`graph` entry ran (official); the mode used is printed (`_timed`).

**Labels and the alpha rule.** `ALGO_LABEL` names every algo a table may
show; a record of an algo without a label (a retired `linr_v4` leg) reaches
no table. The per-sweep tables show one row per parameter set, labelled with
it (`_arm`): `postfilter ($\alpha$=1)` is the baseline's headline row and
`$\alpha$=8` its strong variant, and no number is ever averaged across
`alpha` or any other parameter. The postfilter is torch by definition
([The postfilter baseline](#the-postfilter-baseline)), so `--backend`
selects it whatever its value (`FIXED_BACKEND`).

### Campaign manifest

[`evaluation/campaign.yaml`](../../evaluation/campaign.yaml) says which
records the paper reads. `bench report <fetched tree> --manifest
evaluation/campaign.yaml` groups the tree's records by cell (the key block)
and, per cell, takes the quality fields (`quality`, `per_query`,
`pass_rate`, `bloom_fp_rate`, the oracle / held-out counts) from the record
at the entry's `quality.code_version` and everything else from the record
at its `perf.code_version` — the reuse rule keeps an old record's quality
while its timing is rerun; such a merged record carries `quality_source:
{seed, code_version}`. The entry is the most specific `entries[].match`
(dataset, suite, algo, backend, optionally filter_kind and sweep; two
equally specific matches fail), else `default`. A cell with no entry is
excluded; one with no record at an accepted code_version is missing its
quality or its perf; neither is ever filled from another code_version.
`report.md` lists all three groups and the manifest's `log` (date, change,
arms rerun). The merged records are written to `<out>/selected/` and
`results.parquet` is aggregated from there, so the shipped table is
exactly what the exhibits read; the provenance verdict judges every
accepted source record (a reused quality record from a branch still vetoes
`--gate`). The samples sidecar has no `code_version` to select by, so the
violin figure is empty under `--manifest`. `hub` names the
`pinkmeme/eval-results` subtree holding an entry's records (`{field}` = the
cell's key field) for the planner's `bench fetch`; the report does not read
it. Without `--manifest` the report takes the latest record per resume key,
for scratch trees and smokes. The shipped manifest's `default` is the `campaign-v2.1` tag's tree hash
(`f01255f1…`, hub `campaign-v2.1/{dataset}-{suite}`; `campaign-v2` was
`408b1188…`); its 12 reuse
entries keep `d1/arxiv`'s `72e5a90` quality and perf for arXiv `filter`
V1-V3 on `c0_maincat` / `all4`
([artifact](../artifacts/campaign-v2/README.md#reuse-entries)). A `match`
cannot name `params` or `seed`, so an entry covers every seed and grid
point under it: one is written only where old records cover them all.

### Paper exhibits

Each exhibit is an `ARTIFACTS` function, so `--only t2` regenerates one.
They select by suite and dataset, not by `--dim` / `--bs` / `--mode`:
every dataset at its own width, `k` 100, bs 1 and 16, latency in `graph`
(the headline mode) or in `eager` where the arm has no `graph` entry
(official; marked $^{e}$). SilverTorch's operating point is
`n_probe` 24; "matched" is `stats.at_recall` on that arm's curve. F2 draws
SilverTorch at each measured `n_probe` (the `synth` sweep), not "at
matched recall", which would be a flat line at the target. The real-sweep
buckets of F2 are half-decade bins of `pass_count / n_items` holding ≥ 20
queries. T1's `claims.yaml` selects with any parquet column plus a
`params` subset (`null` = absent) and must hit one arm (else the report
fails naming them); `metric` is `latency`, `recall`, `matched:<target>` or
`field:<column>`, and `vs` makes it a ratio. The report never writes a
verdict.

**Not compiled.** No TeX toolchain is installed on this box, so the
fragments are checked structurally (`tests/bench/test_report.py`:
balanced environments, balanced braces and math, the labels
present, no unescaped `_` outside math, `\texttt` and `\label`), never
compiled.

## Results storage

Nothing the harness writes is committed. The records are the product of this
project and the only thing the paper may cite (CLAUDE.md rule 2); the box they
are produced on is rented and its disks are small ([storage](storage.md)); and
126 cells with their latency vectors are 78 MB, which is not a thing to keep
next to the code. So a results tree has two homes, one for each phase of its
life, and git holds only the pointer.

| what | where | why |
|---|---|---|
| `results/<suite>/<dataset>-d<dim>.jsonl` (the records), `*.samples.jsonl` and the `*.perquery/` sidecars while a leg runs | **local disk**, `results/` under `evaluation/` (gitignored; `--out` elsewhere) | the working state: `append_record` writes a cell and `fsync`s it, resume reads it back, and neither may depend on the network. Inside the repo checkout, which on a pod is `/workspace` and survives a restart |
| the same records, samples and per-query sidecars, plus `results.parquet`, once a leg finishes | **HF Hub**, `pinkmeme/eval-results/<leg>/`, private (`bench upload`) | the archive. The JSONL goes up next to the Parquet because it is the lossless form: failed records with their tracebacks, superseded records, the nested `quality` and `env`, `window_sm_mhz`, `kernels` — everything the flat table drops — and it is what resume and the manifest's provenance read |
| `report/` (`*.tex`, figures, `report.md`) behind a gate | **git**, under `docs/artifacts/<plan>/` — without its `results.parquet` (gitignored), which `bench report` regenerates from the Hub copy | small, reviewed, and what a gate's text points at |
| raw dumps behind a documented finding (kernel timings, probe outputs, ETL logs) | **HF Hub**, `pinkmeme/eval-results/artifacts/<plan>/<same path>` | the finding lives in prose in [validation](../validation.md) and the system pages; the dump is only for re-derivation |
| `results/_parity/*/*.npz` (600–680 MB per run), `results/_logs/` | **nowhere** — deleted | rewritten by every run, and the parity *verdict* (`jaccard_vs_first@k`, `score_max_abs_diff`) is already inside the record. `bench upload` skips every `_`-prefixed path part |

A leg is finished when its records are on the Hub and verified: `bench upload
--results results --path-in-repo <leg> --verify`, after which the local tree
may be deleted. The Hub subtree's `MANIFEST.json` carries the sha256 of every
file it holds, and git records the sha256 of each manifest in
[hub-index.md](../artifacts/hub-index.md), so git can prove
what the Hub has without downloading it. The golden cells are the one
exception: they are a gate fixture the tests read, and stay in
[`evaluation/golden/`](../../evaluation/golden/README.md).

### `bench upload`

One invocation publishes one `--path-in-repo` subtree of one results tree:

```bash
uv run bench upload --results results --path-in-repo d1-a --verify
uv run bench upload --results /scratch/tmp/h1-hub/artifacts/kernel-opt \
    --path-in-repo artifacts/kernel-opt --dry-run
```

`--repo-id` defaults to `upload.RESULTS_REPO` = `pinkmeme/eval-results` — the
registry entry for *results*, beside `eval_datasets.hub.EVAL_REPOS` for
*datasets*. When the tree holds records (`records.record_files`: any
`<suite>/*.jsonl`), the upload first regenerates `results.parquet` from them,
so the Hub copy never carries a table older than its records. The listing is
`upload.files`: the tree minus anything under a `_`-prefixed path part and
minus the two files a previous upload generated. Everything goes up in **one
commit** (`create_commit`, so a failure leaves no half-published subtree),
together with:

- **`<prefix>/MANIFEST.json`** — `report.provenance` over the records being
  uploaded (`code_version`, `commit`, `git_branch`, `schema_version`, `gpu`,
  `host`, the run window, the status counts, the `unstable` count, the gate and
  the citability verdict with its reasons) plus `{path, bytes, sha256}` per
  file. It is the same function `bench report` calls, so the Hub copy cannot
  claim more than the tables would. A real upload prints the manifest's own
  sha256, the value [hub-index.md](../artifacts/hub-index.md)
  records.
- **`README.md` at the repo root** — regenerated from *every* manifest in the
  repo (the existing ones are fetched first), so a second upload does not drop
  the first subtree from the front page. Generated from the records, never
  hand-written.

**Citability survives the trip.** `--gate STEP` is the only route to
`"citable": true`, and the evidence vetoes it exactly as in
[Report](#report-reportpy): a `failed` or `partial` record, an `env.dirty` one,
or one whose `env.git_branch` is not `staging` / `development` / `main` keeps the verdict at
`false` and lists why. Read a downloaded file's `MANIFEST.json` before believing
a number came from a campaign.

**Private by default** (`--private/--public`, default private). Going public is
roadmap F4's decision — the Zenodo DOI and the anonymised review mirror — not
this command's.

**`--verify`** downloads the subtree it just wrote into a temp directory (on the
VM's local disk, never `/workspace`) and checks every sha256 against the
manifest; a mismatch is a non-zero exit. An upload path that has never been
downloaded is not a backup.

### `bench fetch`

`bench fetch --path-in-repo d1-a [--results results]` is the way back: it
downloads the subtree into a temp directory, checks every file against the
subtree's `MANIFEST.json`, and copies the files into the results tree. It
refuses — before copying anything — when the tree already holds a *different*
copy of one of them (a live campaign's JSONL is never overwritten); identical
files are fine, so fetching twice is a no-op. A fetched tree is an ordinary
results tree: `bench report` reads it and `bench run --resume` appends to it.

`tests/bench/test_upload.py` pins both with no network: the scratch
exclusion, the checksums, the four ways evidence beats `--gate`, the generated
README, the round-trip check catching a changed byte, the commit's operation
list with the regenerated `results.parquet` and `private=True`, a fetched tree
that resume can read, and a fetch refusing to overwrite a different local copy.

## Inputs (`inputs.py`)

`load_inputs(ds, device, with_filters=True)` returns `item_embs [N, D]`
fp32 on device; `queries [U, D]`, `targets [U, T]` (`-1`-padded,
0-indexed), `n_targets [U]` on CPU; `qa [U, C]` int64 CPU, `item_attrs
[N, C, A]` and `clause_is_reverse [C]` on device (or `None`) with their
full-bytes `attrs_digest` (the oracle fingerprint's item side, hashed
once here); `n_items`, `n_queries`. Two loaders, keyed on the `Dataset`:

| `Dataset` field set | path |
|---|---|
| `checkpoint` | SASRec: `training.encode.encode_split` — `load_model_for_eval` + `encode_queries` over `test.parquet`, cached as `<ckpt-dir>/encoded_queries_v2.pt` keyed on ckpt mtime + `max_seq_length`, the *full* split (a free-disk budget of `0.7 × free − 4 GiB` guards the write); the padding row is dropped, target ids shifted −1 |
| `content_dir` | pre-encoded text: `eval_datasets.layout.load_text_items` (`text_emb.pt`, or `shard_index.json` + shards; the `.meta.json` nomic prefixes asserted; fp16 → fp32 + L2-normalise) and `load_text_queries` (`query_emb.pt` + `heldout.parquet`, 1-indexed → 0-indexed) |

The readers and their checks live in
[`eval_datasets/layout.py`](../../evaluation/eval_datasets/layout.py) — the
on-disk contract as code, shared with the ETL that writes it
([datasets.md](datasets.md#the-layout-contract-layoutpy)): both item
layouts load (the modern `[N, …]` one and the legacy 1-indexed `[N+1, …]`
one the Hub copies still are — `drop_legacy_padding_row` by content, then
`check_items_aligned`), `load_query_attrs` reads the `Dataset`'s
`query_attrs` (`eval_split.parquet`'s `query_attrs_narrow`, or the `.pt`
named by `filters.query_attrs`) and checks its row count against the
*full* split, and `apply_users_limit` is the one
`users_limit` site: a prefix over queries, targets, `n_targets` and `qa`
together (a prefix, not a sample, so quality stays comparable with the golden cells). `sweep_qa(qa, clauses)`
codes inactive clauses `-1` and skip-masks rows left without a live
clause; `query_pool(inputs, qa_s, skip, bs=, seed=, n_pool=4096, device=)`
draws the fixed-seed perf pool from the kept rows.

## Tests

One tree, [`evaluation/tests/`](../../evaluation/tests/), mirroring the
three packages, all CPU-only (`backend="torch"`; the Triton, CUDA-event,
graph-capture and SASRec paths run in the GPU gates). `tests/conftest.py`
holds the tiny-dataset writer every package's tests share and skips tests
marked `gpu` without CUDA (`[tool.pytest.ini_options]` registers the
marker and `testpaths`):

```bash
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
```

`[tool.pytest.ini_options] pythonpath = [".", "../retrieve/src"]` makes the
suite import `bench`, `training`, `eval_datasets` and `retrieve` from the
checkout it sits in. Without it a worktree on the shared venv tests the
checkout the venv's `.pth` names. Console scripts (`uv run --no-sync bench
…`) still resolve through that `.pth`; in a worktree run `python -m
bench.cli …` from `evaluation/`.

### Lint

`ruff check evaluation` selects `E, W, F, I, UP` plus bugbear (`B`, so
`zip()` needs `strict=`), comprehensions (`C4`), simplify (`SIM`), unused
`noqa` (`RUF100`), blind `except` (`BLE001`) and no inline imports
(`PLC0415`, a preview rule enabled alone through `explicit-preview-rules`).
Each per-file ignore in `evaluation/pyproject.toml` carries its reason: the
ETL CLIs defer torch, sentence-transformers and transformers to the
subcommand that needs them, and goodreads' broad `except`s wait on their own
cleanup. An inline `noqa` says why too. `retrieve/` selects the same set.
One ruff version everywhere: `ruff==0.15.6` in the
workspace `dev` group and the same `rev` in `.pre-commit-config.yaml`, whose
hooks run `ruff check` and `ruff format` on `retrieve/` and `evaluation/`, the merge-conflict,
large-file (1 MB — records and dumps belong on the Hub), end-of-file and
trailing-whitespace hooks (the last two never on `articles/`, `docs/artifacts/`,
`evaluation/golden/`), and `scripts/check_doc_links.py`.

| file | checks |
|---|---|
| `test_dependency_direction.py` | every `import` / `from` in `bench/`, `training/`, `eval_datasets/` resolved with `ast`: `bench` → `training.encode`, `eval_datasets.layout`; `training` → `eval_datasets.{hub,layout}`; `eval_datasets` → nothing; the library only from `bench`, through `retrieve` (the algo and filter classes) and `retrieve.interfaces` (`DISPATCH`, `FilterModule`). The walk must see more than 25 files, and an allow-list entry no import uses fails as stale |
| `test_env_readers.py` | every `os.environ.get` / `os.getenv` / `os.environ[...]` of a `RETRIEVE_*` name under `evaluation/` sits in its owner (`RETRIEVE_DATA_ROOT`: `eval_datasets/hub.py`, read through `data_root()`); same file-count and stale-owner guards |
| `bench/test_paths.py` | `PATHS == derive(DISPATCH)`: the derived table equals the harness's expected paths, the grid is complete, `DISPATCH` names every algo × backend and adds only `Postfilter` to the library's |
| `bench/test_records.py` | `resume_key` canonical and `code_version`-sensitive; append / read round trip; one torn trailing line; `aggregate` one row per perf entry, last record per key, typed columns; two jobs differing only in `--checkpoint` key apart; a real D1 arxiv key block (schema 2) keeps today's arxiv job's resume key; a schema-2 goodreads record is gSASRec; the schema-4 columns (null on a schema-3 record) and `perf_window_medians_ms` equal to the nested entry `report._attach` joins |
| `bench/test_measure.py` | `stats` vs numpy, `latency` control flow, `latency_group`'s A, B, A, B windows (G-interleave), `index_bytes` dedup, `provenance` git fields, `official_commit` from PEP 610 (git, local dir, no `direct_url.json`, not installed), `dirty` scoped to the library subtree with the `files:` fallback, H-PROVENANCE (`code_version` / `dirty` / `files_hash` follow an imported package in another checkout, a commit there moves the stamp, a wheel outside git gets the content hash), `atomic_write`, `clocks` shape, `graph_callable` refusals |
| `bench/test_metrics.py` | padding / IDCG / denominator contracts; running sums equal per-row means to 1e-9; `jaccard_at_k` |
| `bench/test_algos.py` | `build` on every `(algo, filter_kind)` torch cell: the `k` setter slices the top-k and changes no buffer; the filter submodule in `index_bytes`; `set_query_params`; build refusals; `postfilter` equal (`torch.equal` on ids and scores) to a hand-computed matmul → topk(`alpha*k`) → dense-mask filter → first `k` reference at every `alpha` in `{1, 2, 4, 8}`, `k` within and past `N`, clause and bloom, the sentinel path exercised; its refusals |
| `bench/test_config.py` | job counts and keys per suite on `tests/bench/data/{mini,text,suites}.yaml`; `disabled`; build/query split; arms (filter kinds, sweeps, per-dataset overrides, empty slots); `ks_by_sweep`; gridded bloom widths, `compile`, `candidate_pool_frac`; each arm error named; narrows and `Job.narrowed`; the `-synth` YAMLs sharing their parent's inputs; `perf: false`; the interleave groups of the real suites, their parse errors, a job in two groups, the merged campaign children, `score_path` off official; G-grid (every real suite × dataset's job / cell counts and the 2026-10-08 rules on every cell); G-key (pinned resume keys of kept cells); every `config/*.yaml` × every suite through `load_matrix` |
| `bench/test_inputs.py` | `load_inputs` on the conftest writer, `users_limit` once, prefix and row-count checks, the legacy layout loading equal to the modern one, misalignment raising, `attrs_digest`, `filters.query_attrs` (a `.pt` replacing `eval_split.parquet`: same full-split row check, same `users_limit` prefix), `sweep_qa`, filters by filter backend, `query_pool` |
| `bench/test_oracle.py` | padding, v4 fields and arithmetic, fingerprint in the file name, an unreadable blob rebuilt, bloom FP rate, `code_version`; the item-chunked oracle and pass counts equal to a one-shot stable-sort reference at three chunk sizes, ties to the lower id; TF32 refused (G-oracle) |
| `bench/test_run.py` | end to end on the tiny fixture, both modes (`graph` = the CPU null entry): record schema, resume, `code_version` invalidation, `partial`, a failed cell + continue, a sticky CUDA error, the quality gate, the parity spill (and a same-backend spill rewritten), reachable-target masking, the plan cache off through `OfficialConfig`, `modes` stamped per job (an eager-only uncapturable record `ok`, a capturable one `partial`), `postfilter`'s `recall_oracle` per `k` (equal to a pass at that `k`, never above the exact V1's); G-cache (seed-free arms copy, V3 / SilverTorch never, misses on other `ks` / `code_version`); a quality-only suite `ok`; G-ids (exact hash, and the canonical hash equal across a tied pair's two orders); G-dump (sidecar mean = record, upload lists it, aggregate skips it); G-interleave (same keys, one group per sweep and seed, `rounds`, whole-group resume); `frac_windows_below_max` against the device max, whichever cell ran first; `score_path` arms sharing the triton parity spill |
| `bench/test_cli.py` | `bench run` via `CliRunner`, `--checkpoint` landing as the record's `inputs` (encoder faked), a real one-child `bench campaign` ending in `results.parquet`, a faked timed-out child, a restart mid-group keeping its parity spill (in-process children), zero cells → exit 1, `bench report` over the campaign's own records, `bench env`'s JSON keys, the child log's start / end clock blocks and histogram, `bench oracle` prebuilding every sweep so `bench run` builds none |
| `bench/test_upload.py` | `bench upload` with no network: `_logs/` and `_parity/` never listed, the manifest's sha256 per file, evidence beating `--gate` four ways, the generated README over several subtrees, `verify` catching a changed byte, the commit's operations (with the regenerated `results.parquet`) and `private=True`; `bench fetch` restoring a tree resume reads and refusing to overwrite a different local copy |
| `bench/test_report.py` | every column the tables read still comes out of `records.aggregate`; every artifact emitted; the LaTeX structurally balanced with its labels and no unescaped `_`; one row per postfilter alpha; an unlabelled algo (`linr_v4`) in no table; a `failed` record excluded and a `partial` / `unstable` one marked; citability off by default and evidence beating `--gate`; two `inputs` under one `(dataset, dim)` refused; an empty tree; a schema-1 record; paired speedups from interleaved arms and recall CIs from sidecars; T3 ids `=` / `= (ties)` / `≠` and kernel-only from `kernels_us` (top-8 fallback marked); F3 one panel per sweep × bs |
| `bench/test_stats.py` | G-stats: a constant's CI is a point; A = 2B pairs to 2.0 with a CI excluding 1; identical arms are no difference; interpolation exact on a linear curve, never extrapolated |
| `bench/test_c4_gate.py` | the golden-comparison gate script, [`c4_gate.py`](../artifacts/evaluation-harness-v2/c4_gate.py), against synthesised schema-1 records |
| `eval_datasets/test_layout.py` | the legacy pad-row rule, `apply_users_limit`, `validate_layout` clean on both layouts and flagging a short `eval_split`, a missing or swapped prefix sidecar, misaligned attrs |
| `eval_datasets/test_synth_filter.py` | G-synth: achieved pass rate within 1 % of target wherever `N·p ≥ 10^4` (N = 2 M), pass sets nested, same seed byte-identical, `query_attrs_synth` all ones at the full-split row count, the real attrs' bytes and `attrs_digest` untouched (modern and legacy layouts), the CLI resolving `data_dir` from the dataset YAML. G-corr (`--correlated`, on 10 tight blobs): items and queries both at their nearest centroid (brute force), the sidecar's mean pass rate equal to the attrs' and within 50 % of p (a random-init local optimum), over 80 % of a query's 10 nearest items passing, the real and uniform files untouched, rebuilds byte-identical, the CLI picking `content_dir` at `--dim` and refusing a dataset without one |
| `eval_datasets/test_yfcc.py`, `test_pubmed.py`, `test_kuairand.py`, `test_openalex.py` | the four ETL loaders on synthetic fixtures |
| `training/test_encode.py` | `training.evaluate`'s recall / ndcg equal `bench.metrics` to 1e-9; `encode_split`'s cache hit / stale key |

## How to run

The repo is a uv workspace; `uv run` from inside `evaluation/` finds it.
On a box that allows it, lock the clocks first (`sudo nvidia-smi -pm 1 &&
sudo nvidia-smi -lgc 1410`); on this A100 container that is denied, so
read latencies against `perf[].sm_mhz` and `env.clocks_drift`.

```bash
cd evaluation
# the golden-comparison cell set: goodreads d128, clause c0_genre, every algo and backend, on the
# golden's gSASRec checkpoint (the config points at E1c); the arxiv golden cell and the full
# commands: evaluation/golden/README.md, "Exact commands"
uv run bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre \
    --seed 0 --checkpoint data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/best_model.pt
# one cell, eager only, no perf — the fastest iteration
uv run bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch --backend triton \
    --filter-kind bloom --sweep c0_maincat --mode eager --skip-perf
# the campaign (roadmap D1), one suite at a time
uv run bench campaign --suite filter --resume
uv run bench campaign --suite deep --resume
uv run bench upload --results results --path-in-repo d1-a --verify   # publish, then check the round trip
uv run bench fetch --path-in-repo d1-a                               # a published leg back into results/
```

Sanity checks after a run (the campaign gate's clauses, state in
[validation](../validation.md#harness-gates)): `median_ms(bs=16) <
16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical
in `quality`; `memory_reserved_mib` flat (±5 %) across a group's cells;
`unstable` entries read against `perf[].sm_mhz`, since clocks are unlocked.

## Extending

**A new algorithm.** A library module first (the LiNR variants are
`retrieve.modules.linr`; a new composition is added there, because the
library retrieves and the harness measures —
[decisions](../decisions.md#harness)):
the filter as `self.filter`, `forward(query, query_clause_attrs=None) ->
(ids, scores)`, `k` forwarding to the final top-k layer, `set_query_params`
for any query-time knob, `capturable` as a class attribute, a `DISPATCH`
row. Then one line in [`algos.py`](../../evaluation/bench/algos.py)'s
`ALGOS` (and a branch in `build` if it registers differently; `PER_K_QUALITY`
if its top-`k` is not the prefix of its top-`k_max`), a suite's
`algos:` in `suites.yaml`; `tests/bench/test_paths.py` and `test_algos.py`
pick it up from the table.

**A new dataset.** One `config/<dataset>.yaml` (above) and the on-disk
artifacts under `data/<dataset>/` — `item_id_map.json`, `train/val/test.parquet`
+ a checkpoint for the SASRec layout, or `heldout.parquet` + `content*/`
(`text_emb.pt`, `query_emb.pt`, their `.meta.json` sidecars) for the
pre-encoded layout; filter sweeps add `item_attrs_narrow.pt`,
`clause_is_reverse_narrow.pt` and `eval_split.parquet` (see
[datasets.md](datasets.md) for the ETL and [filtering.md](filtering.md)
for the predicate spec). `bench check --dataset <name>` validates the
directory. Add the dataset to the suites it belongs in. The ETL is the
`eval-data` console script: `uv run eval-data yambda prep …`, `uv run
eval-data arxiv all …`, `uv run eval-data goodreads all …`.

### HuggingFace I/O

[`eval_datasets/hub.py`](../../evaluation/eval_datasets/hub.py) is the
single source of truth for HF reads / writes of **datasets and checkpoints**
(for *results* see [Results storage](#results-storage)): `EVAL_REPOS` maps each
dataset to its `pinkmeme/eval-<dataset>` HF dataset repo (eval inputs +
`checkpoints/<ckpt-id>/`), `RAW_REPOS` maps upstream raw sources into
`data/_raw/<source>/`.

```bash
uv run eval-data fetch yambda-500m            # pull eval inputs to data/yambda-500m/
uv run eval-data fetch arxiv-papers --dims d64,d128 --include-checkpoints
uv run eval-data publish goodreads-work-id --dry-run
uv run eval-data publish-checkpoint yambda-500m gsasrec-d128-drop0.5 --dry-run
```

The local data root resolves to `evaluation/data/` by default; override
with `RETRIEVE_DATA_ROOT=/some/path`. `hub.data_root()` is its only reader
(`tests/test_env_readers.py`); every ETL default goes through it. The pod image sets it to
`/workspace/data` ([storage](storage.md)).

## See also

- [architecture.md](architecture.md) — package layout and the
  `RetrievalModule` / `FilterModule` contracts.
- [kernels.md](kernels.md) — Triton and CUDA C++ kernel internals.
- [filtering.md](filtering.md) — filter API and the with-filters story.
- [datasets.md](datasets.md) — the ETL that produces the on-disk layout
  read here, and the SASRec training pipeline.
- [checkpoints.md](checkpoints.md) — the trained checkpoint inventory
  and the HF Hub workflow.
- [testing.md](testing.md) — the library correctness suite (separate
  from this perf harness).
