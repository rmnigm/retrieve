---
title: evaluation
created: 2026-09-26
updated: 2026-10-11
type: entity
tags: [harness]
sources: [evaluation/bench/, evaluation/config/, evaluation/tests/]
---

# `evaluation/` — the benchmark harness (`bench`)

The retrieval benchmark that produces the study's numbers: the
measurement protocol, the module map, the config matrix, the `bench` CLI,
the JSONL record, resume and the campaign process model.
**Nothing this harness produces is a result until its gate passed**: `bench report` marks a
tree NOT CITABLE unless `--gate` names a passed gate and the evidence agrees
([Report](#report)).

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
   runtime, torch/triton versions, the `silvertorch` extension's build
   (`silvertorch_build`), git commit + dirty flags, branch,
   `code_version`, hostname, python, UTC start, and one `nvidia-smi` clock
   sample (`env.sm_mhz_idle`, `mem_mhz`, `sm_max_mhz`, `power_limit_w`).
   Clocks cannot be locked in the container, so every perf entry carries its
   own under-load `sm_mhz` (see [Reading the numbers](#reading-the-numbers)).
2. **Inputs, once per (dataset, dim).** Item embeddings, queries, held-out
   targets, query attrs; `users_limit` applied once, as a prefix. Filter
   modules: one per *filter backend* (`triton` for triton and cuda
   cells, `torch` for torch cells). Per sweep: synthesised query attrs,
   skip mask, exact oracle top-`max(ks)` (the [blob v4](#oracle-blob-v4)
   cache) **plus** `target_in_filter[u]` (the held-out item passes the exact
   mask) and `pass_rate` (exact), the axis both papers organise their tables
   around. The harness measures no bloom false-positive rate: SilverTorch's
   bloom is Meta's, its false positives are counted outside the timed code.
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
   reach `recall_oracle@k_max ≥ 0.99` (`run.EXACT_MIN_RECALL`, fp16 tolerance); a failure is
   recorded as `failed` and then raises `QualityGateError`, which ends the
   run. Cross-backend correctness belongs to the library's parity suite
   ([testing](testing.md)); the harness keeps only a *wiring* check, the
   parity spill: the first backend to run a cell writes its top-`k_max` ids
   (int32) and scores (float32) to
   `results/_parity/<dataset>-d<dim>_<algo>/<hash>.npz` (`run.parity_group`
   names the directory; the hash is over the key block minus `backend` and
   minus `params.score_path`, Meta's scorer's epilogue, so `h2h`'s
   cuda arms compare against the triton spill; it includes `suite`,
   `filter_kind` and `sweep`), and later backends
   record `jaccard_vs_first@k` and `score_max_abs_diff` against it. There is
   no "missing reference" state: whichever backend runs a cell first writes
   the spill (`parity: "reference"`), so with `--resume` after the triton
   cells are done, `torch` becomes the reference and `cuda` records
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
   a mislabelled number. The filter's query side is prepared outside the timed
   call (§ Query preparation below). Per variant (`measure.latency`): 50 warm-up calls → sync →
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
   (`measure.kernel_summary`; the top-8 sum is a lower bound that favours
   the arm with more launches), and the same sums per named kernel scope as
   `kernel_scopes` (scorer / topk / epilogue / other, so the scorer
   column compares like with like). The sums skip `## …` events: the profiler books a
   compiled call's `## Call CompiledFxGraph <hash> ##` range as a device event
   spanning that graph's kernels, so counting it would count them twice (the top-8
   list keeps it). In a
   long-lived process the profiler drops the device activities at the end of
   a short window, one more kernel per session as the process ages, so H2H-FINAL
   stored empty or truncated lists. `measure.profile_once` therefore
   brackets the call with two `torch.cuda._sleep` sentinel kernels
   (`spin_kernel`, left out of `kernels`). It retries with an idle pad of
   0.01, 0.1, 1 and 5 s on both sides of the window until both sentinels are
   recorded, and raises (the cell fails at the perf stage) if they never are.
6. **Seeds and repeats.** Timing repeats are the 3 windows above (no
   rebuild). Seeds change the IVF (k-means), the OPORP projection
   (`seed`) and the pool; V1/V2 and the postfilter are seed-invariant in
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
   the papers' 10-80 M / 15.5 M, an 80 GB card, k 100 / 1000 with k 2048 on
   a subset against 1024/2000, closed-loop QPS rather than open-loop load, no
   live updates. `graph` is the headline mode and `eager` the secondary one,
   reported beside it.
8. **Cell cost.** Quality once per cell (not per k); perf the suite's
   k × bs × modes × 3 windows. Build params sweep separately from query params (one
   build, many query configs).

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
  Every `nvidia-smi` call (`clocks()`, `clock_report()`) targets
  `measure.smi_device()`, the first entry of `CUDA_VISIBLE_DEVICES` (an index
  or a UUID; `0` when unset), because `nvidia-smi -i` ignores that variable.
- **The job's clock log.** `env.frac_windows_below_max` is the share of a
  cell's window samples below the device's max SM clock, `env.sm_max_mhz`
  (`nvidia-smi` `clocks.max.sm`, sampled at process start; 1410 MHz on the
  A100), so it does not depend on which cells ran before; `null` without a
  device max. The record reuse rule reads it (under 10 %). The full
  view is in the job's log: `bench campaign` writes `nvidia-smi -q -d CLOCK`
  before the child starts and after it ends, and `run` logs the histogram
  of every window sample of the process at its end
  (`sm_mhz under load (n=…): 1410×…, 1395×…`, `measure.clock_histogram`).
- **No L2 flush between calls.** `measure.latency` does not flush L2
  (`triton.testing.do_bench` does by default). Consecutive calls take
  consecutive batches of the query pool, so what depends on the query
  (which index rows a batch reads) varies call to call, while what every
  call reads stays in L2 as it would in serving. Cache state is therefore
  the serving steady state, not a cold-cache worst case; a flushed number
  could be higher on bandwidth-bound variants, and it is not measured.
- **One sync per window, not per call.** `_time_calls` records a CUDA
  event pair around each call and calls `torch.cuda.synchronize()` once
  before the window and once after it, never inside the loop, so a sync
  does not inflate the per-call numbers. The host is free to run ahead
  unless the module itself syncs (the first eager call runs under
  `set_sync_debug_mode("warn")` to catch that).
- **No sub-launch-floor flag.** Per-call numbers under about 10 µs
  would be mostly timer and launch overhead; the smallest recorded `min_ms`
  is more than an order of magnitude above that, so no entry carries a flag for it.
- **`trimmed_mean_ms`** is the mean after dropping `n // 10` calls from
  each end of the chosen window: a central value that one-sided
  interference on the shared GPU moves less than `mean_ms`. `mean_ms`
  stays the papers' comparable number.
- **Samples go to a JSONL sidecar** (`<name>.samples.jsonl`, one line per
  perf entry with the key block, `k`, `bs`, `mode`, `ms: [...]`), because
  parquet cannot be appended per cell; `bench merge` carries them into
  `samples.parquet`.

## Architecture

Twelve modules under [`evaluation/bench/`](../../evaluation/bench/), plus
the three the harness takes from its sibling packages
(`eval_datasets.layout` for the on-disk readers, `eval_datasets.hub` for the Hub repo names,
`training.encode` for the SASRec encode). No base class, no context bags, no stats dataclasses, no
results-IO layer; two things are classes on purpose — `Job`, the
resolved build spec whose `key(params)` *is* the record's key block, and
the algo `nn.Module`s (the library's, and the harness's one baseline),
which exist for `buffers()`, `.k`, `torch.compile` and because their
`forward` is the readable spec of each cascade.

| module | owns |
|---|---|
| [`measure.py`](../../evaluation/bench/measure.py) | `setup`, `warm_gpu_once`, `provenance` (GPU, driver, CUDA, torch, triton, `silvertorch_build` — the `silvertorch` extension that would load: `so_path`, `so_sha256`, `so_bytes` and `nvcc_append_flags` from a `_build_flags.json` beside the `.so` (`null` = the extension's `setup.py` defaults; `null` overall without the package), commit, `dirty` = `subtree_dirty()` over `LIB` (the imported `retrieve` package's directory, in whichever checkout holds it), `repo_dirty`, branch, `code_version` = `LIB`'s tree hash, or `files:<sha256>` of the sources on disk when it is dirty, `lib_dir` = `LIB`, host, python, started; [Resume](#resume)), `clocks()` (one `nvidia-smi` sample), `timed_build`, `index_bytes` (Σ buffers, submodules included, deduplicated), `stats`, `latency_group(fns, bs=, mode=)` (§2.5 windows of each arm, round-robin across arms, IQR + outlier counts, `load: closed_loop`, `peak_fwd_mib`, the under-load `sm_mhz`) and `latency(fn, …)`, its one-arm case, `graph_callable` (raises `NotCapturable` with the record's `reason`; no dynamo reset, so interleaved arms stay captured side by side), `profile_once` |
| [`records.py`](../../evaluation/bench/records.py) | what a record *is*: `SCHEMA_VERSION`, `KEY_FIELDS`, `resume_key`, `record_path`, `samples_path`, `append_record` (one `write` + `fsync`), `read_records` / `read_keys` (one torn trailing line tolerated), `record_files`, `latest` (last record per key), `record_columns` / `perf_columns` (the flat columns of a record and of a perf entry), `write_table`, `aggregate(results_dir) → results.parquet` (one row per perf entry — what `report.py` reads), `read_table`, `sha256` |
| [`metrics.py`](../../evaluation/bench/metrics.py) | `accumulator(ks, device)` / `accumulate(acc, ids, targets, num_targets=None, ranked=False)` (returns the chunk's per-row recall, the [sidecar](#per-query-sidecar)'s values) / `finalize(acc)` (`null` metrics when `n == 0`, `null_if_empty`) — recall, ndcg, precision, mrr at every `k` from one top-`k_max` list as float64 running sums on device; `ranked=True` scores against the oracle's own top-`k` prefix; `per_row`, `jaccard_at_k`, `drop_self` ([Self-match](#self-match)). `training/evaluate.py` keeps its own frozen copy, pinned to agree (`training/test_encode.py`) |
| [`algos.py`](../../evaluation/bench/algos.py) | the algorithm table: `ALGOS` name → class (`LiNRV1`–`LiNRV3`, `SilverTorch`; `Postfilter`, the harness's own), `FILTER_KINDS`, `BACKENDS`, `FILTER_MODE` (`clause` → `exact`), `DISPATCH` (the library's table plus the `Postfilter` row), `PATHS` **derived from it**, `filter_backend` (`cuda` → `triton`), `build(algo, item_embs, k=, backend=, …)` (construct + `register_index`, ≈ 25 lines), `build_filter`, `is_valid_combo`, `official_config` (`score_path` / `recheck` → `OfficialConfig`) |
| [`postfilter.py`](../../evaluation/bench/postfilter.py) | `Postfilter`, the generic-torch baseline ([below](#the-postfilter-baseline)) |
| [`config.py`](../../evaluation/bench/config.py) | `Dataset`, `Job`, `load_dataset`, `load_matrix` — the config matrix below; `interleave_units`, `shared_key` (the [interleave groups](#interleaved-groups)) |
| [`inputs.py`](../../evaluation/bench/inputs.py) | `load_inputs` (dispatch to `training.encode.encode_split` or the `eval_datasets.layout` text readers; `users_limit` once, as a prefix), `sweep_qa`, `build_filters` (keyed by filter backend), `exact_filter`, `query_pool` |
| [`oracle.py`](../../evaluation/bench/oracle.py) | the exact filtered oracle as blob v4, `attrs_digest`, `pass_rate` |
| [`run.py`](../../evaluation/bench/run.py) | `run(jobs, out_dir=...)` — the cell loop; `MODES`, `QUALITY_CHUNK = 16`, `QUALITY_CPU_THREADS = 1`, `EXACT_ALGOS`, `SEED_FREE_QUALITY`, `PER_K_QUALITY`, `PERF_STAT_KEYS`, `IDS_PROBE_BATCHES`, `CLOCK_DRIFT` |
| [`cli.py`](../../evaluation/bench/cli.py) | `bench run` / `campaign` / `oracle` / `check` / `upload` / `fetch` / `merge` / `report` / `env` |
| [`report.py`](../../evaluation/bench/report.py) | `bench report`: `results.parquet` and `report.md` (provenance, the citability verdict, coverage, one line per cell, the failed / partial / unstable cells). See [Report](#report-reportpy) |
| [`upload.py`](../../evaluation/bench/upload.py) | `bench upload`: publish a results tree (records, samples, a freshly aggregated `results.parquet`) to the HF results repo with a `MANIFEST.json` (provenance + a sha256 per file) and a generated README; `bench fetch`: one subtree back, checked against its manifest. See [Results storage](#results-storage) |
| [`merge.py`](../../evaluation/bench/merge.py) | `bench merge`: the records of several fetched trees as `records.parquet`, `timings.parquet`, `samples.parquet` and `manifest.yaml`, one accepted record per cell. See [Merged tables](#merged-tables-bench-merge) |

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

| algo | `none` | `clause` / `bloom` | `cuda` |
|---|---|---|---|
| `linr_v1_filter_mask` (`PostfilterKNN`, fp16 cuBLAS + mask) | `cublas` (triton and torch collapse) | `cublas+triton` / `cublas+torch` (the filter's kernel); no bloom cell | — |
| `linr_v2` (`PrefilterKNN` over the filter's candidate list) | — (the candidate source is the filter) | `triton` / `torch`; no bloom cell | — |
| `linr_v3` (`OneBitKNN` top-`candidate_pool` → `PrefilterKNN`) | `triton` / `torch` | `triton` / `torch`; no bloom cell | — |
| `silvertorch` (IVF + INT8, predicate fused: `filter_mode` none / exact / bloom) | — | — | `cuda` (its only backend) |
| `postfilter` (the harness's baseline: fp16 cuBLAS, top-`alpha*k`, then the filter) | — | `cublas+torch` (no triton cell); no bloom cell | — |

A bloom cell exists on `silvertorch` only (the bloom filter is SilverTorch's;
LiNR runs the exact filter):
`PATHS` is `None` on every other algo's bloom triple, so `load_matrix` skips
them.

`cuda` is Meta's SilverTorch code, adapted (`silvertorch/`), and exists for `silvertorch` only;
its standalone filter modules are Triton (`algos.filter_backend`); its `OfficialConfig` params
are `score_path` and `recheck`; it is capturable (META-FORK C4: no host sync, shape functions for
compile), so it records `graph` entries. Its sources are outside
`LIB`, so `code_version` does not see a change there; the record's
`env.silvertorch_build.so_sha256` identifies the build. Records written before RENAME carry the
backend's old name, `official-fork`; `records.read_records` maps it to `cuda` (`RENAMED_BACKENDS`),
so `bench merge` and `bench report` see one name. `official` (Meta's code as published, which left
the workspace at RENAME) keeps its label in those records.

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
n_lists`. The build param `bloom_path` (`partial` | `full`, the S9 co-design ablation) ran on
Meta's code as published and left with it (RENAME): a config that still carries it is a
`ConfigError`; the FR records keep it in `params`. The build param `recheck` (bool, META-FORK
C13) sets `OfficialConfig(recheck=…)` on `silvertorch / bloom` through `algos.official_config`
(anywhere else `load_matrix` raises `ConfigError`): `false` is Meta's
bloom, false positives included, which recall then counts as misses
against the exact filtered oracle; without the key the library default
runs, the recheck. The study's bloom suites set it
`false` (the user's filtering directive of 2026-10-12: no recheck in
timed code). `params` are the merged build + query params over
`algos.SILVERTORCH_DEFAULTS` (`n_lists 1024, n_probe 24, n_iter 10`); on
`silvertorch` bloom cells `run.py` merges the suite's `bloom` defaults
(`m_bits`, `k_hash`) in as well. The probe-pool check is the library's,
in `set_query_params` and at forward.

### The postfilter baseline

`postfilter` ([`bench/postfilter.py`](../../evaluation/bench/postfilter.py))
is the study's baseline: what a
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

### Query preparation

Every filtered arm encodes its batch's query attrs once, outside the timed
call, through its `prepare_queries` (the user's rule, 2026-10-10: forward is
the algorithm; preparation is reported, not hidden):

| arm | `prepare_queries` returns |
|---|---|
| `silvertorch`, bloom | Meta's parsed plans, uploaded once (`plan_blob`), one per distinct expression |
| `silvertorch`, exact | the int32 attrs (`query_attrs`) |
| LiNR V1-V3, `postfilter` | the standalone filter's encoding: bloom query signatures, or the int64 attrs |

`run.perf` prepares every batch of a cell's fixed pool before its first
timed call (`run.prepare_pool`), each module with its own encoding. Every
timed and hashed call then takes `(query batch, prepared batch)`. The mean
cost of one batch's preparation (host + device, synced) is recorded on every
entry as `query_prep_ms`. The quality pass and the compile warm-up prepare
each chunk the same way.

## Config: one YAML per dataset + `suites.yaml`

The files under [`evaluation/config/`](../../evaluation/config/):
[`goodreads.yaml`](../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../evaluation/config/arxiv.yaml),
[`yambda-500m.yaml`](../../evaluation/config/yambda-500m.yaml),
[`pubmed.yaml`](../../evaluation/config/pubmed.yaml), the
synthetic-selectivity siblings `goodreads-synth.yaml`, `arxiv-synth.yaml`,
`pubmed-synth.yaml` (the dot-product build) and `laion{1,3,10,30}m-synth.yaml` plus arXiv's
cluster-correlated `arxiv-corr-synth.yaml`, and each one's second and third draw `<name>-synth-s1.yaml` /
`-s2.yaml` ([datasets](datasets.md#synthetic-selectivity-attrs)) (goodreads, arxiv and
pubmed in the `filter` suite, pubmed at 768; yambda-500m is unfiltered and
read by its own suite), and
[`suites.yaml`](../../evaluation/config/suites.yaml). `users_limit:
10000`. The grid is one backend per algo (`triton` for LiNR, `cuda` for `silvertorch`; records
written before RENAME carry `official-fork` / `official`), without LiNR V4 (it was ours, not
LiNR's, and is out of the harness) and without a dim ablation.
[`tests/bench/test_config.py`](../../evaluation/tests/bench/test_config.py)
pins the grid: every (suite, dataset)'s job and cell counts and the
rules above on every cell (G-grid), and the resume keys of cells the
redesign kept, as `b96e1f2` computed them (G-key). It also runs
every `config/*.yaml` against every suite through `load_matrix`: a listed
dataset expands to jobs, an unlisted one is refused by name and still
resolves at each of its dims.

`suites.yaml` holds the study grid. Seed 0 everywhere; seeds `{0, 1, 2}` in `filter` and in the `seeds-*` suites (the
interleaved rounds are the timing repeats elsewhere). Batch sizes are `{1, 16}` (bs 64 only in `codesign-laion30m`,
`c7-scorepath`, `nsweep-codesign` and `seeds-c7`) and ks `{100, 1000}` or a subset. Every SilverTorch cell scans at most a quarter of the
lists (`n_probe` ≤ `n_lists` / 4), and its `n_lists` / `n_probe` come from the tune ([IVF tuning](#ivf-tuning)).
No LiNR V4. SilverTorch runs only `none` /
`bloom` cells in these suites (the study's filter is LiNR's; its clause cells exist in `PATHS`).
There is no unfiltered `quality` suite; synth's
`p1` sweep is the unfiltered point. The cell counts per (suite, dataset) are pinned in
`evaluation/tests/bench/test_config.py` (`GRID`).

| suite | datasets | arms |
|---|---|---|
| `filter` | goodreads, arxiv, pubmed, three kept sweeps each; seeds 0-2 | V1, V2, V3 (pool 5000) triton and `postfilter` alpha {1, 8}, clause only; Meta's bloom at the dataset's tuned `n_lists` and `n_probe` {24, n95} (PubMed 4096 × {24, 512, 1024}) |
| `deep` | goodreads, arxiv, kept bloom sweeps; the exact V1 / V2 references come from `filter` | Meta's bloom along `n_probe` {8 … `n_lists` / 4} at the tuned `n_lists` (goodreads 4096, arxiv 2048) |
| `synth` | goodreads-, arxiv-synth (uniform) and arxiv-corr-synth (cluster-correlated, `c001` `c003` `c01`) ([datasets](datasets.md#synthetic-selectivity-attrs)) | V1, V2 triton clause at every rate (goodreads-synth ten, arxiv-synth nine: C1); official bloom 24 on arxiv-synth `p001`, `p1`; V3 `candidate_pool_frac` {0.01, 0.05} and `postfilter` {1, 8} on goodreads-synth's first seven rates and arxiv-corr-synth; V2 torch eager and compiled on goodreads-synth `p001 p01 p1`; LiNR and the postfilter clause only. k 1000 is dropped where N·p < 4000 |
| `c3` | goodreads-, arxiv-, laion30m-synth `p001 p01 p1`, clause, k 100 | V1 triton vs torch.compile max-autotune; plain torch on goodreads-synth |
| `c3-real` | the `filter` kept sweeps of goodreads, arxiv, pubmed | V1 triton vs torch.compile max-autotune |
| `codesign` | arxiv (kept sweeps), goodreads `c0_genre` | Meta's bloom along `n_probe` {8, 32, 128}, `n_lists` arxiv 1664 / goodreads 1024, k 100 (the partial-vs-full ablation ran on the published code: FR records `official`, `params.bloom_path`) |
| `bloomwidth` | goodreads `c0_genre`, arxiv kept, pubmed `c0_mesh` | Meta's bloom `k_hash` {3, 5}; bs 16; quality only (`perf: false`) |
| `bloomwidth-timed` | the same | Meta's bloom at `k_hash` 5, k 100, bs 16, timed |
| `v3bits` | goodreads-synth (its first 7 rates), goodreads (`filter`'s kept sweeps); pubmed d768 (kept sweeps) | V3 triton, `candidate_pool_frac` {0.01, 0.05}, clause only; goodreads `k_bits` {64, 128}; pubmed `k_bits` {256, 768} (256 divides 768 and sits below LiNR's 512; 768 is the default) |
| `h2h` | goodreads `c0_genre`, arxiv `c0_maincat`, `none` + `bloom`, d128 | SilverTorch `score_path` {fp16, int32}, `n_probe` 24, bs {1, 16}, k {100, 1000}; one interleave group per cell; run with `--interleave --profile` |
| `laion30m`, `laion30m-synth` | laion30m d256 `c0_domain`, `tags4` (pass ≈ 0.0095 / 0.0027); laion30m-synth nine rates; seed 0, k 100, clause | V1, V2 triton (bs 16; synth bs 1 + 16). One V1 + V2 process per sweep (68 GB) |
| `laion30m-x` | laion30m `c0_domain`, `tags4`, bs 1 + 16 | V2, then V1 alone |
| `codesign-laion30m` | laion30m `c0_domain`, `tags4`, bs 16 + 64, eager | Meta's bloom, 16384 × {32, 128} (the FR's partial / full ablation: records `official`) |
| `c7-scorepath` | laion30m `c0_domain`, `tags4`, bs 16 + 64 | SilverTorch at `score_path` {fp16, int32}, bloom, 16384 × {32, 128}, one group per sweep; `--interleave --profile` |
| `nsweep-synth` | laion1m-, laion3m-, laion10m-, laion30m-synth and arxiv-synth at d256, nine rates; seeded nested LAION subsets ([datasets](datasets.md#subsets)) | V1, V2 triton interleaved |
| `nsweep` | laion1m / 3m / 10m / 30m `c0_domain`, `tags4`, bs 1 + 16 | V1, V2 in one interleave group per sweep; at 30 M V2, then V1 alone (memory) |
| `nsweep-codesign` | laion1m / 3m / 10m / 30m `c0_domain`, `tags4`, bs 16 + 64, eager | Meta's bloom at `n_probe` {32, 128} at the scaled `n_lists` (the FR's partial / full ablation: records `official`) |
| `seeds-c1`, `seeds-c1-laion30m`, `seeds-c7` | the V1 / V2 crossover cells at bs 16 (arxiv-synth d128 p 0.2 / 0.5, laion30m-synth p 0.1 / 0.2) and the 30 M score-path cell at bs 64 | seeds {0, 1, 2}; own suites, so these keys are separate from the main grid's seed-0 cells of `synth` / `laion30m-synth` / `c7-scorepath` |

SilverTorch has the `cuda` backend only and LiNR and the postfilter
have no bloom cell at all; YFCC is out of the study.

### IVF tuning

SilverTorch's `n_lists` and its matched-recall `n_probe` depend on the dataset and are tuned, not
the library default 1024. **n95** is the smallest `n_probe` reaching `recall_oracle@100 ≥ 0.95` on a
dataset's median kept sweep. IVF-TUNE measures it quality-only (seed 0, bs 16, `n_probe`
doubling, capped at `n_lists` / 4 = 25 % of the items scanned, user) on goodreads and PubMed, per
`n_lists`, and picks the point that scans the fewest items (`n_probe` · N / `n_lists` + `n_lists`). The picks per
dataset are the `n_lists` / `n_probe` values in `suites.yaml`. The tuning
records are artifacts in their own results tree, never campaign cells. `n_lists` starts at ≈ 4 √N to a power of two
and the cap decides where 0.95 is out of reach: that slot takes `n_lists` / 4. An n95 is written only where it was measured. Each value is one
`datasets:` slot per arm: `filter`'s SilverTorch arms get `{build: {n_lists: [L]}, query:
{n_probe: [24, n95]}}`, and `synth`'s get `n_lists` only, because it sweeps `n_probe`
explicitly (user, 2026-10-08). A slot left `{}` runs at the library default `n_lists` with
`n_probe` 24 only. n95 / `n_lists` depends on the pass rate as well as N: goodreads `c0_genre`
(0.33) needs `n_lists` / 64, and PubMed `all5` (0.018) needs more than `n_lists` / 16.

`codesign` was the S9 ablation of the bloom path on Meta's code as published: `partial` (fused partial masks over the
probed clusters) against `full` (a full-N bool mask, then IVF); the FR records carry `bloom_path` in `params`, so
their keys do not collide with the `filter` / `deep` cells, and latency and `peak_fwd_mib` are the compared fields.
Since RENAME the suite is one arm (the adapted code evaluates the bloom over the probed clusters only).
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
# normalize: false                                                   # content_dir only: score raw dot products (pubmed)
# self_match: true                                                    # optional: queries are catalogue items (Self-match)
filters:                                                              # optional
  attrs: item_attrs_narrow.pt                                         # relative to data_dir
  reverse: clause_is_reverse_narrow.pt                                # optional
  # query_attrs: query_attrs_synth.pt  # optional .pt [U_full, C]; default eval_split.parquet
  clause: {c0_genre: [0], c1_lang_reverse: [1], all4: [0, 1, 2, 3]}   # name: active clauses
  bloom: {c0_genre: [0], old_sweep: {clauses: [2], disabled: true}}   # long form: disabled
```

`gt_dir` is derived (`<data_dir>/gt_d{dim}`). Unknown keys
raise `ConfigError` naming the file. `normalize` (default `true`) decides whether a
`content_dir` dataset's items and queries are L2-normalised at load, so that the oracle and
every algo score cosine; `false` keeps the stored vectors and every arm and the oracle score
the raw dot product (PubMed: MedCPT is trained for it; user, 2026-10-13). It is refused on a
`checkpoint` dataset, and with `false` the loader refuses vectors whose sidecars do not say
`normalization: "none"` ([datasets](datasets.md#the-layout-contract-layoutpy)).

```yaml
# config/suites.yaml — a suite = cells run on every listed dataset × the dims both list
filter:
  datasets: [goodreads, arxiv, pubmed]
  dims: [128, 768]                  # optional; default: the dataset's dims
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
      backends: [cuda]
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
no meaning: `bloom_path` anywhere (it left with the published code), `score_path` off silvertorch / cuda, `recheck` off silvertorch / bloom, `m_bits` /
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
bench merge    TREE [TREE ...] --out DIR                      # fetched trees -> records / timings / samples tables + manifest
bench report   [results] [--out DIR] [--gate STEP]                          # results.parquet + report.md
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
the config's encoder. It exists for comparisons against records made on an older encoder.

`bench campaign` is the process loop: for every suite (in the
order of `SUITES = ("filter", "deep", "codesign")` for `all`), every listed dataset and every
`(dataset, dim, algo, backend)` group of `load_matrix`, one child process
`python -m bench.cli run --dataset … --dim … --suite … --algo …
--backend … --resume` (same interpreter, `cwd = evaluation/`), sequential.
With `--interleave` the groups one [interleave unit](#interleaved-groups)
spans share a child (`cli._children`: connected groups of `config.interleave_units`),
run as `--algo a … --backend b … --interleave`; the set has to be exactly one
`--algo` × `--backend` product, else the campaign stops with the groups named,
and the log name joins them with `+` (`codesign_arxiv-d128_silvertorch_cuda.log`,
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
blob](#oracle-blob-v4) of every sweep the suite's jobs read, one per
`(dim, sweep, k_max)` (a `none` cell's is the unfiltered exact top-k, shared by every suite on
the same inputs), in its own process, with no module and no timing
(`run.prebuild_oracles`); it exits 1 when the narrows select no sweep.
Run it before a campaign: `bench run` still builds a missing blob, but warns
that it should have been prebuilt.
`bench check --dataset D` runs `eval_datasets.layout.validate_layout` on
the dataset's directory at every dim (files, prefix sidecars, the row
alignments the readers enforce) and exits 1 on any problem — run it before
a campaign on a freshly staged dataset.
`bench merge` turns any set of fetched trees into three typed tables and a
manifest ([Merged tables](#merged-tables-bench-merge)).
`bench env` prints `measure.provenance() | measure.clocks()` as one JSON
object — the provenance fields of the record's `env` block plus one
`nvidia-smi` sample under `clocks()`'s names (`sm_mhz`, `mem_mhz`,
`sm_max_mhz`, `power_limit_w`) — to paste next to a number measured outside the harness.

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
   `sweep_qa` → `build_filters` → `exact_filter` (none on a `none` cell) +
   `oracle.load_or_build` (blob v4 at `k_max = max(ks)`, `k_max + 1` on a
   [self-excluded](#self-match) cell; a `none` cell's blob is unfiltered and keyed with
   `run.NO_ATTRS`, so it does not depend on the dataset's attrs); the row masks: `keep` (not
   skip-masked), `oracle_rows` (kept with ≥ 1 survivor), `heldout_rows` (kept, ≥ 1 target, and on filter cells
   `target_in_filter`);
4. per job: `measure.setup(job.seed)`, one `timed_build` of
   `algos.build(...)` with the first cell's params; `index_mib`,
   `filter_mib` (the module's `filter` submodule, 0 without one); a build
   failure writes a `failed` record per cell of the job;
5. per cell: `set_query_params` for the query part of `params`; quality
   (`module.k = k_max`, `k_max + 1` on a [self-excluded](#self-match) cell
   and back to `k_max` after the pass; chunks of 16 kept rows, `accumulate(...,
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
6. after each unit (one job, or one interleave group): release its arms
   ([Arm release](#arm-release)).

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
cannot leave a resumable record without its vector; `records.read_records` ignores
(and logs) one torn trailing line — the cell in flight when the process
died — and raises on a malformed line anywhere else.

### Self-match

On arXiv and PubMed (and arxiv-synth, arxiv-corr-synth, pubmed-synth and their draws, which share the parent's queries) a query is a
catalogue item's own text and its single held-out target is that item, which stays in the index
([datasets](datasets.md#arxiv)), so the exact oracle's top-1 is the query itself and so is every
method's when it finds it. Those dataset files set `self_match: true` (`Dataset.self_match`,
default false; LAION queries are never in the catalogue, Goodreads' are user
histories). On such a dataset every **filter** cell is self-excluded (`Job.self_excluded`; a
`none` cell's oracle keeps the own item and its key is unchanged):

- the self id of query `u` is its held-out target `targets[u, 0]`;
- the oracle is requested at `k_max + 1` with `slice_wider=True`: a cached blob of the same
  inputs at a larger `k_gt` (the k 1000 blob serves k 101) is cut to its prefix in memory
  ([Oracle blob v4](#oracle-blob-v4)), else one is built at `k_max + 1`;
- `metrics.drop_self(ids [B, K], self_ids [B]) → [B, K − 1]` removes the row's own item
  wherever it ranks (order kept; a row without it loses its last entry; `-1` stays `-1`), applied
  to the oracle row (`sweep_assets`' `oracle_topk`, `[U, k_max]`) and to the module's
  `k_max + 1` ids before `accumulate(..., ranked=True)`; `oracle_rows` then needs a survivor
  other than the own item. `PER_K_QUALITY` arms run at `k + 1` the same way;
- the **held-out** side keeps the un-excluded first `k_max` ids (its target *is* the own item),
  and so do the parity spill and the per-query sidecar's `heldout_recall@k`;
  `recall_oracle@k` there is the self-excluded value. Held-out recall on these datasets measures
  **self-retrieval** (does the method return the query's own document), never end-task
  recall: every cell of a `self_match` dataset, `none` cells included, records
  `quality.heldout_label: "self_retrieval"`;
- the timed path is untouched: `perf` sets `module.k = k` per entry as before.

`sweep_assets` logs, per sweep, the share of kept rows with a reachable target whose oracle top-1
is the own item (near 1.0 when the flag is right; a near-duplicate document can outrank it, and `drop_self`
removes the own item wherever it ranks) and the record carries it as
`quality.self_top1_share`, next to `quality.self_excluded: true`. The key block gains
`self_excluded: true` on these cells only (`Job.key`, `records.key_block`), so their resume and
quality-cache keys, sidecar names and parity spills differ from every record written before the
rule ([Resume](#resume)), while every other cell keeps its key.

### Arm release

A unit's modules (index, filter, compiled and captured callables) die before
the next unit builds, so one process runs several 10-30 M arms in a row
(30 M arms ran out of memory when the last arm's index stayed resident).
`run.py` notes `torch.cuda.memory_allocated()`
once the unit's inputs and sweep assets are ready (`base`); at the unit's end
it drops every name that still holds an arm (`built` and the per-arm
`module` / `b` locals), then `gc.collect()`, `torch._dynamo.reset()`,
`empty_cache()`, and raises if more than `ARM_FREE_SLACK_MIB` (256 MiB) over
`base` is still allocated: a leaked index fails the run loudly instead of the
next build failing on an out-of-memory error. The arms of one interleave group stay resident
together by design (they are timed round-robin).

A failure frees its tensors too. The build, cell and perf failure paths release
*after* their `except` block, once the exception (whose traceback holds the
half-built tensors) is gone, and they log through `run._log_failure`
(`logger.error` with `traceback.format_exception`) instead of
`logger.exception`: loguru's default `diagnose` reads `run`'s own `f_locals`,
and on Python 3.11 that snapshot stays cached on the frame, keeping `exc`
and with it the failed arm's tensors for the rest of the run (each retry of a
failed 30 M build saw more memory held).

### Interleaved groups

A suite's `interleave:` list names comparison groups: `by` is the field the
arms differ in (`algo`, `backend` or build params such as
`score_path`; one name or a list), `values` (one `by` field only) limits which
arms join. `config.interleave_units(jobs)` puts the jobs whose key is equal but
for the `by` fields (query params aside, the seed included, so a group is per
seed) in one unit, in matrix order; every other job is a unit of its own. A
job in two groups of two or more is a `ConfigError` at `load_matrix`. The
shipped groups: `filter` and `synth` V1 vs V2 (per backend and compile mode);
`h2h` the cuda arms by `score_path` (`codesign` has none since RENAME).
`bloomwidth-timed` has none.

Without `--interleave` nothing changes. With it, `run` builds every arm of a
unit (assets per arm's step-3 key, usually shared), then per query combo runs
each arm's quality in order and times all arms together: per `(bs, k, mode)`
one `measure.latency_group` call warms and calibrates each arm (its own `N`)
and then runs window `i` of every arm before window `i + 1` (A, B, A, B, A, B
over the 3 windows) inside one process. In
`graph` mode dynamo is reset once before the variant, then each arm is
captured (`graph_callable` no longer resets, so captures coexist); an arm that
cannot capture (a `max-autotune` arm) gets its null `not_capturable` entry and the others
run round-robin without it. Each arm writes its own record, under the key a
non-interleaved run gives it, with `interleave: {group, arms, position}`:
`group` is the first 16 hex of the sha1 of the cell's key block minus the `by`
fields (`config.shared_key`; the seed and query params included), `arms` the
labels in run order (`algo/backend` plus each `by` build param, e.g.
`silvertorch/cuda/score_path=int32`), `position` this arm's index. Its
measured perf entries carry `rounds` (the window count, equal across the
group), and window `i` of every arm is round `i`, so a paired ratio can be
read window by window. Resume works per arm, but a group whose arms are not all
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
overstates the seed's reach. It stays as recorded: records are never rewritten.

### Bloom widths as build params

`m_bits` / `k_hash` in an arm's `build:` grid (silvertorch / bloom only,
else a `ConfigError`) become that job's `bloom`. It feeds the module's
index, the record's `bloom` and `index_mib`. A bloom cell has no standalone
filter (`filter_mib` 0): the bloom is SilverTorch's own, and the exact
oracle does not depend on the width and is shared. A cell that does not grid
them keeps the suite default outside `params`, so its key is byte-for-byte
the pre-grid one. The `bloomwidth` suites grid them. With `OfficialConfig(bloom_width="upstream")`
`SilverTorch` ignores `m_bits`: the index's width is then
`OfficialConfig.b_multiplier`'s, and `k_hash` is its search `k`.

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
× mean pass count over the sweep's oracle rows))` (`run.passing_mean`). On uniform synth every
query passes the same items, so the mean is the achieved pass count (N·p in
expectation); on arxiv-corr-synth it is the mean of the queries' cluster sizes. The key keeps the fraction, the module gets the int, and
the record's `candidate_pool` carries it, with the mean it is a fraction of as
`pool_passing_mean`, so the effective pool reads as `candidate_pool / pool_passing_mean` of the
passing set (above 1 when the floor exceeds it); both are `results.parquet` columns.

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
| `self_excluded` | bool, only when true | in the key block of a [self-excluded](#self-match) cell; absent everywhere else |
| `path` | str | `PATHS[(algo, filter_kind, backend)]` |
| `n_items`, `n_queries` | int | catalogue size, queries after `users_limit` |
| `n_kept` | int | queries not skip-masked (every query on `none` cells) |
| `n_queries_oracle` | int | kept queries with ≥ 1 survivor (the oracle metrics' `n`; every kept query on `none` cells) |
| `n_queries_heldout` | int | queries the held-out metrics cover (kept, ≥ 1 target, `target_in_filter` on filter cells) |
| `n_targets_in_filter` | int | held-out targets those queries are scored against: every valid target on `none` cells, only the ones the exact mask admits on filter cells |
| `pass_rate` | float | exact mask pass rate over kept queries (`1.0` on `none`). Records written before the field was retired also carry `bloom_fp_rate`, our own hash's false-positive rate; `results.parquet` and `bench merge` drop it |
| `bloom` | dict / null | `{m_bits, k_hash}` on bloom cells: the suite default, or the cell's own when `params` grids them ([bloom widths](#bloom-widths-as-build-params)) |
| `k_max`, `ks`, `batch_sizes` | | the suite's, `k_max = max(ks)` |
| `build_s` | float | construction + `register_index`, sync on each side (same value on every cell of one build) |
| `compile_s` | float / null | a `compile` arm's first forward (one quality chunk), apart from `build_s` ([compiled arms](#compiled-arms)); `null` elsewhere |
| `candidate_pool` | int / null | the `candidate_pool` the module ran with when `params` sets one: the literal, or the value `candidate_pool_frac` resolved to ([pool fractions](#pool-fractions)) |
| `pool_passing_mean` | float / null | on a `candidate_pool_frac` cell, the mean exact pass count over the sweep's oracle rows that the fraction is of ([pool fractions](#pool-fractions)) |
| `index_mib` | float | Σ buffers of the algo module, filter submodule included |
| `filter_mib` | float | Σ buffers of the filter submodule alone (`0.0` without one; `silvertorch` carries its attrs inside `index_mib`) |
| `seed_scope` | str | `pool` on the seed-free arms (`SEED_FREE_QUALITY`: the seed moves only the perf pool), `pool+build` on SilverTorch and `linr_v3`; for `linr_v3` at `k_bits = D` the label overstates the seed's reach, because its quality is seed-free ([Quality cache](#quality-cache)) |
| `interleave` | dict / null | `{group, arms, position}` when the cell ran in an [interleave group](#interleaved-groups); `null` otherwise |
| `quality_source` | dict / null | `null` when this record computed its quality; `{seed, code_version}` of the record the [quality cache](#quality-cache) copied it from |
| `per_query` | str / null | the [per-query sidecar](#per-query-sidecar) relative to the results root; `null` without quality |
| `quality` | dict / null | `heldout: {recall@k, ndcg@k, precision@k, mrr@k for k in ks, n}`, each metric `null` when `n == 0`; `oracle: {…}` (ranked-prefix targets; the unfiltered exact top-k on `none` cells); `jaccard_vs_first@k` per `k`, `score_max_abs_diff`, `parity` (`reference` = this record wrote the spill file, `vs_<backend>` = compared against it, `shape_mismatch:…`); on a [self-excluded](#self-match) cell `self_excluded: true` and `self_top1_share`; on every cell of a `self_match` dataset `heldout_label: "self_retrieval"` (its held-out recall is self-retrieval, not end-task recall); `null` with `--skip-quality` |
| `perf` | list / null | one entry per `(bs, k, mode)` (table below); `null` with `--skip-perf` and on a quality-only suite (`perf: false`) |
| `unstable` | bool | any perf entry `unstable` (window spread > 5 %), or `clocks_drift` |
| `memory_reserved_mib` | float / null | `torch.cuda.memory_reserved()` after the cell — the leak detector across a group's cells |
| `elapsed_s` | float | wall time of the cell |
| `env` | dict | `gpu, driver, cuda, torch, triton, silvertorch_build, commit, dirty, repo_dirty, git_branch, code_version, lib_dir, host, python, started, config_sha`; the process-start sample `sm_mhz_idle, mem_mhz, sm_max_mhz, power_limit_w`; the cell's `sm_mhz_load` (median of its perf entries' under-load samples; `null` without perf or CUDA), `clocks_drift` (any under-load sample > 5 % from the process's first) and `frac_windows_below_max` (the share of the cell's `window_sm_mhz` samples below `sm_max_mhz`, the device's max SM clock; `null` without samples or without a device max) |
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
| `query_prep_ms` | on every entry: the mean host + device ms of one pool batch's `prepare_queries` (the filter's query encoding, done before timing, § Query preparation); `null` on unfiltered cells and arms without a prepare step. Records before OFFICIAL-REWORK carry `cache_plans` instead (`false` = the official expression parse was inside every timed forward) |
| `load` | `"closed_loop"` |
| `kernels` | `--profile`, eager only: top-8 CUDA kernels `{kernel, us, calls}` |
| `kernels_us`, `kernels_calls` | `--profile`, eager only: device µs and kernel launches summed over every kernel of the call, sentinels and `## …` profiler ranges excluded (records before H-KSUM lack them; records between H-KSUM and its fix double-count compiled calls, which no eager `h2h` arm is; T3 reads these, and for such a record the top-8 sum, marked as a lower bound) |
| `kernel_scopes` | `--profile`, eager only: `{scope: {us, calls}}` for `scorer`, `topk`, `epilogue`, `other` (`measure.KERNEL_SCOPES`, by substring of the kernel name, first match wins), adding up to `kernels_us` / `kernels_calls`. `scorer` holds the scoring scope: Meta's `fused_kmean_ann::` (`process_cluster*`, payload and warp-size kernels) and `bloom_search::` kernels; `topk` the torch top-k / sort kernels (`mbtopk`, `sbtopk`, `radixSortKVInPlace`, `bitonicSortKVInPlace`); `epilogue` the index / gather / scatter kernels. In `results.parquet` as `perf_kernels_<scope>_us` / `_calls` (older records lack them) |
| `ids_sha256` | sha256 of the ids the timed callee (the eager module or the graph replay) returns on the first 8 batches of that `(bs, seed)` pool (`run.IDS_PROBE_BATCHES`), int64 row-major, batch after batch, run once after the windows; `null` when the variant did not run. Equal eager and graph hashes are the eager-vs-graph identity gate. Graph mode is inductor's code, not the eager ops replayed, so this gate catches arithmetic that inductor lowers differently. Records before library tree `5d158f20` differ in graph `quantize_int8` ([kernels](kernels.md#quantize_int8-retrieveindexing), validation, *V-GRAPH-IDS*) |
| `ids_sha256_canon` | the same ids with each row re-ordered by (score desc, id asc) before hashing: two backends with bit-equal scores whose tied ids come in another order hash equal (cuda int32 against Triton). Eager vs graph identity reads `ids_sha256`. An added field, schema stays 4 |
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
| `pass_count` | int64 | items the exact mask passes for that query (the oracle blob's `pass_counts`); `n_items` on `none` cells |
| `recall_oracle@{k}` | float32 | per-query recall against the oracle prefix, for every `k` in the record's `ks`; NaN where the row has no oracle (no survivor) |
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
every cell; a doc or plan edit invalidates none.
`run` reads the file once (`records.read_records`) and rebuilds the key from each record as
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
goodreads → `gsasrec-d{dim}-drop0.5-id` (what every schema-2 goodreads record
ran on); a text dataset → today's `config/<dataset>.yaml` resolved at the
record's dim, so a schema-2 arxiv / pubmed record keeps exactly the resume key a
new run computes (they were written with the same `content_dir`); any other
checkpoint dataset raises (no identity left). The derivation reads the repository's
`evaluation/config/`, not a run's `--config-dir`: it describes what those
records ran on, which is the checked-in config. `SCHEMA_VERSION` went 2 → 3 because the record
layout gained a key field.

`records.aggregate(results_dir)` writes `results.parquet` — `records.latest`
(the last record per key of every `<suite>/*.jsonl`), one row per perf entry
(one row with null perf columns when the record has none). Columns, in order: the key block (`params` as canonical
JSON); the record scalars `status, path, n_items, n_queries, n_kept,
n_queries_heldout, n_queries_oracle, n_targets_in_filter, pass_rate,
k_max, build_s, index_mib, filter_mib, unstable,
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
are in it, so the table alone carries a bootstrap's inputs. Types are inferred per column (int, double, bool, string,
list<string>), so a column that mixes types across records fails the
aggregation instead of reaching a table; the column set is the union over
the records, so it varies with `ks` and schema version. Parquet over CSV
because the table is typed (no string round trip to re-guess `True`, `""` or
`1e-05`), several times smaller than the JSONL it comes from, and read in one call by polars, pyarrow or DuckDB.

Two dirty flags, both `null` outside a git checkout: `env.dirty` is
`git status --porcelain -- .` in the imported package's directory (untracked files
included — a new kernel module is measured code too) and is the flag
`report.py` refuses to cite; `env.repo_dirty` is the tracked
files anywhere else (`--untracked-files=no`, informational: a docs or
harness edit). The results tree is gitignored, so a campaign appending to
its JSONL never flips it.

### Merged tables (`bench merge`)

`bench merge TREE [TREE ...] --out DIR` reads any set of results trees as
`bench fetch` leaves them (one per Hub subtree; the directory name is the
`subtree` label, so `bench merge fetched/fr/* --out tables/` merges a whole
run) and writes four files under `DIR`, `merge.TABLES_VERSION` = 1. The
records are read as they are, never rewritten; two retired things are left
out on read: the `bloom_fp_rate` field, and every bloom record of an algo
other than SilverTorch (LiNR's and the postfilter's; a sidecar's lines still
pair, then the record is dropped). A tree with no record files (an untimed
or config subtree) is listed in the manifest with none.

**The accepted record.** A cell is a resume key (key block +
`code_version`). Among the records of one cell, wherever they ran, the
accepted one is the `ok` record over a `partial` over a `failed` one, a
stable (`unstable` false) over an unstable one, then the latest
`env.started`, then the later line; every other record of that cell is
superseded by it (`superseded_by` names the accepted record's subtree).
This is the rerun policy as data: a `-rerun` subtree's stable record
replaces its leg's unstable one, a cell unstable in every run keeps its
record and the flag, an OOM `failed` line gives way to the retry that
followed it. The host is a plain column (`env_host`), not part of the rule.

| file | one row per | columns |
|---|---|---|
| `records.parquet` | record line of every input tree | `subtree`, `cell` (sha1 of the resume key, 16 hex), `accepted`, `status`, `unstable`, the key block (`params` as canonical JSON), `env_host`, `env_started`, `env_code_version`, `env_commit`, `tag` (the git tags at `env_commit` in the checkout the merge runs in, comma-joined; null without), `file`, `line`, `superseded_by`; then every column `results.parquet` gives a record (`records.record_columns`: the scalars, `env_*`, `quality_source_seed`, `interleave_*`, `heldout_*`, `oracle_*`, `quality_*`) and the rest of `env` flattened (`env_driver` … `env_config_sha`, `env_silvertorch_build_so_sha256`; records written before RENAME flatten to `env_official_build_so_sha256` / `env_official_fork_build_so_sha256`) |
| `timings.parquet` | perf entry of every record | the same first twenty columns (through `tag`), then the `perf_*` columns of `results.parquet` (`records.perf_columns`: `perf_k`, `perf_bs`, `perf_mode`, the stats, `perf_window_medians_ms`, `perf_sm_mhz`, `perf_ids_sha256(_canon)`, `perf_rounds`, `perf_kernels_<scope>_us` / `_calls`, `perf_reason` on a variant that did not run) |
| `samples.parquet` | samples-sidecar line | the same first twenty columns, `k`, `bs`, `mode`, `n`, `ms` (list of double: the chosen window's per-call vector). A sidecar line is paired with its record by file order (the j-th line of a variant is the j-th record of that key with that variant timed; the sidecar and the record file are appended in the same cell order, samples first). A sidecar with more or fewer lines than the records' timed variants (a cell killed between its samples and its record) gives that file no rows, with a warning; a line that pairs with the wrong record raises; a tree without a sidecar gives no rows |
| `manifest.yaml` | — | `tables_version`, `built`, `rows`; `subtrees`: per input tree `path_in_repo`, `manifest_sha256`, `uploaded` (from its `MANIFEST.json` when `bench fetch` kept one, else null), `hosts`, `gpus`, `commits`, `tags`, `branches`, `code_versions`, `started` [first, last], `records` {status: n}, `unstable`, `accepted`, `superseded`, `supersedes` (the subtrees whose records it replaced), `samples`; `cells`: per `(dataset, suite, algo, backend)` the `subtrees` its accepted records come from, `cells`, `accepted_ok` / `accepted_partial` / `accepted_failed`, `seeds`, `code_versions`, `filter_kinds`, `sweeps` |

The tables carry the same values as the per-leg `results.parquet` (both are
built from `records.record_columns` and `records.perf_columns`); they add the
provenance the flat table drops and the accept verdict across trees. A row
of `timings` with `accepted` false is a superseded measurement, kept so the
tables are lossless. The per-query sidecars stay files, referenced by
`per_query`. `tests/bench/test_merge.py` pins the rule (an unstable original
and its stable rerun, a stable original and its unstable rerun, an OOM line
and its retry, a later record from another machine), the samples pairing (an
orphaned sidecar, a wrongly paired line), and the manifest's counts.

### Oracle blob v4

`oracle.load_or_build(gt_dir, sweep, k_gt, item_embs=, queries=, targets=,
qa_sweep=, skip_mask=, clauses=, filter_mod=, attrs_digest=, device=, slice_wider=False)` returns one dict,
cached at `<gt_dir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt`. With `slice_wider` (the
[self-match](#self-match) cells) and no blob at this fingerprint, the smallest cached blob of the
sweep whose fingerprint matches the same inputs at its own larger `k_gt` is returned with `topk`
cut to the first `k_gt` columns, `k_gt` set and `sliced_from` naming the wider one; nothing is
written. Its prefix can differ from a build at `k_gt` only in which of several exactly tied items
sit at the cut:

| key | value |
|---|---|
| `version` | `4` |
| `topk` | `[U, k_gt]` int64 exact filtered top-k (0-indexed, `-1` padded; skipped rows all `-1`) |
| `pass_counts` | `[U]` int64 items passing the exact mask; `-1` on skipped rows |
| `pass_rate` | mean of `pass_counts / n_items` over kept rows |
| `targets_in_filter` | `[U, T]` bool, target `t` of user `u` passes the mask |
| `target_in_filter` | `[U]` bool, any target passes |
| `n_items`, `n_queries`, `n_kept`, `k_gt`, `sweep`, `clauses` | shape of the build |
| `fingerprint` | sha256 over shapes, dtypes and a 64-row linspace sample (its float32 index clamped to the last row, which past 2^24 rows it overshoots) of `item_embs`, `queries`, `targets`, `qa_sweep`; the full bytes of `item_attrs` and `clause_is_reverse` (`oracle.attrs_digest`, computed once per `(dataset, dim)` in `inputs.load_inputs` as `inputs["attrs_digest"]`); plus `clauses` and `k_gt` |
| `code_version`, `harness_commit`, `torch`, `created` | provenance |

**Item-chunked** (`oracle.compute`): per batch of 64 kept queries the loop
walks the items in chunks of `ITEM_CHUNK = 2_000_000` rows: `q @ chunk.T` in
fp32 (the function raises unless TF32 is off and the matmul precision is
`highest`, which `measure.setup` sets), `-inf` where the exact filter's
`evaluate_mask(prepare_queries(qa), start, end)` (a `[B, end − start]` bool over that item
range) fails, a local `topk`, merged into the running `[B, k]` with ids
offset by the chunk start; equal scores go to the lower id (a stable sort by
id, then by score). `pass_counts` and `targets_in_filter` accumulate per
chunk the same way. No `[B, N]` score or mask exists and
`item_embs` is never transposed into a copy (the chunk's `.t()` is a view
cuBLAS reads transposed). The blob version and fingerprint did not change:
a blob built in one shot holds the same content up to exact-score ties
(the one-shot `torch.topk` promised no order among them).

The fingerprint is in the file name, so a stale blob is never read (a
different dim off the same `data_dir`, regenerated attrs — even one
edited value, since the item side is hashed in full — a retrained
checkpoint, another `users_limit` or clause set each produce a new file),
and the blob is a portable artifact. The filter passed in
is always exact (`inputs.exact_filter`: the clause module itself on `clause`
cells, a fresh `ExactAttributeFilter` on `bloom` cells) — bloom's false
positives never leak into ground truth. Bloom pass rates are not cached.

## Report (`report.py`)

`bench report <results> [--out DIR] [--gate STEP]` writes `results.parquet`
(`records.aggregate`) and `report.md` for one results tree: the provenance
block and the citability verdict, a coverage table (one line per cell group
with its seeds and status counts), one line per record (dataset, dim, suite,
filter, sweep, algo, backend, params, seed, status, `recall_oracle@k_max`
and the median ms of every timed `(k, bs, mode)`), then the failed cells
with their stage and error, the partial records and the unstable variants
with their spread. It is the way to get a run's numbers in one place; it
builds no paper table or figure (this repository reproduces no paper exhibit), and
across several legs the [merged tables](#merged-tables-bench-merge) are
the thing to read. A tree holding two `inputs` under one `(dataset, dim)`
is refused with the identities named.

**Citability.** Nothing is citable by default. `--gate
NAME` declares a validation gate passed, and only then can the report
say so, but the evidence vetoes the flag: a `failed` or `partial` record, an
`env.dirty` one, or one whose `env.git_branch` is not `staging` /
`development` / `main` keeps the verdict at NOT CITABLE with the reasons
listed in full (`provenance()["blockers"]`, also what `bench upload` puts in
the manifest; `["marks"]` is the short form). A `status: failed` record
never reaches a number; a `partial` record is marked `*`, an `unstable`
perf entry `†`. Latencies are read against the per-variant under-load
`perf[].sm_mhz`; the idle `env.sm_mhz_idle` is provenance only, never
compared with an under-load sample.

## Results storage

Nothing the harness writes is committed. The records are the product of a run; the box they
are produced on is rented and its disks are small; and
126 cells with their latency vectors are 78 MB, which is not a thing to keep
next to the code. So a results tree has two homes, one for each phase of its
life, and git holds only the pointer.

| what | where | why |
|---|---|---|
| `results/<suite>/<dataset>-d<dim>.jsonl` (the records), `*.samples.jsonl` and the `*.perquery/` sidecars while a leg runs | **local disk**, `results/` under `evaluation/` (gitignored; `--out` elsewhere) | the working state: `append_record` writes a cell and `fsync`s it, resume reads it back, and neither may depend on the network. Inside the repo checkout, which on a pod is `/workspace` and survives a restart |
| the same records, samples and per-query sidecars, plus `results.parquet`, once a leg finishes | **HF Hub**, `<owner>/eval-results/<leg>/` (`bench upload`; the owner from `RETRIEVE_HUB_OWNER`) | the archive. The JSONL goes up next to the Parquet because it is the lossless form: failed records with their tracebacks, superseded records, the nested `quality` and `env`, `window_sm_mhz`, `kernels` — everything the flat table drops — and it is what resume and the manifest's provenance read |
| raw dumps behind a documented finding (kernel timings, probe outputs, ETL logs) | **HF Hub**, `<owner>/eval-results/artifacts/<plan>/<same path>` | the finding lives in prose in the system pages; the dump is only for re-derivation |
| the [merged tables](#merged-tables-bench-merge) of a run's legs (`bench merge`) | **HF Hub**, `<owner>/eval-results/artifacts/<plan>/<tables>` | derived from the legs, rebuilt by one command; what a results script reads instead of the legs |
| `results/_parity/*/*.npz` (600–680 MB per run), `results/_logs/` | **nowhere** — deleted | rewritten by every run, and the parity *verdict* (`jaccard_vs_first@k`, `score_max_abs_diff`) is already inside the record. `bench upload` skips every `_`-prefixed path part |

A leg is finished when its records are on the Hub and verified: `bench upload
--results results --path-in-repo <leg> --verify`, after which the local tree
may be deleted. The Hub subtree's `MANIFEST.json` carries the sha256 of every
file it holds, and `bench upload` prints the manifest's sha256, so a note elsewhere can prove
what the Hub has without downloading it.

### `bench upload`

One invocation publishes one `--path-in-repo` subtree of one results tree:

```bash
uv run bench upload --results results --path-in-repo d1-a --verify
uv run bench upload --results /scratch/tmp/h1-hub/artifacts/kernel-opt \
    --path-in-repo artifacts/kernel-opt --dry-run
```

`--repo-id` defaults to `upload.results_repo()` = `<owner>/eval-results`, the owner read
from `RETRIEVE_HUB_OWNER` (required when the option is absent), beside
`eval_datasets.hub.eval_repo(dataset)` for *datasets*. When the tree holds records (`records.record_files`: any
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
  claim more than the tables would. A real upload prints the manifest's own sha256.
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
a release decision, not this command's.

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
files are fine, so fetching twice is a no-op. The subtree's `MANIFEST.json`
is copied in beside the files (`bench merge` reads its provenance; `upload.files`
never lists it, so a re-upload does not carry it); a tree already holding
another subtree's manifest is refused too, since one fetched tree is one subtree. A fetched tree is an ordinary
results tree: `bench report` reads it and `bench run --resume` appends to it.

`tests/bench/test_upload.py` pins both with no network: the scratch
exclusion, the checksums, the four ways evidence beats `--gate`, the generated
README, the round-trip check catching a changed byte, the commit's operation
list with the regenerated `results.parquet` and `private=True`, a fetched tree
that resume can read with its `MANIFEST.json` kept, and a fetch refusing to
overwrite a different local copy.

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
together (a prefix, not a sample, so quality stays comparable across runs). `sweep_qa(qa, clauses)`
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
trailing-whitespace hooks (the last two never on `silvertorch/`), and `scripts/check_doc_links.py`.

| file | checks |
|---|---|
| `test_dependency_direction.py` | every `import` / `from` in `bench/`, `training/`, `eval_datasets/` resolved with `ast`: `bench` → `training.encode`, `eval_datasets.{hub,layout}`; `training` → `eval_datasets.{hub,layout}`; `eval_datasets` → nothing; the library only from `bench`, through `retrieve` (the algo and filter classes) and `retrieve.interfaces` (`DISPATCH`, `FilterModule`). The walk must see more than 25 files, and an allow-list entry no import uses fails as stale |
| `test_env_readers.py` | every `os.environ.get` / `os.getenv` / `os.environ[...]` of a `RETRIEVE_*` name under `evaluation/` sits in its owner (`RETRIEVE_DATA_ROOT`: `eval_datasets/hub.py`, read through `data_root()`); same file-count and stale-owner guards |
| `bench/test_paths.py` | `PATHS == derive(DISPATCH)`: the derived table equals the harness's expected paths, the grid is complete, `DISPATCH` names every algo × backend and adds only `Postfilter` to the library's |
| `bench/test_records.py` | `resume_key` canonical and `code_version`-sensitive; append / read round trip; one torn trailing line; `aggregate` one row per perf entry, last record per key, typed columns; two jobs differing only in `--checkpoint` key apart; a real schema-2 arxiv key block keeps today's arxiv job's resume key once today's `self_excluded` is set aside (it is there, so the record never resumes the cell); a schema-2 goodreads record is gSASRec; the schema-4 columns (null on a schema-3 record) and `perf_window_medians_ms` equal to the nested entry `report._attach` joins |
| `bench/test_measure.py` | `stats` vs numpy, `latency` control flow, `latency_group`'s A, B, A, B windows (G-interleave), `index_bytes` dedup, `provenance` git fields, `silvertorch_build` (the extension's sha256 and flags 610 (git, local dir, no `direct_url.json`, not installed), `official_build` (the `.so`'s sha256, the flags file, absent package), `dirty` scoped to the library subtree with the `files:` fallback, H-PROVENANCE (`code_version` / `dirty` / `files_hash` follow an imported package in another checkout, a commit there moves the stamp, a wheel outside git gets the content hash), `atomic_write`, `clocks` shape, `graph_callable` refusals |
| `bench/test_metrics.py` | padding / IDCG / denominator contracts; running sums equal per-row means to 1e-9; `jaccard_at_k`; `drop_self` (own item at rank 1, in the middle, absent, with padding) |
| `bench/test_algos.py` | `build` on every `(algo, filter_kind)` torch cell: the `k` setter slices the top-k and changes no buffer; the filter submodule in `index_bytes`; `set_query_params`; build refusals; `official_config`'s `recheck` / `score_path` params and their refusals; `postfilter` equal (`torch.equal` on ids and scores) to a hand-computed matmul → topk(`alpha*k`) → dense-mask filter → first `k` reference at every `alpha` in `{1, 2, 4, 8}`, `k` within and past `N`, clause and bloom, the sentinel path exercised; its refusals |
| `bench/test_config.py` | job counts and keys per suite on `tests/bench/data/{mini,text,suites}.yaml`; `disabled`; build/query split; arms (filter kinds, sweeps, per-dataset overrides, empty slots); `ks_by_sweep`; gridded bloom widths, `compile`, `candidate_pool_frac`; each arm error named; narrows and `Job.narrowed`; the `-synth` YAMLs sharing their parent's inputs; `perf: false`; the interleave groups of the real suites, their parse errors, a job in two groups, the merged campaign children, `score_path` off official; G-grid (every real suite × dataset's job / cell counts and the 2026-10-08 rules on every cell); G-key (pinned resume keys of kept cells; the self_match datasets' filter cells pinned with `self_excluded`); every `config/*.yaml` × every suite through `load_matrix`, at exactly the dims the dataset and the suite share |
| `bench/test_inputs.py` | `load_inputs` on the conftest writer, `users_limit` once, prefix and row-count checks, the legacy layout loading equal to the modern one, misalignment raising, `attrs_digest`, `filters.query_attrs` (a `.pt` replacing `eval_split.parquet`: same full-split row check, same `users_limit` prefix), `sweep_qa`, filters by filter backend, `query_pool` |
| `bench/test_oracle.py` | padding, v4 fields and arithmetic, fingerprint in the file name (its row sample ending on the last row past 2^24 rows), an unreadable blob rebuilt, bloom FP rate, `code_version`; the item-chunked oracle and pass counts equal to a one-shot stable-sort reference at three chunk sizes, ties to the lower id; TF32 refused (G-oracle); `slice_wider` cutting a wider blob of the same inputs only, writing nothing |
| `bench/test_run.py` | end to end on the tiny fixture, both modes (`graph` = the CPU null entry): record schema, resume, `code_version` invalidation, `partial`, a failed cell + continue, a sticky CUDA error, the quality gate, the parity spill (and a same-backend spill rewritten), reachable-target masking, `modes` stamped per job (an eager-only uncapturable record `ok`, a capturable one `partial`), `postfilter`'s `recall_oracle` per `k` (equal to a pass at that `k`, never above the exact V1's); the quality cache (seed-free arms copy, V3 never, misses on other `ks` / `code_version`); a quality-only suite `ok`; G-ids (exact hash, and the canonical hash equal across a tied pair's two orders); G-dump (sidecar mean = record, upload lists it, aggregate skips it); G-interleave (same keys, one group per sweep and seed, `rounds`, whole-group resume); `frac_windows_below_max` against the device max, whichever cell ran first; `clocks_drift` against the first under-load sample (never the idle one) and `unstable` from drift or a perf entry's spread, each alone; a perf-phase OOM failing its whole interleaved group, not sticky, and resume re-running only that group; the spill's int32 ids exact past 2^24 and a `shape_mismatch`; `score_path` arms sharing the triton parity spill; [self-match](#self-match): an exact arm at oracle recall 1.0 with its own item excluded on both sides (key and record flags, the `none` cell untouched), and a finder of only the own item scoring 1.0 before the rule and 0 after |
| `bench/test_cli.py` | `bench run` via `CliRunner`, `--checkpoint` landing as the record's `inputs` (encoder faked), a real one-child `bench campaign` ending in `results.parquet`, a faked timed-out child, a restart mid-group keeping its parity spill (in-process children), zero cells → exit 1, a failed cell → `bench run` exit 1 and `bench campaign` exiting with its worst child's code, `bench report` over the campaign's own records, `bench env`'s JSON keys, the child log's start / end clock blocks and histogram, `bench oracle` prebuilding every sweep so `bench run` builds none |
| `bench/test_merge.py` | `bench merge` on two tiny trees: the accept rule per cell (an unstable original and its stable rerun, a stable original and its unstable rerun, an OOM `failed` line and its retry, a later record from another machine winning), the three tables' rows and provenance columns, the samples pairing (an orphaned sidecar gives no samples, a wrongly paired line raises), a tree without a sidecar, the manifest's counts, two trees sharing a name refused |
| `bench/test_upload.py` | `bench upload` with no network: `_logs/` and `_parity/` never listed, the manifest's sha256 per file, `n_records` counting only `<suite>/*.jsonl` (0 from a suite directory or a JSON-only tree), evidence beating `--gate` four ways, the generated README over several subtrees, `verify` catching a changed byte, the commit's operations (with the regenerated `results.parquet`) and `private=True`; `bench fetch` restoring a tree resume reads (its `MANIFEST.json` kept, never re-listed) and refusing to overwrite a different local copy |
| `bench/test_report.py` | `report.md` lists every cell with its recall and timings and marks partial / unstable ones, a `failed` record is in no line, citability off by default with the evidence beating `--gate` four ways and every reason stated, two `inputs` under one `(dataset, dim)` refused, an empty tree, a schema-1 record |
| `eval_datasets/test_layout.py` | the legacy pad-row rule, `apply_users_limit`, `validate_layout` clean on both layouts and flagging a short `eval_split`, a missing or swapped prefix sidecar, misaligned attrs |
| `eval_datasets/test_synth_filter.py` | G-synth: achieved pass rate within 1 % of target wherever `N·p ≥ 10^4` (N = 2 M), pass sets nested, same seed byte-identical, `query_attrs_synth` all ones at the full-split row count, the real attrs' bytes and `attrs_digest` untouched (modern and legacy layouts), the CLI resolving `data_dir` from the dataset YAML. G-corr (`--correlated`, on 10 tight blobs): items and queries both at their nearest centroid (brute force), the sidecar's mean pass rate equal to the attrs' and within 50 % of p (a random-init local optimum), over 80 % of a query's 10 nearest items passing, the real and uniform files untouched, rebuilds byte-identical, the CLI picking `content_dir` at `--dim` and refusing a dataset without one |
| `eval_datasets/test_pubmed.py` | the PubMed ETL loader on synthetic fixtures |
| `training/test_encode.py` | `training.evaluate`'s recall / ndcg equal `bench.metrics` to 1e-9; `encode_split`'s cache hit / stale key |

## How to run

The repo is a uv workspace; `uv run` from inside `evaluation/` finds it.
On a box that allows it, lock the clocks first (`sudo nvidia-smi -pm 1 &&
sudo nvidia-smi -lgc 1410`); on this A100 container that is denied, so
read latencies against `perf[].sm_mhz` and `env.clocks_drift`.

```bash
cd evaluation
# one cell set: goodreads d128, clause c0_genre, every algo and backend, on the gSASRec checkpoint
# (the config points at E1c)
uv run bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre \
    --seed 0 --checkpoint data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/best_model.pt
# one cell, eager only, no perf — the fastest iteration
uv run bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch --backend cuda \
    --filter-kind bloom --sweep c0_maincat --mode eager --skip-perf
# a campaign, one suite at a time
uv run bench campaign --suite filter --resume
uv run bench campaign --suite deep --resume
uv run bench upload --results results --path-in-repo <leg> --verify  # publish, then check the round trip
uv run bench fetch --path-in-repo <leg>                              # a published leg back into results/
```

Sanity checks after a run: `median_ms(bs=16) <
16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical
in `quality`; `memory_reserved_mib` flat (±5 %) across a group's cells;
`unstable` entries read against `perf[].sm_mhz`, since clocks are unlocked.

## Extending

**A new algorithm.** A library module first (the LiNR variants are
`retrieve.modules.linr`; a new composition is added there, because the
library retrieves and the harness measures):
the filter as `self.filter`, `prepare_queries(query_clause_attrs)` (the
filter's query encoding) and `forward(query, prepared=None) -> (ids,
scores)`, `k` forwarding to the final top-k layer, `set_query_params`
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
(for *results* see [Results storage](#results-storage)): `eval_repo(dataset)` names each
dataset's `<owner>/eval-<dataset>` HF dataset repo (`RETRIEVE_HUB_OWNER`; eval inputs +
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
(`tests/test_env_readers.py`); every ETL default goes through it. The GPU pod image sets it to
`/data`.

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
