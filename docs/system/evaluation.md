# `evaluation/` — the benchmark harness (`bench`)

Live reference for the retrieval benchmark that produces the thesis and
paper numbers: the measurement protocol, the module map, the config
matrix, the `bench` CLI, the JSONL record, resume and the campaign process
model. The plan it implements is
[evaluation-harness-v2.md](../plans/evaluation-harness-v2.md) (H); its
package shape is [evaluation-package-layout.md](../plans/evaluation-package-layout.md)
(V, roadmap C5) and what it may take from the library is
[library-harness-boundary.md](../plans/library-harness-boundary.md) (X).
**Nothing this harness has produced is citable until the roadmap's gate
for the step that produced it is green** (CLAUDE.md rule 2).

`evaluation/` is three packages with one dependency direction, `bench` →
`training` → `eval_datasets` (`tests/test_dependency_direction.py` enforces
it): `bench` measures algorithms (this document), `training` makes the
embeddings the sequential datasets need, `eval_datasets` owns what is on
disk. For the other two see [datasets.md](datasets.md) and
[checkpoints.md](checkpoints.md); for the algorithm internals
[kernels.md](kernels.md) and [architecture.md](architecture.md); for the
library's correctness suite, [testing.md](testing.md).

## Measurement protocol

Reproduced verbatim from H §2 (the amendment at the top of H applies: read
every mention of the two deleted SilverTorch backends as `official`, whose
`graph` mode is recorded as `null` with reason `not_capturable`). Where the code deviates, the
[Deviations](#deviations-from-h) section says so.

One **cell** = `(dataset, dim, filter_kind, sweep, algo, backend, params, seed)`. Per cell, in
one process, in this order. Everything below is `eager` unless labelled `graph`.

1. **Environment, once per process.** Seed torch/CUDA; TF32 off (one function); record
   provenance: GPU name, driver, CUDA runtime, torch/triton versions, git commit + dirty flag,
   hostname, UTC start, `nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.limit`
   sampled *after* warm-up. Clock locking is recommended, not enforced (`nvidia-smi -pm 1 &&
   nvidia-smi -lgc 1410` in the runbook); the harness warns when `clocks.sm` drifts >5 % from the
   first cell. **On the current A100 container locking is impossible** (`-lgc` is denied, no
   sudo; verified 2026-09-06): every number so far was taken at the unlocked application clock
   (1410 MHz under load, recorded per perf entry as `sm_mhz` and per cell as
   `env.sm_mhz_load`), and cross-run latency comparisons must check that field first — see
   the clocks note under [Deviations](#deviations-from-h). One 3-matmul GPU warm-up (keep
   `warm_gpu_once`).
2. **Inputs, once per (dataset, dim).** Item embeddings, queries, held-out targets, query attrs;
   `users_limit` applied in exactly one place (prefix, as today, so quality stays golden-comparable).
   Filter modules: one per *filter backend* (`triton` for triton and official cells, `torch` for torch
   cells). Per sweep: synthesised query attrs, skip mask, exact oracle top-`max(ks)` (fingerprint
   cache as today) **plus** `target_in_filter[u]` (held-out item passes the exact mask) and
   `pass_rate` (exact; bloom cells also record the bloom rate → `bloom_fp_rate`) — the axis both
   papers organise their tables around (LiNR high/low pass-rate; SilverTorch FP rate vs bits).
3. **Build.** `build_s` = wall time of algo construction + `register_index` with a sync on each
   side. `index_mib = Σ numel·itemsize` over `algo.buffers()` (the filter is a submodule, so mask
   algos include it, codesigned includes its attrs); `filter_mib` for the filter alone. No allocator
   deltas. Per-phase build time (k-means / quantize / assemble) stays a roadmap item; it needs a
   3-line `build_timings` dict in `SilverTorch.register_index` (main.py:132–143), not harness code.
4. **Quality, eager, once per cell at `k_max = max(ks)`.** Stream all kept users in chunks of 16
   (the OOM bound from passes.py:31–36 stands), accumulate `recall/ndcg/precision/mrr` at every `k`
   in `ks` from the one top-`k_max` list (exact for every algo here: same candidate set, same
   scores, `torch.topk` sorted). Targets: **oracle** on filter cells, **held-out** always (on filter
   cells restricted to users with `target_in_filter`, and `n_queries_heldout` recorded). Metrics
   accumulate as running sums on device; one `.item()` per metric at the end. Exact algos assert
   `recall_oracle@k ≥ 0.99` (fp16 tolerance per main.tex:706); a failure kills the run.
   **Amended by §8.2 K:** backends are no longer the innermost loop — each runs in its own
   process — so cross-backend parity is not free any more. Correctness lives where it always
   belonged, in the library's parity suite (**O** §5 T1–T7, roadmap B2, `torch.equal` on scores);
   the harness keeps only the *wiring* check that C4 and **O** WP-6 gate on, via a spill file:
   the first backend of a `(dataset, dim, algo, params, seed)` group writes its top-`k_max` ids to
   `results/_parity/<key>.npy` (≤ 40 MB, one alive at a time, deleted when the group closes) and
   later backends record `jaccard@k` and `score_max_abs_diff` against it. ~15 lines in `run.py`.
5. **Perf, per `(k, bs, mode)`** with `mode ∈ {eager, graph}` and `module.k = k` set before each
   variant (a WP-1 check confirms no layer bakes `k` into buffers at register time). Inputs: the
   fixed-seed pool of 4,096 query batches (and attr batches) rotated round-robin, identical across
   backends and modes. No L2 flush: the pool rotation makes index reads cold naturally while the
   structures serving keeps hot stay hot. `graph` = `torch.compile(mode="reduce-overhead",
   dynamic=False, fullgraph=True)` of a per-bs callable; after warm-up assert
   `counters["inductor"]["cudagraph_skips"] == 0` and one `cudaGraphLaunch` per call (as graphs.py
   130–144 does) — otherwise the cell is an error, not a mislabelled number. Protocol per variant:
   50 warm-up calls → sync → **3 windows** of `N = clamp(2 s / median_est, 1000, 5000)` calls, each
   call bracketed by CUDA events on the current stream, wall clock around the window with one
   sync at the end. From the window with the median median: `median_ms, mean_ms, p95_ms, p99_ms,
   min_ms, n`, `qps = N·bs / wall_s` (closed-loop single client — the in-process analogue of the
   papers' client-side QPS), `host_gap_ms = wall/N − mean_gpu_ms` (wp4 `wall_vs_gpu.txt`'s number,
   diagnostic), and `spread = (max − min) / median` of the three window medians; `spread > 0.05`
   sets `unstable: true`. The per-call vector of the chosen window goes to the samples parquet.
   `peak_fwd_mib` = `max_memory_allocated − allocated_before` over the first eager window only
   (graph mode allocates nothing). The first eager call runs under
   `torch.cuda.set_sync_debug_mode("warn")` to catch hidden host syncs. `--profile` (optional,
   off by default) wraps one eager call in `torch.profiler` and stores per-kernel CUDA µs
   (top 8 kernels) — the wp4 `kernel_only.py` split, 25 lines.
6. **Seeds and repeats.** Timing repeats are the 3 windows above (no rebuild). Seeds change the
   IVF (k-means), the OPORP projection (`v3_seed`) and the pool; v1/v2/v4 are seed-invariant in
   quality, so `seeds: [0, 1, 2]` applies only to the `deep` suite and the headline `filter` cells
   (d128 `c0_*`, `all4`); report median across seeds with min–max whiskers; paired tests offline.
7. **Comparability, stated once in the paper.** Same as SilverTorch: A100, D=128, INT8 IVF,
   n_probe grid incl. **24** (their production setting; add it to the deep grid), bloom bits/FP-rate,
   mean latency at bs=16 vs recall (their Fig. 5), QPS and P99 from the same vector. Same as LiNR:
   V1–V4 by batch size (1, 16), mean + p95, label recall on filtered sets, high/low pass-rate
   bucketing by `pass_rate`. Necessary differences: in-process (no RPC, no user tower, no
   5,000-request replay), 0.8–5.4 M items vs 10–80 M / 15.5 M, 80 GB card, k ≤ 1000 vs 1024/2000,
   closed-loop QPS rather than open-loop load, no live updates. `eager` is the comparable number;
   `graph` is the deployed-best-case number and is reported alongside, never instead.
8. **Cell cost.** Quality once per cell (not per k), perf 3 k × 3 bs × 2 modes × 3 windows ≈ 2 min
   per cell on A100; the campaign in §6 WP-5 is ~700 cells ≈ 24 h (today: 3 builds per cell).
   Two amendments pull in opposite directions and roughly cancel: §8.2 A (build params sweep
   separately from query params) removes ~5 of every 6 k-means builds in the `deep` suite, while
   §8.2 K (a process per backend, ~200–250 boundaries instead of ~80) adds a dataset reload and a
   CUDA context init per boundary — *est.* 30–60 s each, 2–4 h over the campaign. WP-5 records
   both so the next plan does not have to guess.

H §8.2 adds to the record: `schema_version`, `status`, `code_version` in
the resume key, `git_branch` / `python` in `env`, IQR and
two outlier counts plus `load: "closed_loop"` per perf entry, build-time
vs query-time parameters (one build, many query configs), `disabled: true`
sweeps, the oracle fingerprint in the blob's file name, and the process
boundary at `(dataset, dim, algo, backend)`.

### Deviations from H

- **Parity spill is `.npz` (ids as int32 + scores as float32), not `.npy`**,
  so `score_max_abs_diff` can be computed; the hash in the file name is over
  the key block minus `backend` (which includes `suite`, `filter_kind` and
  `sweep` — the ids differ per sweep). `bench run` never deletes the
  directory (the next backend's run needs it); `bench campaign` does, when
  the `(dataset, dim, algo)` group closes. There is no "missing reference"
  state: whichever backend runs a cell first writes the spill file
  (`parity: "reference"`), so with `--resume` after the triton cells are
  done, `torch` becomes the reference and `official` records `vs_torch`.
- **Samples go to a JSONL sidecar** (`<name>.samples.jsonl`, one line per
  perf entry with the key block, `k`, `bs`, `mode`, `ms: [...]`), not a
  parquet: parquet cannot be appended per cell. `bench report` reads the
  sidecar directly for the latency violins.
- **Resume re-runs `failed` and `partial` records**; only `status: ok` at the
  same `code_version` counts as done. `report.py` must take the last record
  per key.
- **The campaign child is `python -m bench.cli run …` on the same
  interpreter**, not `uv run bench run …` (no second `uv` resolution per
  child; `uv run bench campaign` already resolved the environment).
- **Exact algos** for the §2.4 gate are `linr_v1_filter_mask` and `linr_v2`
  (`linr_v4` is int8 and not exact); the gate records the cell as `failed`
  before raising `QualityGateError`.
- `--flush-l2` is not implemented (H mentions it as a C4 throwaway).
- `build_s` is recorded on every cell of a build (a `deep` job with six
  `n_probe` values repeats the same `build_s` six times).

Two things to read the numbers with, stated rather than changed:

- **Per-call `*_ms` is closed-loop latency, not kernel time.** The CUDA
  events bracket the GPU timeline between the two `record`s; when the
  GPU idles waiting for the host (eager, `bs = 1`) the interval includes
  launch latency and `host_gap_ms ≈ 0`. That is the latency the papers
  report; `--profile` (`kernels`) is the kernel-time view.
- **Clocks are compared load against load, never against idle** (C4's
  finding L4-c, fixed in C5). `env.sm_mhz_idle` is the process-start
  sample — provenance, with `mem_mhz`, `sm_max_mhz`, `power_limit_w` — and
  is compared with nothing. Every perf entry's `sm_mhz` is sampled right
  after its last timing window's sync, under load; `env.sm_mhz_load` is
  the median of a cell's under-load samples, and `env.clocks_drift` fires
  when any of them is more than 5 % (`run.CLOCK_DRIFT`) from the process's
  *first* under-load sample. The pre-C5 harness compared the idle
  start-of-process sample against under-load ones, so it flagged the GPU
  *boosting* (10 of C4's 18 `unstable` cells), and recorded
  `clocks_locked` as "within 2 % of `--expected-sm-mhz`", which read `true`
  on a box that cannot lock clocks; both fields are gone (`schema_version`
  2). Read an unlocked-clock run's latencies against `perf[].sm_mhz` (H
  §7's fallback); C4's [c4_gate.py](../plans/evaluation-harness-v2-artifacts/c4_gate.py)
  still reads the schema-1 `env.sm_mhz`.

## Architecture

Ten modules under [`evaluation/bench/`](../../evaluation/bench/), plus
the two the harness takes from its sibling packages
(`eval_datasets.layout` for the on-disk readers, `training.encode` for the
SASRec encode). No base class, no context bags, no stats dataclasses, no
results-IO layer (H §4); two things are classes on purpose — `Job`, the
resolved build spec whose `key(params)` *is* the record's key block, and
the library's algo `nn.Module`s, which exist for `buffers()`, `.k`,
`torch.compile` and because their `forward` is the readable spec of each
cascade.

| module | owns |
|---|---|
| [`measure.py`](../../evaluation/bench/measure.py) | `setup`, `warm_gpu_once`, `provenance` (GPU, driver, CUDA, torch, triton, commit, `dirty` = `subtree_dirty()` over `retrieve/src/retrieve`, `repo_dirty`, branch, `code_version` = the subtree's tree hash, or `files:<sha256>` of the sources on disk when the subtree is dirty, host, python, started), `clocks()` (one `nvidia-smi` sample), `timed_build`, `index_bytes` (Σ buffers, submodules included, deduplicated), `stats`, `latency(fn, bs=, mode=)` (§2.5 windows, IQR + outlier counts, `load: closed_loop`, `peak_fwd_mib`, the under-load `sm_mhz`), `graph_callable` (raises `NotCapturable` with the record's `reason`), `profile_once` |
| [`records.py`](../../evaluation/bench/records.py) | what a record *is*: `SCHEMA_VERSION`, `KEY_FIELDS`, `resume_key`, `record_path`, `samples_path`, `append_record` (one `write` + `fsync`), `read_records` / `read_keys` (one torn trailing line tolerated), `flatten(results_dir) → flat.csv` (one row per perf entry, last record per key — what `report.py` reads) |
| [`metrics.py`](../../evaluation/bench/metrics.py) | `accumulator(ks, device)` / `accumulate(acc, ids, targets, num_targets=None, ranked=False)` / `finalize(acc)` — recall, ndcg, precision, mrr at every `k` from one top-`k_max` list as float64 running sums on device; `ranked=True` scores against the oracle's own top-`k` prefix; `per_row`, `jaccard_at_k`. `training/evaluate.py` keeps its own frozen copy (V D5), pinned to agree |
| [`algos.py`](../../evaluation/bench/algos.py) | the algorithm table (X §3): `ALGOS` name → library class (`LiNRV1`–`LiNRV4`, `SilverTorch`), `FILTER_KINDS`, `BACKENDS`, `FILTER_MODE` (`clause` → `exact`), `PATHS` **derived from `retrieve.interfaces.DISPATCH`**, `filter_backend` (`official` → `triton`), `build(algo, item_embs, k=, backend=, …)` (construct + `register_index`, ≈ 25 lines), `build_filter`, `is_valid_combo` |
| [`config.py`](../../evaluation/bench/config.py) | `Dataset`, `Job`, `load_dataset`, `load_matrix` — the config matrix below |
| [`inputs.py`](../../evaluation/bench/inputs.py) | `load_inputs` (dispatch to `training.encode.encode_split` or the `eval_datasets.layout` text readers; `users_limit` once, as a prefix), `sweep_qa`, `build_filters` (keyed by filter backend), `exact_filter`, `query_pool` |
| [`oracle.py`](../../evaluation/bench/oracle.py) | the exact filtered oracle as blob v4, `attrs_digest`, `pass_counts`, `pass_rate`, `bloom_fp_rate` |
| [`run.py`](../../evaluation/bench/run.py) | `run(jobs, out_dir=...)` — the cell loop; `MODES`, `QUALITY_CHUNK = 16`, `EXACT_ALGOS`, `PERF_STAT_KEYS`, `CLOCK_DRIFT` |
| [`cli.py`](../../evaluation/bench/cli.py) | `bench run` / `campaign` / `check` / `upload` / `report` |
| [`report.py`](../../evaluation/bench/report.py) | `bench report`: `flat.csv`, the thesis and paper tables as LaTeX, the figures, the methodology paragraph and `report.md`; the `ARTIFACTS` dispatch table, the citability verdict and the `PAPER_REPORTED` constants. See [Report](#report-reportpy) |
| [`upload.py`](../../evaluation/bench/upload.py) | `bench upload`: publish a results tree to the HF results repo with a `MANIFEST.json` (provenance + a sha256 per file) and a generated README. See [Results storage](#results-storage) |

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
`None` because its candidate source is the filter.
`tests/bench/test_paths.py` pins the derivation.

| algo | `none` | `clause` / `bloom` | `official` |
|---|---|---|---|
| `linr_v1_filter_mask` (`PostfilterKNN`, fp16 cuBLAS + mask) | `cublas` (triton and torch collapse) | `cublas+triton` / `cublas+torch` (the filter's kernel) | — |
| `linr_v2` (`PrefilterKNN` over the filter's candidate list) | — (the candidate source is the filter) | `triton` / `torch` | — |
| `linr_v3` (`OneBitKNN` top-`candidate_pool` → `PrefilterKNN`) | `triton` / `torch` | `triton` / `torch` | — |
| `linr_v4` (`PostfilterKNNInt8`, `_int_mm` + mask) | `cublas` | `cublas+triton` / `cublas+torch` | — |
| `silvertorch` (IVF + INT8, predicate fused: `filter_mode` none / exact / bloom) | `triton` / `torch` | `triton` / `torch` | `official` |

`official` is Meta's reference backend (roadmap B1) and exists for
`silvertorch` only; its standalone filter modules are Triton
(`algos.filter_backend`), and it is eager-only (`SilverTorch.capturable`
is `False` there), so its `graph` perf entries are `null` with `reason:
not_capturable`.

Every algo module is the library's (X §4): `forward(query,
query_clause_attrs=None) -> (ids [B, k], scores [B, k])`, `torch.topk`-sorted
rows, `-1` ids where a row has fewer than `k` survivors, `k` settable after
`register_index`, `set_query_params` (`n_probe` on `SilverTorch`,
`candidate_pool` on `LiNRV3`) re-validating without a rebuild, `capturable`
a class attribute, the filter a submodule (`self.filter` on the LiNR
variants; `SilverTorch` fuses the predicate and carries the attribute
buffers inside `index_mib`). `build(algo, item_embs, k=, backend=,
filter_kind=, filter_mod=, item_attrs=, clause_is_reverse=, params=,
seed=)` is the one factory; it refuses `None`-path cells and `n_probe >
n_lists`. `params` are the merged build + query params over
`algos.SILVERTORCH_DEFAULTS` (`n_lists 1024, n_probe 24, n_iter 10`); on
`silvertorch` bloom cells `run.py` merges the suite's `bloom` defaults
(`m_bits`, `k_hash`) in as well. The old harness wrapper's `k`-setter
probe-pool check is gone: the library validates in `set_query_params` and
at forward.

## Config: one YAML per dataset + `suites.yaml`

Seven files under [`evaluation/config/`](../../evaluation/config/):
[`goodreads.yaml`](../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../evaluation/config/arxiv.yaml),
[`yambda-500m.yaml`](../../evaluation/config/yambda-500m.yaml),
[`yambda-5b.yaml`](../../evaluation/config/yambda-5b.yaml),
[`yfcc10m.yaml`](../../evaluation/config/yfcc10m.yaml) (in the `filter`
suite since 2026-09-16), [`pubmed.yaml`](../../evaluation/config/pubmed.yaml)
(in no suite yet) and
[`suites.yaml`](../../evaluation/config/suites.yaml). Their values are
the old configs' (`users_limit: 10000`, the same sweeps, ks, batch sizes,
`n_probe` grid) so C4 can compare against A1's golden cells;
`tests/test_config.py` asserts the d128 `filter` / `quality` cell sets
equal the deleted `config/<dataset>/d128-*.yaml` ones.

```yaml
# config/<dataset>.yaml — one per dataset; every string may carry {dim}
data_dir: data/goodreads-work-id
checkpoint: data/goodreads-work-id/checkpoints/gsasrec-d{dim}-drop0.5-id/best_model.pt
#   or, for pre-encoded text datasets, a per-dim mapping instead of `checkpoint`:
#   content_dir: {64: content_d64, 128: content_d128, 256: content}   # relative to data_dir
dims: [64, 128, 256]
encode: {batch_size: 512, num_workers: 8, max_seq_length: 200}       # SASRec datasets only
users_limit: 10000                                                    # or null
filters:                                                              # optional
  attrs: item_attrs_narrow.pt                                         # relative to data_dir
  reverse: clause_is_reverse_narrow.pt
  clause: {c0_genre: [0], c1_lang_reverse: [1], all4: [0, 1, 2, 3]}   # name: active clauses
  bloom: {c0_genre: [0], old_sweep: {clauses: [2], disabled: true}}   # long form: disabled
```

`gt_dir` is derived (`<data_dir>/gt_d{dim}`); `output`, `split`, `device`,
`gt_subdir`, `query_emb_path` and `content_subdir` are gone. Unknown keys
raise `ConfigError` naming the file.

```yaml
# config/suites.yaml — a suite = cells run on every listed dataset × its dims
filter:
  datasets: [goodreads, arxiv]
  dims: [128]                       # optional; default: the dataset's dims
  filter_kinds: [clause, bloom]     # none | clause | bloom
  ks: [100, 500, 1000]
  batch_sizes: [1, 8, 16]
  algos: {linr_v2: [triton, torch], silvertorch: [triton, torch, official]}
  params:                           # per algo; dict-of-lists = grid, list-of-dicts = combos
    silvertorch: {build: {n_lists: [1664, 8192]}, query: {n_probe: [4, 8, 24, 32]}}
  seeds: {default: [0], headline: {sweeps: [c0_genre, all4], dims: [128], seeds: [0, 1, 2]}}
  bloom: {m_bits: 1024, k_hash: 5}  # optional per-suite override of the top-level default
bloom: {m_bits: 1024, k_hash: 5}
```

`build:` params rebuild the index; `query:` params (`n_probe`,
`candidate_pool` — the `QUERY_PARAMS` set) are applied with
`set_query_params` to the built index, so the `deep` suite is two
k-means per `(dataset, sweep, seed)`, not twelve. Putting a query param
under `build:` (or vice versa) is a `ConfigError`. `seeds:` is a list, or
the `default` / `headline` form (headline seeds apply to the named sweeps
at the named dims, whatever the filter kind). CLI narrows (`dims`,
`algos`, `backends`, `filter_kinds`, `sweeps`, `seeds`, `ks`,
`batch_sizes`) filter the suite's lists *before* the `PATHS` collapse;
`ks` / `batch_sizes` are replacements, not selections, and set
`Job.narrowed` when they differ from the suite's (`run` then records
`partial`).

`load_matrix(dataset_yaml, suites_yaml, suite, **narrows)` returns `Job`s
grouped by `Job.group == (dataset, dim, algo, backend)` — the campaign's
process boundary — in the order `dim → algo → backend → filter_kind →
sweep → build → seed`. A `Job` is one build: `dataset, dim, suite,
filter_kind, sweep, clauses, algo, backend, path, build, query, ks,
batch_sizes, seed, bloom, data: Dataset, narrowed`; `job.cells()` lists the
`params = build | query` of each cell and `job.key(params)` is the
record's key block. `none` cells have `sweep == "full_scan"` and
`clauses is None`.

## CLI

```
bench run      --dataset D --suite S [--dim N]* [--algo A]* [--backend B]* [--filter-kind K]*
               [--sweep W]* [--k N]* [--bs N]* [--seed N]* [--mode eager|graph]*
               [--skip-quality] [--skip-perf] [--profile] [--out results] [--output FILE]
               [--resume|--force] [--config-dir config]
bench campaign --suite quality|filter|deep|all [--dataset D]* [--dim N]* [--mode M]*
               [--skip-quality] [--skip-perf] [--profile] [--out results] [--resume|--force]
               [--config-dir config] [--timeout 6.0]
bench check    --dataset D [--dim N]* [--config-dir config]   # eval_datasets.layout.validate_layout
bench upload   [--repo-id user/repo] [--results DIR] [--path-in-repo PREFIX] [--gate STEP]
               [--private|--public] [--verify] [--dry-run]
bench report   [results] [--out DIR] [--gate STEP] [--only NAME]* [--dim 128] [--k 100]
               [--bs 1] [--compare-bs 16] [--mode eager|graph] [--backend triton]
               [--sweep W] [--batch-dataset D] [--budget-ms MS]*
```

`*` = repeatable. `bench run` expands one `(dataset, suite)` through
`load_matrix` (every repeatable option is a narrow) and runs the cells in
*this* process; `--resume` (the default) skips cells already `ok` at the
current `code_version`, `--force` re-runs them (their old records stay in
the file). `--mode` defaults to both; `--output` overrides the per-`(suite,
dataset, dim)` file. `--k`, `--bs` and `--mode` *replace* the suite's
lists rather than select cells, so a run with any of them (or a
`--skip-*` flag) writes `status: partial` records — resume re-runs them,
and the next full run is never fooled by an iteration-day cell. The exit
code is 1 when any cell failed, and 1 with a message when the narrows
select zero cells (a `--sweep` typo is an error, not an empty success).

`bench campaign` is the loop of H §3.4 / §8.2 K: for every suite (in the
order quality, filter, deep for `all`), every listed dataset and every
`(dataset, dim, algo, backend)` group of `load_matrix`, one child process
`python -m bench.cli run --dataset … --dim … --suite … --algo …
--backend … --resume` (same interpreter, `cwd = evaluation/`), sequential.
The child's stdout + stderr go to
`results/_logs/<suite>_<dataset>-d<dim>_<algo>_<backend>.log` (appended, the
command line first); one summary line per child (`time suite dataset dim
algo backend rc seconds log`) goes to `results/_logs/campaign.log` and the
terminal; a non-zero rc is recorded and the loop continues; a child
still running after `--timeout` hours (default 6) is killed and recorded
as `rc=timeout` (exit code 124, noted in its log); the exit code
is the worst child rc, or 1 when a listed dataset expands to no groups or
no child was launched at all (`--dataset` / `--dim` selecting nothing).
`results/_parity/` is deleted when the `(dataset, dim, algo)` group
closes. A backend is the thing under test, so it gets
the process: no dynamo cache, allocator arena or CUDA-graph pool outlives
it, at the cost of a dataset reload and a CUDA context init per group.
`bench check --dataset D` runs `eval_datasets.layout.validate_layout` on
the dataset's directory at every dim (files, prefix sidecars, the row
alignments the readers enforce) and exits 1 on any problem — run it before
a campaign on a freshly staged dataset.

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
   skip-masked), `oracle_rows` (kept with ≥ 1 survivor — the old harness's
   zero-target skip), `heldout_rows` (kept, ≥ 1 target, and on filter cells
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
   by CPU masks + `index_select`, so no per-chunk sync); the parity spill;
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
`os.replace`), so a crash mid-save leaves the old file or nothing, never
a torn one, and an unreadable file at the oracle's fingerprint path is
rebuilt with a warning. Each JSONL line is one `write` + `fsync`; the
samples line is appended *before* its record, so a crash between the two
cannot leave a resumable record without its vector; `read_keys` ignores
(and logs) one torn trailing line — the cell in flight when the process
died — and raises on a malformed line anywhere else.

## Output: one JSONL record per cell

`results/<suite>/<dataset>-d<dim>.jsonl`, appended by the process the
moment a cell finishes (`json.dumps` of one line, `allow_nan=False` — NaN
and ±inf become `null` — then `write` + `fsync`). Nested, not wide:
`polars.read_ndjson(path).explode("perf").unnest("perf")` flattens the
perf entries.

| field | type | value |
|---|---|---|
| `schema_version` | int | `2` (C5: the clock fields below; 1 = the C4 records) |
| `status` | str | `ok`, `partial` (the record does not carry everything the suite asked for), `failed` |
| `partial_reasons` | list / null | why `partial`: any of `skip_quality`, `skip_perf`, `modes` (a `--mode` subset), `ks_bs` (`--k` / `--bs` replaced the suite's lists — `Job.narrowed`) |
| `dataset`, `dim`, `suite`, `filter_kind`, `sweep`, `algo`, `backend`, `params`, `seed` | | the key block = `Job.key(params)` (`KEY_FIELDS`); `params` is the native dict of build + query params (`{}` when the algo takes none) |
| `path` | str | `PATHS[(algo, filter_kind, backend)]` |
| `n_items`, `n_queries` | int | catalogue size, queries after `users_limit` |
| `n_kept` | int | queries not skip-masked (every query on `none` cells) |
| `n_queries_oracle` | int / null | kept queries with ≥ 1 survivor (the oracle metrics' `n`); `null` on `none` cells |
| `n_queries_heldout` | int | queries the held-out metrics cover (kept, ≥ 1 target, `target_in_filter` on filter cells) |
| `n_targets_in_filter` | int | held-out targets those queries are scored against: every valid target on `none` cells, only the ones the exact mask admits on filter cells |
| `pass_rate` | float | exact mask pass rate over kept queries (`1.0` on `none`) |
| `bloom_fp_rate` | float / null | mean per-query `(bloom − exact) / (N − exact)`; bloom cells only |
| `bloom` | dict / null | `{m_bits, k_hash}` on bloom cells |
| `k_max`, `ks`, `batch_sizes` | | the suite's, `k_max = max(ks)` |
| `build_s` | float | construction + `register_index`, sync on each side (same value on every cell of one build) |
| `index_mib` | float | Σ buffers of the algo module, filter submodule included |
| `filter_mib` | float | Σ buffers of the filter submodule alone (`0.0` without one; `silvertorch` carries its attrs inside `index_mib`) |
| `quality` | dict / null | `heldout: {recall@k, ndcg@k, precision@k, mrr@k for k in ks, n}`; on filter cells also `oracle: {…}` (ranked-prefix targets); `jaccard_vs_first@k` per `k`, `score_max_abs_diff`, `parity` (`reference` = this record wrote the spill file, `vs_<backend>` = compared against it, `shape_mismatch:…`); `null` with `--skip-quality` |
| `perf` | list / null | one entry per `(bs, k, mode)` (table below); `null` with `--skip-perf` |
| `unstable` | bool | any perf entry `unstable` (window spread > 5 %), or `clocks_drift` |
| `memory_reserved_mib` | float / null | `torch.cuda.memory_reserved()` after the cell — the leak detector across a group's cells |
| `elapsed_s` | float | wall time of the cell |
| `env` | dict | `gpu, driver, cuda, torch, triton, commit, dirty, repo_dirty, git_branch, code_version, host, python, started, config_sha`; the process-start sample `sm_mhz_idle, mem_mhz, sm_max_mhz, power_limit_w`; the cell's `sm_mhz_load` (median of its perf entries' under-load samples; `null` without perf or CUDA) and `clocks_drift` (any under-load sample > 5 % from the process's first) |
| `stage`, `error` | str | `failed` records only: where it died and the traceback |

Perf entry:

| key | value |
|---|---|
| `k`, `bs`, `mode` | the variant; `mode ∈ {eager, graph}` |
| `n`, `median_ms`, `mean_ms`, `p95_ms`, `p99_ms`, `min_ms`, `iqr_ms` | of the chosen window (median of the three window medians); quantiles linear-interpolated |
| `qps` | `n · bs / wall_s` of that window (closed-loop, one client) |
| `host_gap_ms` | `wall / n − mean_ms` |
| `outliers_std`, `outliers_tukey` | counts beyond 3 σ / the 1.5 IQR fences, never dropped |
| `spread`, `unstable` | `(max − min) / median` of the three window medians; `> 0.05` |
| `window_medians_ms` | the three medians |
| `peak_fwd_mib` | eager only, first window: `max_memory_allocated − allocated_before` |
| `sm_mhz` | the SM clock sampled right after the last window's sync, with the GPU still at its load clock — the per-variant value H §7's unlocked-clock fallback compares, and the only clock `clocks_drift` looks at; `null` without CUDA |
| `cache_plans` | on every entry: `false` on `silvertorch`/`official` (`run.perf` replaces `module.official` with `cache_plans=False` before the first variant, so every timed forward pays the CPU expression parse), `null` on backends without a plan cache |
| `load` | `"closed_loop"` |
| `kernels` | `--profile`, eager only: top-8 CUDA kernels `{kernel, us, calls}` |
| `reason` | present when the variant could not run (`not_capturable`, `cuda_unavailable`, `cudagraph_skips=N`, `cudaGraphLaunch per call = N, expected 1`); every stat key is then `null` |

`results/<suite>/<dataset>-d<dim>.samples.jsonl` holds the per-call vector
of the chosen window: one line per perf entry, `{key block, k, bs, mode,
ms: [...]}`, written before the cell's record.

### Resume

The resume key is `records.resume_key(job.key(params), code_version)` —
canonical JSON of the key block plus `measure.code_version()`: the library
subtree's tree hash (`git rev-parse HEAD:retrieve/src/retrieve`) when the
subtree is clean, else `files:<sha256>` over the `retrieve/**/*.py` sources
actually on disk (also the value outside a git checkout; the two namespaces
are disjoint). A kernel edit — committed or not — therefore invalidates
every cell; a doc or plan edit invalidates none. `records.read_keys(path)`
rebuilds the key from a record as `resume_key({k: rec[k] for k in
KEY_FIELDS}, rec["env"]["code_version"])` and keeps the last status per
key; a cell is skipped when that status is `ok`. `--force` runs everything
and appends. `records.flatten(results_dir)` writes `flat.csv` — the last
record per key of every `<suite>/*.jsonl`, one row per perf entry, the key
block + record scalars + `env_*` + `heldout_*` / `oracle_*` / `quality_*` +
`perf_*` columns — the one table `report.py` (D4) reads.

Two dirty flags, both `null` outside a git checkout: `env.dirty` is
`git status --porcelain -- retrieve/src/retrieve` (untracked files
included — a new kernel module is measured code too) and is the flag
`report.py` refuses to cite (H §8.2 F); `env.repo_dirty` is the tracked
files anywhere else (`--untracked-files=no`, informational: a docs or
harness edit). `evaluation/results/` is excluded from `repo_dirty`: the
harness's outputs are data — committed and mirrored to HF by
`bench upload` (H §3.2, §8.2 G/I) — so a campaign appending to a
committed JSONL is not a dirty tree.

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
`flat.csv` first (H §8.2 G — `records.flatten`, and the artifact to ship
with the paper), then one file per artifact, then `report.md`. Every table
and figure is built from `flat.csv`; `records.read_records` is read a
second time for the provenance block alone, because `schema_version`,
`partial_reasons`, `stage` and `error` are not columns of `flat.csv`.

| artifact (`--only` name) | file | what it is |
|---|---|---|
| `recall_nofilter` | `tables/tab-recall_nofilter.tex` | held-out Recall@k on `filter_kind: none` cells, datasets × algos (`tab:recall_nofilter`) |
| `pareto` | `tables/tab-pareto_<dataset>.tex` | condition × algo: oracle recall, `median_ms`, speedup vs LiNR V1, `index_mib` (`tab:pareto_<dataset>`) |
| `batch_scaling` | `tables/tab-batch_scaling.tex` | amortised ms/query per algo × batch size, each cell carrying its window spread (`tab:batch_scaling`) |
| `memory` | `tables/tab-memory.tex` | `index_mib`, datasets × algos (`tab:memory`) |
| `parity` | `tables/tab-backend_parity.tex` | per `(dataset, sweep, algo, backend)`: `path`, `jaccard_vs_first@k`, `score_max_abs_diff`, eager vs graph median and their ratio |
| `recall_at_budget` | `tables/tab-recall_at_budget.tex` | H §8.2 H: best recall reachable under each `--budget-ms` p99 budget, with the algo that reached it |
| `paper_comparison` | `tables/tab-paper_comparison.tex` | our `--compare-bs` eager mean / p99 / QPS / pass rate beside the numbers SilverTorch and LiNR report, with H §2.7's differences as footnotes. The published rows are the `PAPER_REPORTED` constant, cited per row |
| `fig_pareto`, `fig_qps_recall`, `fig_batch_scaling` | `figures/*.png` | recall vs latency, QPS vs recall, amortised latency vs batch (error bars = window spread) |
| `fig_deep_sweep` | `figures/fig-deep-sweep-*.png` | one per swept parameter: recall and latency against its value, whiskers = min–max across seeds |
| `fig_latency_violin` | `figures/fig-latency-violin.png` | per-call distributions from the samples sidecar (`<name>.samples.jsonl`, one torn trailing line tolerated) |
| `methodology` | `methodology.tex` | the thesis's §"Методология замеров" itemize, its constants read live out of `measure.latency`, `inputs.query_pool` and `run` so text and code cannot drift |
| — | `report.md` | provenance, citability verdict, the selection used, a coverage table, the failed cells with their stage and error, the partial records, the unstable variants with their spread, the artifact list |

**Citability (CLAUDE.md rule 2).** Nothing is citable by default. `--gate
STEP` declares that step's roadmap gate green, and only then can an
artifact come out unmarked — but the evidence vetoes the flag: a `failed`
or `partial` record, an `env.dirty` one, or one whose `env.git_branch` is
not `staging` / `development` / `main` (rule 2's own wording: "harness numbers from a
branch are not paper material") keeps the marker on. Otherwise
every `.tex` carries a `% PROVENANCE: *** NOT CITABLE ***` banner with the
reasons, its caption starts with `\textbf{[PRE-CAMPAIGN RECORDS — NOT
CITABLE]}`, and every figure gets a diagonal watermark. Every artifact
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
never compared with an under-load one — the artifact that made 92 of 99 of
C4's latency rows appear to fail (H §12.4).

**Selection.** One set of options narrows every artifact: `--dim`, `--k`,
`--bs` (`--compare-bs` for the paper table), `--mode`, `--backend`,
`--sweep`, `--budget-ms`. Where a table's shape allows only one value per
cell and the records hold several parameter sets, the smallest by
canonical-JSON order is shown and a caption footnote names all of them;
several seeds are reduced to their median (min–max whiskers in the
figures). `tab:batch_scaling` needs one dataset: `--batch-dataset`, else
the best-covered one, named in the caption. A selection that matches
nothing emits the table with a `--- no matching cells ---` row rather than
failing.

**Not compiled.** No TeX toolchain is installed on this box, so the
fragments are checked structurally (`tests/bench/test_report.py`:
balanced environments, balanced braces and math, the thesis's labels
present, no unescaped `_` outside math, `\texttt` and `\label`), never
compiled.

## Results storage

Three destinations, decided by size and by how expensive the file is to
recreate. The records are the product of this project and the only thing the
paper may cite (CLAUDE.md rule 2), and they currently live on a rented box whose
`/workspace` volume is a ~26 GB quota — so where each output goes is a rule, not
a habit.

| what | where | why |
|---|---|---|
| `results/<suite>/<dataset>-d<dim>.jsonl` (the records), `flat.csv`, `report/` (`*.tex`, `report.md`), the validation record in the plan | **git**, under `docs/plans/<plan>-artifacts/` while a step is in flight and `evaluation/results/` for the campaign | kilobytes to a few MB, line-diffable, and they are the evidence. 79 records so far are 750 KB in total |
| `*.samples.jsonl` (the per-call latency vectors), `.perkernel/` profiles, figures | **HF Hub**, `pinkmeme/eval-results`, private | ~60× the records: 45 MB of sidecars behind those same 750 KB of records, and nobody reads a diff of them. Regenerable only by re-running the cell on the GPU |
| `results/_parity/*.npz` (600–680 MB per run), `results/_logs/` | **nowhere** — deleted | rewritten by every run, and the parity *verdict* (`jaccard_vs_first@k`, `score_max_abs_diff`) is already inside the record. C4, C5 and B3 each deleted theirs; `.gitignore` covers both directories and `bench upload` skips every `_`-prefixed path part |

Both halves stay in step: the Hub copy of a subtree carries the sha256 of every
file it holds, and that manifest is committed next to the plan
([results-storage/](../plans/evaluation-harness-v2-artifacts/results-storage/)),
so git can prove what the Hub has without downloading it.

### `bench upload`

One invocation publishes one `--path-in-repo` subtree of one results tree:

```bash
uv run bench upload --results results --path-in-repo d1-a --verify
uv run bench upload --results ../docs/plans/official-silvertorch-artifacts/b3/e2e \
    --path-in-repo b3 --dry-run
```

`--repo-id` defaults to `upload.RESULTS_REPO` = `pinkmeme/eval-results` — the
registry entry for *results*, beside `eval_datasets.hub.EVAL_REPOS` for
*datasets*. The listing is `upload.files`: the tree minus anything under a
`_`-prefixed path part and minus the two files a previous upload generated.
Everything goes up in **one commit** (`create_commit`, so a failure leaves no
half-published subtree), together with:

- **`<prefix>/MANIFEST.json`** — `report.provenance` over the records being
  uploaded (`code_version`, `commit`, `git_branch`, `schema_version`, `gpu`,
  `host`, the run window, the status counts, the `unstable` count, the gate and
  the citability verdict with its reasons) plus `{path, bytes, sha256}` per
  file. It is the same function `bench report` calls, so the Hub copy cannot
  claim more than the tables would.
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

`tests/bench/test_upload.py` pins all of it with no network: the scratch
exclusion, the checksums, the four ways evidence beats `--gate`, the generated
README, the round-trip check catching a changed byte, and the commit's operation
list with `private=True`.

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
`check_items_aligned`), `load_query_attrs` checks `eval_split.parquet`'s
row count against the *full* split, and `apply_users_limit` is the one
`users_limit` site: a prefix over queries, targets, `n_targets` and `qa`
together (kept a prefix, not a sample; H §7). `sweep_qa(qa, clauses)`
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

| file | checks |
|---|---|
| `test_dependency_direction.py` | every `import` / `from` in `bench/`, `training/`, `eval_datasets/` resolved with `ast`: `bench` → `training.encode`, `eval_datasets.{layout,hub}`; `training` → `eval_datasets.{hub,layout,timesplit}`; `eval_datasets` → nothing; the library only through `retrieve` (the X §2 names), `retrieve.functional`, `retrieve.interfaces` |
| `bench/test_paths.py` | `PATHS == derive(DISPATCH)`: the derived table equals the harness's expected paths, the grid is complete, `DISPATCH` names every algo × backend |
| `bench/test_records.py` | `resume_key` canonical and `code_version`-sensitive; append / read round trip; one torn trailing line; `flatten` one row per perf entry, last record per key |
| `bench/test_measure.py` | `stats` vs numpy, `latency` control flow, `index_bytes` dedup, `provenance` git fields, `dirty` scoped to the library subtree with the `files:` fallback, `repo_dirty` excluding `results/`, `atomic_write`, `clocks` shape, `graph_callable` refusals |
| `bench/test_metrics.py` | padding / IDCG / denominator contracts; running sums equal the pre-v2 per-row means to 1e-9; `jaccard_at_k` |
| `bench/test_algos.py` | `build` on every `(algo, filter_kind)` torch cell: the `k` setter slices the top-k and changes no buffer; the filter submodule in `index_bytes`; `set_query_params`; build refusals |
| `bench/test_config.py` | job counts and keys per suite on `tests/bench/data/{mini,text,suites}.yaml`; `disabled`; build/query split; seeds; narrows and `Job.narrowed`; the real `goodreads` / `arxiv` d128 cell sets |
| `bench/test_inputs.py` | `load_inputs` on the conftest writer, `users_limit` once, prefix and row-count checks, the legacy layout loading equal to the modern one, misalignment raising, `attrs_digest`, `sweep_qa`, filters by filter backend, `query_pool` |
| `bench/test_oracle.py` | padding, v4 fields and arithmetic, fingerprint in the file name, an unreadable blob rebuilt, bloom FP rate, `code_version` |
| `bench/test_run.py` | end to end on the tiny fixture, both modes (`graph` = the CPU null entry): record schema, resume, `code_version` invalidation, `partial`, a failed cell + continue, a sticky CUDA error, the quality gate, the parity spill, reachable-target masking, the plan cache off through `OfficialConfig` |
| `bench/test_cli.py` | `bench run` via `CliRunner`, a real one-child `bench campaign`, a faked timed-out child, zero cells → exit 1, `bench report` over the campaign's own records |
| `bench/test_upload.py` | `bench upload` with no network: `_logs/` and `_parity/` never listed, the manifest's sha256 per file, evidence beating `--gate` four ways, the generated README over several subtrees, `verify` catching a changed byte, the commit's operations and `private=True` |
| `bench/test_report.py` | every column the tables read still comes out of `records.flatten`; every artifact emitted; the LaTeX structurally balanced with the thesis's labels and no unescaped `_`; a `failed` record excluded and a `partial` / `unstable` one marked; citability off by default and evidence beating `--gate`; an empty tree; a schema-1 record |
| `bench/test_c4_gate.py` | the C4 gate script against synthesised schema-1 records |
| `eval_datasets/test_layout.py` | the legacy pad-row rule, `apply_users_limit`, `validate_layout` clean on both layouts and flagging a short `eval_split`, a missing or swapped prefix sidecar, misaligned attrs |
| `eval_datasets/test_yfcc.py`, `test_pubmed.py` | the two ETL loaders on synthetic fixtures (moved from `eval_datasets/tests/`) |
| `training/test_encode.py` | `training.evaluate`'s recall / ndcg equal `bench.metrics` to 1e-9 (V D5); `encode_split`'s cache hit / stale key |

## How to run

The repo is a uv workspace; `uv run` from inside `evaluation/` finds it.
On a box that allows it, lock the clocks first (`sudo nvidia-smi -pm 1 &&
sudo nvidia-smi -lgc 1410`); on this A100 container that is denied, so
read latencies against `perf[].sm_mhz` and `env.clocks_drift`.

```bash
cd evaluation
# the C4 gate cell set: goodreads d128, clause c0_genre, every algo and backend
uv run bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre
# one cell, eager only, no perf — the fastest iteration
uv run bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch --backend triton \
    --filter-kind bloom --sweep c0_maincat --mode eager --skip-perf
# the campaign (roadmap D1)
uv run bench campaign --suite all --resume
uv run bench upload --results results --path-in-repo d1-a --verify   # publish, then check the round trip
```

Sanity checks after a run (H WP-5's gates): `median_ms(bs=16) <
16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical
in `quality`; `memory_reserved_mib` flat (±5 %) across a group's cells;
no `unstable` cell at locked clocks.

## Extending

**A new algorithm.** A library module first (the LiNR variants are
`retrieve.modules.linr`; a new composition is added there —
[library-harness-boundary.md](../plans/library-harness-boundary.md) §2):
the filter as `self.filter`, `forward(query, query_clause_attrs=None) ->
(ids, scores)`, `k` forwarding to the final top-k layer, `set_query_params`
for any query-time knob, `capturable` as a class attribute, a `DISPATCH`
row. Then one line in [`algos.py`](../../evaluation/bench/algos.py)'s
`ALGOS` (and a branch in `build` if it registers differently), a suite's
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
with `RETRIEVE_DATA_ROOT=/some/path` (on the A100 box `/workspace/data`,
with `evaluation/data` a symlink to it).

## Old results

[`evaluation/results/archive/`](../../evaluation/results/archive/) holds
the old harness's `<name>.json` + `.yaml` + `.perkernel/` campaign outputs
(pre-schema, pre-oracle-fix; not citable), moved there at C5 (H §5). The
harness writes `results/<suite>/*.jsonl`. A1's golden cells live under
[`evaluation/golden/`](../../evaluation/golden/README.md) — information,
not a gate, since the 2026-09-15 steer.

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


**Official backend and the plan cache.** `SilverTorch(backend=
"official")` in bloom mode parses each query batch into official
expression plans on the CPU (≈ 59 µs per call at B=16, plan §13.2 —
10–20 % of an eager bloom forward) and memoises the result per
expression tuple by default. `latency()` replays one fixed batch, so
after the first call a cached official cell never pays the parse and
its latency is not what serving fresh queries costs. An official cell
that is timed must run with `OfficialConfig(cache_plans=False)` (every
forward parses; results are bit-identical) or report both settings as
separate, labelled rows. The harness does the former (C4): the module
is built with the library's default config (cache on) and quality runs
against it; `run.perf` then replaces `module.official` with
`OfficialConfig(cache_plans=False)` — `cache_plans` is read per forward by
`parse_plans`, nothing at build depends on it, so this is an in-place
`dataclasses.replace`, no rebuild — before the first timed variant, and
every perf entry records `cache_plans` (`false` there, `null` on the
backends that have no such cache). A cell whose perf entries say
`cache_plans: true` is not a timing number.
