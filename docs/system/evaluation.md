# `evaluation/` retrieval harness (v2)

Live reference for the retrieval benchmark that produces the thesis and
paper numbers: the measurement protocol, the module map, the config
matrix, the `bench` CLI, the JSONL record, resume and the campaign process
model. It describes the code as it is on the `dev/c1-harness-v2` branch
after roadmap steps C1–C3 ([00-roadmap.md](../plans/00-roadmap.md) Phase
C); the plan it implements is
[evaluation-harness-v2.md](../plans/evaluation-harness-v2.md) (H). **Nothing
this harness has produced is citable until the C4 gate on the A100 is
green** (CLAUDE.md rule 2): everything below was authored and CPU-tested on
the Mac, and the GPU paths (Triton kernels, CUDA-event timing, graph
capture, `nvidia-smi` clocks, the SASRec encode) run for the first time in
C4.

For the algorithm internals see [kernels.md](kernels.md) and
[architecture.md](architecture.md); for the datasets on disk and the
SASRec checkpoints, [datasets.md](datasets.md) and
[checkpoints.md](checkpoints.md); for the library's correctness suite,
[testing.md](testing.md).

## Measurement protocol

Reproduced verbatim from H §2 (the amendment at the top of H applies: read
every `cuda` / `cute` as `official`, whose `graph` mode is recorded as
`null` with reason `not_capturable`). Where the code deviates, the
[Deviations](#deviations-from-h) section says so.

One **cell** = `(dataset, dim, filter_kind, sweep, algo, backend, params, seed)`. Per cell, in
one process, in this order. Everything below is `eager` unless labelled `graph`.

1. **Environment, once per process.** Seed torch/CUDA; TF32 off (one function); record
   provenance: GPU name, driver, CUDA runtime, torch/triton versions, git commit + dirty flag,
   hostname, UTC start, `nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.limit`
   sampled *after* warm-up. Clock locking is recommended, not enforced (`nvidia-smi -pm 1 &&
   nvidia-smi -lgc 1410` in the runbook); the harness warns when `clocks.sm` drifts >5 % from the
   first cell. One 3-matmul GPU warm-up (keep `warm_gpu_once`).
2. **Inputs, once per (dataset, dim).** Item embeddings, queries, held-out targets, query attrs;
   `users_limit` applied in exactly one place (prefix, as today, so quality stays golden-comparable).
   Filter modules: one per *filter backend* (`triton` for triton/cuda/cute cells, `torch` for torch
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
the resume key, `git_branch` / `python` / `clocks_locked` in `env`, IQR and
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
  the `(dataset, dim, algo)` group closes. A missing reference (the first
  backend's cells were skipped by `--resume`) records `null` values.
- **Samples go to a JSONL sidecar** (`<name>.samples.jsonl`, one line per
  perf entry with the key block, `k`, `bs`, `mode`, `ms: [...]`), not a
  parquet: parquet cannot be appended per cell. `bench report` (D4) converts.
- **Resume re-runs `failed` and `partial` records**; only `status: ok` at the
  same `code_version` counts as done. `report.py` must take the last record
  per key.
- **The campaign child is `python -m retrieval.cli run …` on the same
  interpreter**, not `uv run bench run …` (no second `uv` resolution per
  child; `uv run bench campaign` already resolved the environment).
- **Exact algos** for the §2.4 gate are `linr_v1_filter_mask` and `linr_v2`
  (`linr_v4` is int8 and not exact); the gate records the cell as `failed`
  before raising `QualityGateError`.
- `--flush-l2` is not implemented (H mentions it as a C4 throwaway).
- `build_s` is recorded on every cell of a build (a `deep` job with six
  `n_probe` values repeats the same `build_s` six times).

## Architecture

Ten modules under [`evaluation/retrieval/`](../../evaluation/retrieval/),
2,657 lines including docstrings (`wc -l`) + 1,699 of tests, against H
§3.1's budget of 1,450 + 350. The tests are over budget by design — C1/C2
locked the metrics, the `PATHS` table and the config expansion against the
old harness — and the code is over mostly in `run.py`'s record assembly
and `data.py`'s two loaders; docstrings are a third of the count.

| module | lines | owns |
|---|---:|---|
| [`bench.py`](../../evaluation/retrieval/bench.py) | 349 | `setup`, `warm_gpu_once`, `provenance` (GPU, driver, CUDA, torch, triton, commit, `dirty` = `subtree_dirty()` over `retrieve/src/retrieve`, `repo_dirty`, branch, `code_version` = the subtree's tree hash, or `files:<sha256>` of the sources on disk when the subtree is dirty, host, python, started), `clocks(expected_sm_mhz)` (`clocks_locked` = within 2 % of the expectation), `timed_build`, `index_bytes` (Σ buffers, submodules included, deduplicated), `stats`, `latency(fn, bs=, mode=)` (§2.5 windows, IQR + outlier counts, `load: closed_loop`, `peak_fwd_mib`), `graph_callable` (raises `NotCapturable` with the record's `reason`), `profile_once` |
| [`metrics.py`](../../evaluation/retrieval/metrics.py) | 111 | `accumulator(ks, device)` / `accumulate(acc, ids, targets, num_targets=None, ranked=False)` / `finalize(acc)` — recall, ndcg, precision, mrr at every `k` from one top-`k_max` list as float64 running sums on device; `ranked=True` scores against the oracle's own top-`k` prefix (the old per-`k` `nt_k`); `per_row`, `jaccard_at_k`. `training/evaluate.py` shares it |
| [`algos.py`](../../evaluation/retrieval/algos.py) | 328 | the five `nn.Module` wrappers (`LinrV1`, `LinrV2`, `LinrV3`, `LinrV4`, `Silvertorch`) with the filter as a submodule, `k` settable, `set_query_params` (`n_probe`, `candidate_pool`); `ALGOS`, `FILTER_KINDS`, `BACKENDS`, `FILTER_BACKEND`, `CAPTURABLE`, `PATHS`; `build`, `build_filter`, `is_valid_combo` |
| [`config.py`](../../evaluation/retrieval/config.py) | 365 | `Dataset`, `Job`, `load_dataset`, `load_matrix` — the config matrix below |
| [`data.py`](../../evaluation/retrieval/data.py) | 291 | `load_inputs` (SASRec encode cached on the full split, or pre-encoded text; `users_limit` once, as a prefix), `sweep_qa`, `build_filters` (keyed by filter backend), `exact_filter`, `query_pool` |
| [`oracle.py`](../../evaluation/retrieval/oracle.py) | 264 | the exact filtered oracle as blob v4, `pass_counts`, `pass_rate`, `bloom_fp_rate`, `resume_key`, `KEY_FIELDS` |
| [`run.py`](../../evaluation/retrieval/run.py) | 582 | `run(jobs, out_dir=...)` — the cell loop, `append_record`, `read_keys`, `record_path`; `SCHEMA_VERSION = 1`, `MODES`, `QUALITY_CHUNK = 16`, `EXACT_ALGOS`, `PERF_STAT_KEYS` |
| [`cli.py`](../../evaluation/retrieval/cli.py) | 190 | `bench run` / `bench campaign` / `bench report` |
| [`upload.py`](../../evaluation/retrieval/upload.py) | 60 | `upload-results`: mirror `results/` to a HF dataset repo |
| [`encode.py`](../../evaluation/retrieval/encode.py) | 117 | `load_model_for_eval`, `encode_queries` — the one file that imports `training.*` |

Two things are classes, on purpose (H §3.1): `Job` — the fully resolved
build spec whose `key(params)` *is* the record's key block and the resume
key — and the five algo `nn.Module`s, which exist for `buffers()`
(memory), `.k` (perf per `k`), `torch.compile` and because their `forward`
is the readable spec of each cascade. No base class, no context bags, no
stats dataclasses, no results-IO layer.

### Algorithms and the `PATHS` table

`PATHS[(algo, filter_kind, backend)]` names the code path that actually
runs, or `None` when there is no such cell. It is what a record's `path`
column carries and what `load_matrix` collapses on: backends that run the
same code become one job (logged once), `None` triples are skipped. The
table agrees with the dispatch table in
[architecture.md](architecture.md#backend-dispatch) — `tests/test_algos.py`
parses that markdown and asserts it.

| algo | `none` | `clause` / `bloom` | `official` |
|---|---|---|---|
| `linr_v1_filter_mask` (`PostfilterKNN`, fp16 cuBLAS + mask) | `cublas` (triton and torch collapse) | `cublas+triton` / `cublas+torch` (the filter's kernel) | — |
| `linr_v2` (`PrefilterKNN` over the filter's candidate list) | — (the candidate source is the filter) | `triton` / `torch` | — |
| `linr_v3` (`OneBitKNN` top-`candidate_pool` → `PrefilterKNN`) | `triton` / `torch` | `triton` / `torch` | — |
| `linr_v4` (`PostfilterKNNInt8`, `_int_mm` + mask) | `cublas` | `cublas+triton` / `cublas+torch` | — |
| `silvertorch` (IVF + INT8, predicate fused: `filter_mode` none / exact / bloom) | `triton` / `torch` | `triton` / `torch` | `official` |

`official` is Meta's reference backend (roadmap B1) and exists for
`silvertorch` only; its standalone filter modules are Triton
(`FILTER_BACKEND`), and it is eager-only (`CAPTURABLE`), so its `graph`
perf entries are `null` with `reason: not_capturable`. Until B1 lands in
`retrieve`, `algos.build(backend="official")` raises `NotImplementedError`,
which a campaign records as `status: failed` on those cells.

Every wrapper is `forward(q, qa=None) -> (ids [B, k], scores [B, k])`,
`torch.topk`-sorted rows, `-1` ids where a row has fewer than `k`
survivors. `k` is a property that forwards to the layer owning the final
top-k; `Silvertorch.k` and `set_query_params(n_probe=)` re-run the two
`register_index` validations. `build(algo, item_embs, k=, backend=,
filter_kind=, filter_mod=, item_attrs=, clause_is_reverse=, params=,
seed=)` is the one factory; it refuses `None`-path cells and
`n_probe > n_lists`. `params` are the merged build + query params; on
`silvertorch` bloom cells `run.py` merges the suite's `bloom` defaults
(`m_bits`, `k_hash`) in as well.

## Config: one YAML per dataset + `suites.yaml`

Five files under [`evaluation/config/`](../../evaluation/config/):
[`goodreads.yaml`](../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../evaluation/config/arxiv.yaml),
[`yambda-500m.yaml`](../../evaluation/config/yambda-500m.yaml),
[`yambda-5b.yaml`](../../evaluation/config/yambda-5b.yaml) and
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
               [--resume|--force] [--expected-sm-mhz 1410] [--config-dir config]
bench campaign --suite quality|filter|deep|all [--dataset D]* [--dim N]* [--mode M]*
               [--skip-quality] [--skip-perf] [--profile] [--out results] [--resume|--force]
               [--config-dir config]
bench report   [results]                        # exits 2: roadmap D4 (H §6 WP-6)
upload-results --repo-id user/repo [--results results] [--private] [--dry-run]
```

`*` = repeatable. `bench run` expands one `(dataset, suite)` through
`load_matrix` (every repeatable option is a narrow) and runs the cells in
*this* process; `--resume` (the default) skips cells already `ok` at the
current `code_version`, `--force` re-runs them (their old records stay in
the file). `--mode` defaults to both; `--expected-sm-mhz 0` disables the
`clocks_locked` expectation; `--output` overrides the per-`(suite,
dataset, dim)` file. `--k`, `--bs` and `--mode` *replace* the suite's
lists rather than select cells, so a run with any of them (or a
`--skip-*` flag) writes `status: partial` records — resume re-runs them,
and the next full run is never fooled by an iteration-day cell. The exit
code is 1 when any cell failed, and 1 with a message when the narrows
select zero cells (a `--sweep` typo is an error, not an empty success).

`bench campaign` is the loop of H §3.4 / §8.2 K: for every suite (in the
order quality, filter, deep for `all`), every listed dataset and every
`(dataset, dim, algo, backend)` group of `load_matrix`, one child process
`python -m retrieval.cli run --dataset … --dim … --suite … --algo …
--backend … --resume` (same interpreter, `cwd = evaluation/`), sequential.
The child's stdout + stderr go to
`results/_logs/<suite>_<dataset>-d<dim>_<algo>_<backend>.log` (appended, the
command line first); one summary line per child (`time suite dataset dim
algo backend rc seconds log`) goes to `results/_logs/campaign.log` and the
terminal; a non-zero rc is recorded and the loop continues; the exit code
is the worst child rc, or 1 when a listed dataset expands to no groups or
no child was launched at all (`--dataset` / `--dim` selecting nothing).
`results/_parity/` is deleted when the `(dataset, dim, algo)` group
closes. A backend is the thing under test, so it gets
the process: no dynamo cache, allocator arena or CUDA-graph pool outlives
it, at the cost of a dataset reload and a CUDA context init per group.

## The cell loop (`run.py`)

`run(jobs, *, out_dir, out_path=None, resume=True, modes=MODES,
skip_quality=False, skip_perf=False, profile=False, expected_sm_mhz=1410,
env_extra=None, latency_kw=None, device=None) -> Counter` runs the jobs in
order:

1. once: `bench.setup(seed)`, `warm_gpu_once`, `provenance()` (+
   `env_extra`, the CLI's `config_sha`), a first `clocks()` sample;
2. per `(dataset, dim)`: `data.load_inputs` (with attrs when any job of the
   group is a filter cell); the previous inputs are freed first;
3. per `(filter_kind, sweep, filter backend, k_max, bloom params)`:
   `sweep_qa` → `build_filters` → on filter cells `exact_filter` +
   `oracle.load_or_build` (blob v4 at `k_max = max(ks)`) and on bloom cells
   `pass_counts` → `bloom_fp_rate`; the row masks: `keep` (not
   skip-masked), `oracle_rows` (kept with ≥ 1 survivor — the old harness's
   zero-target skip), `heldout_rows` (kept, ≥ 1 target, and on filter cells
   `target_in_filter`);
4. per job: `bench.setup(job.seed)`, one `timed_build` of
   `algos.build(...)` with the first cell's params; `index_mib`,
   `filter_mib` (the module's `filter` submodule, 0 without one); a build
   failure writes a `failed` record per cell of the job;
5. per cell: `set_query_params` for the query part of `params`; `clocks()`
   (drift > 5 % from the first sample → warning + `clocks_drift`); quality
   (`module.k = k_max`, chunks of 16 kept rows, `accumulate(...,
   ranked=True)` against the oracle prefix for `oracle_rows`,
   `accumulate(..., targets, n_targets)` for `heldout_rows`; row selection
   by CPU masks + `index_select`, so no per-chunk sync); the parity spill;
   the exact-algo gate; perf per `(bs, k, mode)` (`query_pool` per bs,
   `module.k = k`, `graph_callable` or a null entry with `reason`,
   `latency`, optional `profile_once`); `memory_reserved_mib`; one
   `append_record`; the samples sidecar; `torch._dynamo.reset()` per bs;
6. after each job: drop the module, `gc.collect()`, `torch._dynamo.reset()`,
   `empty_cache()`.

Any exception inside a cell (an OOM on the torch path included) becomes a
`status: failed` record with the traceback and `stage` (`build`,
`query_params`, `quality`, `perf`) and the loop continues. Two exceptions
stop the process: `KeyboardInterrupt`, and `QualityGateError` — an exact
algo (`linr_v1_filter_mask`, `linr_v2`) below `recall_oracle@k_max ≥ 0.99`
— which is recorded first.

## Output: one JSONL record per cell

`results/<suite>/<dataset>-d<dim>.jsonl`, appended by the process the
moment a cell finishes (`json.dumps` of one line, `allow_nan=False` — NaN
and ±inf become `null` — then `write` + `fsync`). Nested, not wide:
`polars.read_ndjson(path).explode("perf").unnest("perf")` flattens the
perf entries.

| field | type | value |
|---|---|---|
| `schema_version` | int | `1` |
| `status` | str | `ok`, `partial` (the record does not carry everything the suite asked for), `failed` |
| `partial_reasons` | list / null | why `partial`: any of `skip_quality`, `skip_perf`, `modes` (a `--mode` subset), `ks_bs` (`--k` / `--bs` replaced the suite's lists — `Job.narrowed`) |
| `dataset`, `dim`, `suite`, `filter_kind`, `sweep`, `algo`, `backend`, `params`, `seed` | | the key block = `Job.key(params)` (`KEY_FIELDS`); `params` is the native dict of build + query params (`{}` when the algo takes none) |
| `path` | str | `PATHS[(algo, filter_kind, backend)]` |
| `n_items`, `n_queries` | int | catalogue size, queries after `users_limit` |
| `n_kept` | int | queries not skip-masked (every query on `none` cells) |
| `n_queries_oracle` | int / null | kept queries with ≥ 1 survivor (the oracle metrics' `n`); `null` on `none` cells |
| `n_queries_heldout` | int | queries the held-out metrics cover (kept, ≥ 1 target, `target_in_filter` on filter cells) |
| `pass_rate` | float | exact mask pass rate over kept queries (`1.0` on `none`) |
| `bloom_fp_rate` | float / null | mean per-query `(bloom − exact) / (N − exact)`; bloom cells only |
| `bloom` | dict / null | `{m_bits, k_hash}` on bloom cells |
| `k_max`, `ks`, `batch_sizes` | | the suite's, `k_max = max(ks)` |
| `build_s` | float | construction + `register_index`, sync on each side (same value on every cell of one build) |
| `index_mib` | float | Σ buffers of the algo module, filter submodule included |
| `filter_mib` | float | Σ buffers of the filter submodule alone (`0.0` without one; `silvertorch` carries its attrs inside `index_mib`) |
| `quality` | dict / null | `heldout: {recall@k, ndcg@k, precision@k, mrr@k for k in ks, n}`; on filter cells also `oracle: {…}` (ranked-prefix targets); `jaccard_vs_first@k` per `k`, `score_max_abs_diff`, `parity` (`reference` = this record wrote the spill file, `vs_<backend>` = compared against it, `shape_mismatch:…`); `null` with `--skip-quality` |
| `perf` | list / null | one entry per `(bs, k, mode)` (table below); `null` with `--skip-perf` |
| `unstable` | bool | any perf entry `unstable`, or `clocks_drift` |
| `memory_reserved_mib` | float / null | `torch.cuda.memory_reserved()` after the cell — the leak detector across a group's cells |
| `elapsed_s` | float | wall time of the cell |
| `env` | dict | `gpu, driver, cuda, torch, triton, commit, dirty, repo_dirty, git_branch, code_version, host, python, started, config_sha` + the cell's `sm_mhz, mem_mhz, sm_max_mhz, power_limit_w, expected_sm_mhz, clocks_locked, clocks_drift` |
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
| `load` | `"closed_loop"` |
| `kernels` | `--profile`, eager only: top-8 CUDA kernels `{kernel, us, calls}` |
| `reason` | present when the variant could not run (`not_capturable`, `cuda_unavailable`, `cudagraph_skips=N`, `cudaGraphLaunch per call = N, expected 1`); every stat key is then `null` |

`results/<suite>/<dataset>-d<dim>.samples.jsonl` holds the per-call vector
of the chosen window: one line per perf entry, `{key block, k, bs, mode,
ms: [...]}`.

### Resume

The resume key is `oracle.resume_key(job.key(params), code_version)` —
canonical JSON of the key block plus `bench.code_version()`: the library
subtree's tree hash (`git rev-parse HEAD:retrieve/src/retrieve`) when the
subtree is clean, else `files:<sha256>` over the `retrieve/**/*.py` sources
actually on disk (also the value outside a git checkout; the two namespaces
are disjoint). A kernel edit — committed or not — therefore invalidates
every cell; a doc or plan edit invalidates none. `run.read_keys(path)` rebuilds
the key from a record as `resume_key({k: rec[k] for k in KEY_FIELDS},
rec["env"]["code_version"])` and keeps the last status per key; a cell is
skipped when that status is `ok`. `--force` runs everything and appends.

Two dirty flags, both `null` outside a git checkout: `env.dirty` is
`git status --porcelain -- retrieve/src/retrieve` (untracked files
included — a new kernel module is measured code too) and is the flag
`report.py` refuses to cite (H §8.2 F); `env.repo_dirty` is the tracked
files anywhere else (`--untracked-files=no`, informational: a docs or
harness edit). `evaluation/results/` is excluded from `repo_dirty`: the
harness's outputs are data — committed and mirrored to HF by
`upload-results` (H §3.2, §8.2 G/I) — so a campaign appending to a
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
| `fingerprint` | sha256 over shapes, dtypes and a 64-row linspace sample of `item_embs`, `queries`, `targets`, `qa_sweep`; the full bytes of `item_attrs` and `clause_is_reverse` (`oracle.attrs_digest`, computed once per `(dataset, dim)` in `data.load_inputs` as `inputs["attrs_digest"]`); plus `clauses` and `k_gt` |
| `code_version`, `harness_commit`, `torch`, `created` | provenance |

The fingerprint is in the file name, so a stale blob is never read (a
different dim off the same `data_dir`, regenerated attrs — even one
edited value, since the item side is hashed in full — a retrained
checkpoint, another `users_limit` or clause set each produce a new file),
and the blob is a portable artifact for roadmap F4. The filter passed in
is always exact (`data.exact_filter`: the clause module itself on `clause`
cells, a fresh `ExactAttributeFilter` on `bloom` cells) — bloom's false
positives never leak into ground truth. Bloom pass rates are not cached.

## Inputs (`data.py`)

`load_inputs(ds, device, with_filters=True)` returns `item_embs [N, D]`
fp32 on device; `queries [U, D]`, `targets [U, T]` (`-1`-padded,
0-indexed), `n_targets [U]` on CPU; `qa [U, C]` int64 CPU, `item_attrs
[N, C, A]` and `clause_is_reverse [C]` on device (or `None`) with their
full-bytes `attrs_digest` (the oracle fingerprint's item side, hashed
once here); `n_items`, `n_queries`. Two loaders, keyed on the `Dataset`:

| `Dataset` field set | path |
|---|---|
| `checkpoint` | SASRec: `encode.load_model_for_eval` + `encode_queries` over `test.parquet`, cached as `<ckpt-dir>/encoded_queries_v2.pt` keyed on ckpt mtime + `max_seq_length`, the *full* split (a free-disk budget of `0.7 × free − 4 GiB` guards the write); the padding row is dropped, target ids shifted −1 |
| `content_dir` | pre-encoded text: `text_emb.pt` (or `shard_index.json` + shards) / `query_emb.pt` + `.meta.json` sidecars whose nomic prefixes are asserted, `heldout.parquet` (1-indexed → 0-indexed); fp16 → fp32 + L2-normalise |

`eval_split.parquet`'s row count is checked against the *full* split
before `users_limit` trims queries, targets, `n_targets` and `qa` together
as a prefix — the one `users_limit` site (kept a prefix, not a sample, for
golden comparability; H §7). `sweep_qa(qa, clauses)` codes inactive
clauses `-1` and skip-masks rows left without a live clause;
`query_pool(inputs, qa_s, skip, bs=, seed=, n_pool=4096, device=)` draws
the fixed-seed perf pool from the kept rows (the same draw as the old
harness at the same seed).

## Tests

[`evaluation/retrieval/tests/`](../../evaluation/retrieval/tests/), all
CPU-only (`backend="torch"`; the Triton, CUDA-event, graph-capture and
SASRec paths run in C4):

```bash
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest retrieval/tests/ -q
```

| file | checks |
|---|---|
| `test_bench.py` | `stats` vs numpy, `latency` control flow, `index_bytes` dedup, `provenance` git fields, `clocks` shape, `graph_callable` refusals |
| `test_metrics.py` | padding / IDCG / denominator contracts; running sums equal the pre-v2 per-row means to 1e-9 (fixed and ranked targets); `jaccard_at_k` |
| `test_algos.py` | `PATHS` covers the grid and agrees with architecture.md's dispatch table (parsed from the markdown); `k`-slice invariance of every layer and wrapper (no buffer changes with `k`); `set_query_params`; build refusals |
| `test_config.py` | job counts and keys per suite on `tests/data/{mini,text,suites}.yaml`; `disabled`; build/query split; seeds; narrows; the real `goodreads` / `arxiv` d128 cell sets equal the deleted YAMLs' |
| `test_data.py` | the pre-encoded loader on a tmp fixture, `users_limit` once, prefix and row-count checks, `sweep_qa`, filters by filter backend, `query_pool` |
| `test_oracle.py` | padding, v4 fields and arithmetic, fingerprint in the file name, bloom FP rate, `code_version`, `resume_key` |
| `test_run.py` | end to end on `conftest.py`'s tiny fixture: record schema, resume, `code_version` invalidation, a failed cell + continue, the quality gate, the parity spill, JSON sanitising |
| `test_cli.py` | `bench run` via `CliRunner`, a real one-child `bench campaign` (`--skip-perf`), the `report` stub |

## How to run

The repo is a uv workspace; `uv run` from inside `evaluation/` finds it.
On the A100 lock the clocks first (`sudo nvidia-smi -pm 1 && sudo
nvidia-smi -lgc 1410`; `env.clocks_locked` records whether it held).

```bash
cd evaluation
# the C4 gate cell set: goodreads d128, clause c0_genre, every algo and backend
uv run bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre
# one cell, eager only, no perf — the fastest iteration
uv run bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch --backend triton \
    --filter-kind bloom --sweep c0_maincat --mode eager --skip-perf
# the campaign (roadmap D1)
uv run bench campaign --suite all --resume
uv run upload-results --repo-id <user/repo> --dry-run
```

Sanity checks after a run (H WP-5's gates): `median_ms(bs=16) <
16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical
in `quality`; `memory_reserved_mib` flat (±5 %) across a group's cells;
no `unstable` cell at locked clocks.

## Extending

**A new algorithm.** One `nn.Module` in [`algos.py`](../../evaluation/retrieval/algos.py):
build the `retrieve` layers in `__init__`, register the filter as
`self.filter`, `forward(q, qa=None) -> (ids, scores)`, a `k` property
forwarding to the final top-k layer, `set_query_params` for any
query-time knob; add it to `ALGOS`, teach `_path` its code path, add it
to a suite's `algos:` in `suites.yaml`. `tests/test_algos.py` will demand
the `k`-slice invariance and the dispatch-table agreement.

**A new dataset.** One `config/<dataset>.yaml` (above) and the on-disk
artifacts under `data/<dataset>/` — `item_id_map.json`, `train/val/test.parquet`
+ a checkpoint for the SASRec layout, or `heldout.parquet` + `content*/`
(`text_emb.pt`, `query_emb.pt`, their `.meta.json` sidecars) for the
pre-encoded layout; filter sweeps add `item_attrs_narrow.pt`,
`clause_is_reverse_narrow.pt` and `eval_split.parquet` (see
[datasets.md](datasets.md) for the ETL and [filtering.md](filtering.md)
for the predicate spec). Add the dataset to the suites it belongs in.
The dataset CLIs are console scripts: `uv run yambda prep …`, `uv run
arxiv all …`, `uv run goodreads all …`.

### HuggingFace I/O

[`eval_datasets/hf_io.py`](../../evaluation/eval_datasets/hf_io.py) is
the single source of truth for HF reads / writes: `EVAL_REPOS` maps
each dataset to its `pinkmeme/eval-<dataset>` HF dataset repo (eval
inputs + `checkpoints/<ckpt-id>/`), `RAW_REPOS` maps upstream raw
sources into `data/_raw/<source>/`.

```bash
uv run eval-fetch yambda-500m            # pull eval inputs to data/yambda-500m/
uv run eval-fetch arxiv-papers --dims d64,d128 --include-checkpoints
uv run eval-publish goodreads-work-id --dry-run
uv run eval-publish-checkpoint yambda-500m gsasrec-d128-drop0.5 --dry-run
```

The local data root resolves to `evaluation/data/` by default; override
with `RETRIEVE_DATA_ROOT=/some/path`.

## Old results

[`evaluation/results/`](../../evaluation/results/) still holds the old
harness's `<name>.json` + `.yaml` + `.perkernel/` campaign outputs
(pre-schema, pre-oracle-fix; roadmap §2: not citable). The new harness
writes `results/<suite>/*.jsonl` next to them; H §5 moves the old files to
`results/archive/` with the D1 rerun. A1's golden cells (roadmap A1) land
under `evaluation/golden/` on their own branch.

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
