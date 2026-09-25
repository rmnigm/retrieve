# How established ANN/IR benchmark harnesses are engineered

Research for: `evaluation/` harness rewrite (3,900 → ~1,800 lines), grounded against the
already-drafted `docs/plans/evaluation-harness-v2.md` (JSONL-per-cell + parquet-of-samples design).
All claims below are sourced from the frameworks' own repos/docs, fetched directly (GitHub
Contents API + raw file reads) in this session unless flagged otherwise in §5.

## §1. Summary table

| Framework | Result format | Sweep declaration | Isolation | Timing protocol |
|---|---|---|---|---|
| **ann-benchmarks** | 1 HDF5 file per `(dataset, count, algo[-batch], hash(build+query args))`; datasets `times`/`neighbors`/`distances`, `attrs` dict for metadata; recall computed lazily and **cached back into the same file** (`metrics/knn` group) | per-algorithm `config.yml`: `{point-type → metric → [defs]}`, each def has `run_groups.<name>.args` (build params, Cartesian product of value-lists) and `.query_args` (reused against the *same built index*, also Cartesian), or explicit `arg_groups` (named dict combos) | 1 Docker container per algorithm per invocation; `cpuset_cpus` pins 1 CPU, `mem_limit`, `timeout`, container removed after | best-of-`run_count` wall clock, one query at a time (closed loop) or all of `X_test` at once under `--batch`; no explicit warmup phase |
| **big-ann-benchmarks** | same HDF5 module (forked `benchmark/results.py`), 1 file per `(dataset, algo, args)`; ground truth is a **shipped/downloaded static file** (`yfcc100m_query_gt100.bin`, or `azcopy`/DiskANN-CLI for streaming), never recomputed at eval time | per-submission `config.yml`: "1 index build config + up to 10 search configs (2 for streaming)"; one shared `run.py --dataset --algorithm --neurips23track` entrypoint for every team | 1 Docker container per team/algorithm run on **one standardized reference machine** (Azure D8lds_v5, or an IPMI-metered box for the T3 hardware track); 12 h build-time cap; CI (`neurips23.yml`) runs every PR's algorithm on a toy dataset before merge | filter/ood/sparse: QPS at fixed recall thresholds, same repeat-and-best-case timing as ann-benchmarks; streaming: runbook-driven insert/delete/replace/search, ranked on recall@10 at checkpoints inside a 1 h / 8 GB budget; T3 layers IPMI power sampling (1 s interval, kW·s) + $0.10/kWh cost on top |
| **VectorDBBench** | 1 dict/JSON leaderboard row per `(db, db_label, case)`: `qps`, `latency`, `recall`, `qp$`, `label`(pass/fail), `note`, `version`, `test_time`; appended to `vectordb_bench/results/leaderboard.json`, not one-file-per-run | declarative `Case`/`CaseConfig` registry (capacity / search-performance / filtering / full-text / streaming cases) + a batch-config YAML where a **list of dicts per DB = N runs**; `--num-concurrency 1,10,20` sweeps concurrency inline | none at the process level — the DB under test is a separately deployed service (cloud or user-managed container); `db_label`/`task_label` disambiguate result rows; the harness itself serializes to "one task at a time" | distinct runner classes per protocol: `serial_runner.py` (closed-loop, one query at a time, reports p50/p95/p99 + recall/ndcg/mrr), `concurrent_runner.py` (N workers, closed-loop, fixed wall-clock `duration`, QPS = completed/duration), `rate_runner.py` (open-loop fixed-rate **inserts**, not queries), `cold_warm_runner.py` |
| **MTEB / BEIR / ir_datasets / ir_measures** | 1 JSON file per `(model, model revision, task)`: `results/{model}/{revision}/{task}.json` with `dataset_revision`, `mteb_version`, `evaluation_time`, `kg_co2_emissions`, `scores.{split}[].{metric}`; results live in a **separate git repo** (`embeddings-benchmark/results`) from the eval code (`embeddings-benchmark/mteb`) | a Python model/task **registry** (`ModelMeta(name, revision, n_parameters, embed_dim, …)`, `mteb.get_model(name, revision)`); no YAML grid — "sweep" = which registered (model, task) pairs to run | none — single-machine researcher-run eval; reproducibility comes from **revision pinning** (model git SHA + dataset git SHA + mteb package version, all recorded in every result file), not process/container isolation | not a systems benchmark, no latency measured; `ir_measures` exists to give every IR metric one canonical parseable name (`nDCG@10`, `AP(rel=2)@1000`) across 8 interchangeable scoring backends (`trec_eval`, `pytrec_eval`, `msmarco`, …) |
| **cuVS/RAFT `cuvs_bench`** (+ FAISS `benchs/`) | cuVS bench reuses the ann-benchmarks HDF5 convention almost verbatim, plus a build-vs-search label split in `data_export.py`; FAISS `benchs/*.py` just print/log per script — no shared schema | cuVS: per-algorithm YAML at `cuvs_bench/config/algos/<name>.yaml`, `groups.<name>.build`/`.search` = dicts of value-lists (same Cartesian idea as ann-benchmarks) plus an optional `constraints` callable to prune invalid build×search pairs; FAISS: no declarative sweep — a `ParameterSpace`/`OperatingPoints` object explores/prunes at run time | cuVS ships `rapidsai/cuvs-bench` (GPU) and `cuvs-bench-cpu` Docker images bundling **all** algorithms in one container (split only GPU/CPU, not per-algorithm); FAISS `benchs/` assumes one researcher-controlled process/machine — no isolation at all | cuVS follows ann-benchmarks' repeat/recall-vs-QPS model; FAISS autotune runs each config until `min_test_duration`/`n_experiments` stabilizes, keeps only the Pareto-optimal (recall, time) frontier |

## §2. Per-framework detail

### ann-benchmarks (erikbern)
Repo: <https://github.com/erikbern/ann-benchmarks>. No central `algos.yaml` any more (an old
top-level file by that name is gone from `main`); the format lives per-algorithm at
`ann_benchmarks/algorithms/<algo>/config.yml`, e.g. faiss
(<https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss/config.yml>):
```yaml
float:
  any:
  - base_args: ['@metric']
    constructor: FaissIVF
    disabled: false
    docker_tag: ann-benchmarks-faiss
    module: ann_benchmarks.algorithms.faiss
    name: faiss-ivf
    run_groups:
      base:
        args: [[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]]
        query_args: [[1, 5, 10, 50, 100, 200]]
```
`args` and `query_args` are each Cartesian-producted independently (`itertools.product` over the
inner lists), and **`query_args` is applied via `set_query_arguments()` against an already-built
index** — the index is built once per `args` combo, then re-queried for every `query_args` combo,
which is the concrete mechanism behind "build once, sweep query params for free." hnswlib's
config (<https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/hnswlib/config.yml>)
shows the alternative style, `arg_groups: [{M: 12, efConstruction: 500}]` — an explicit list of
named dicts instead of a full Cartesian product, used whenever not every combination of build
params is meaningful. Every entry carries a `disabled: true/false` kill switch so a flaky/slow
config can be dropped from a run without deleting its definition from git history.
Docker isolation is one container per algorithm per invocation
(`ann_benchmarks/runner.py::run_docker`, <https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/runner.py>):
`cpuset_cpus` pins the container to one CPU, `mem_limit` defaults to available RAM, `network_mode=host`,
code/data mounted read-only, `results/` read-write, container removed after `container.wait(timeout)`.
Results: `ann_benchmarks/results.py`'s `build_result_filepath` builds
`results/<dataset>/<count>/<algo>[-batch]/<md5-ish-hash-of-json(args+query_args)>.hdf5`; `store_results`
writes `attrs` (algo, dataset, build_time, index_size, batch_mode, run_count, best_search_time,
distance, count, plus anything `algo.get_additional()` returns) and three fixed-size datasets
(`times`, `neighbors`, `distances`, padded with `-1`/`inf`). `data_export.py`
(<https://github.com/erikbern/ann-benchmarks/blob/main/data_export.py>) walks every HDF5 file,
calls `compute_metrics_all_runs`, and writes one flat CSV — this is the framework's only rollup
step, fully decoupled from the raw per-cell files. Recall is **not** an ID-set intersection: `get_recall_values`
(<https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/metrics.py>) counts
how many returned distances are `<= true_kth_distance + epsilon` (`epsilon=1e-3`), so ties at the
k-th boundary don't spuriously fail recall; the result is memoized into `metrics/knn` inside the
same HDF5 file ("Found cached result" branch) so `data_export.py`/`plot.py` never recompute it
twice. Timing (`run_individual_query` in `runner.py`): `run_count` full passes over `X_test`, each
query timed with `time.time()`; the **fastest of the `run_count` passes wins** ("best of N," not
mean/median); `--batch` mode calls `algo.batch_query(X_test)` once and either uses
`get_batch_latencies()` per-item or divides total time evenly. Plotting is the Pareto convention
(recall on x, QPS on y, log scale) via `plot.py`/`create_website.py`. Explicitly out of scope
(README, <https://github.com/erikbern/ann-benchmarks/blob/main/README.md>): billion-scale datasets
(redirected to big-ann-benchmarks) and multi-threading (single CPU enforced unless `--batch`).
Also see the paper: Aumüller, Bernhardsson, Faithfull, "ANN-Benchmarks" (<https://arxiv.org/pdf/1807.05614>).

### big-ann-benchmarks (NeurIPS'21/'23)
Repo: <https://github.com/harsha-simhadri/big-ann-benchmarks>, an explicit fork/extension of
ann-benchmarks (credited in its README). Four NeurIPS'23 tracks — filter, streaming, sparse, ood
(<https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md>). **Filter
track**: YFCC100M, 10M CLIP-embedded images (`uint8`, dim 192), each with a bag of tags from a
200,386-word vocabulary; 100K queries each carry 1–2 required tags. The filtered ground truth is
**not computed per run** — `YFCC100MDataset`/`YFCCImagesDataset` in
`benchmark/datasets.py` (<https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py>,
lines ~723–900) point at a static shipped file, `yfcc100m_query_gt100.bin` (or a `_sampled_1m`/`_10m`
variant), downloaded like the rest of the dataset. Baseline: faiss on YFCC-10M hits 3,200 QPS at
90% recall on the reference machine. Submission = a PR adding `neurips23/<task>/<team>/` with a
`Dockerfile`, a Python class subclassing the track's `BaseANN` (e.g. `BaseFilterANN`), and a
`config.yml` specifying **exactly 1 build config + up to 10 search configs (2 for streaming)** —
this cap is the mechanism that keeps every submission's sweep small and comparable. All baselines
ran on one fixed machine, "Azure Standard D8lds v5 (8 vcpus, 16 GiB), Intel Xeon Platinum 8370C @
2.80GHz" — a single hardware SKU standing in for "controlled environment" across dozens of
independently-authored Docker images. A GitHub Actions CI job
(`.github/workflows/neurips23.yml`) runs each PR's algorithm against a toy dataset before it's
accepted. Results reuse ann-benchmarks' HDF5 module verbatim (`benchmark/results.py`); rollup is
`python data_export.py --output res.csv` then either `plot.py --dataset X --neurips23track Y`
(QPS-vs-recall Pareto, one PNG per dataset/track) or `eval/show_operating_points.py` for a
"best recall above a QPS threshold" table. **Streaming track**: 10M/30M-point MS Turing slices;
the index starts empty and must execute a YAML "runbook" of `insert`/`delete`/`replace`/`search`
operations (roughly a 4:4:1 mix) within a 1 h wall-clock / 8 GB DRAM budget; ranked by average
recall@10 across the search checkpoints in the runbook; ground truth per checkpoint is downloaded
via `azcopy` or computed offline with DiskANN's `compute_groundtruth` CLI
(`benchmark/streaming/compute_gt.py`) — again, a precomputed artifact, not a runtime computation.
The **NeurIPS'21 T3 "any hardware" track**
(<https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips21/t3/README.md>,
<https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips21/t3/LEADERBOARDS.md>)
adds power/cost as first-class leaderboard axes: IPMI `POWER_IN` sensors sampled at 1 s intervals
(`benchmark/sensors/power_capture.py`), summed to kW·s, priced at a flat $0.10/kWh, combined with
hardware SRP into a $-per-100K-QPS cost metric — four separate leaderboards (recall, throughput,
power, cost). Competition summary paper: <https://arxiv.org/abs/2409.17424>.

### VectorDBBench (Zilliz)
Repo: <https://github.com/zilliztech/VectorDBBench>. "Case" is the unit of comparison: capacity
cases (insert until full, report count), search-performance cases (fixed dataset size/dim,
QPS+latency+recall at various concurrency), filtering cases (int-filter and label-filter variants
— scalar-predicate ANN, directly analogous to our filter modes), full-text cases (BM25 over
MS MARCO/HotpotQA), and streaming/"insertion-under-load" cases. 30+ target systems (Milvus,
Qdrant, Weaviate, pgvector, Elasticsearch, Pinecone, …). Result rows
(<https://github.com/zilliztech/VectorDBBench/blob/main/vectordb_bench/results/leaderboard.json>)
carry `db`, `db_label`, `case`, `qps`, `latency`, `recall`, `qp$` (queries-per-dollar), `label`
(pass/fail/timeout), `note`, `version`, `test_time` — one flat row per (db config, case), which
**is** the framework's leaderboard format, not a separate export step. Sweeps are declared as a
batch-config YAML where a DB key maps to a **list** of config dicts (one run per list entry) plus
`--num-concurrency 1,10,20` for inline concurrency sweeps. No process/container isolation is
provided by the harness itself — the vector DB under test is a separately deployed service (local
container or managed cloud instance the user stands up); the harness only serializes itself to one
task at a time and uses `db_label`/`task_label` to keep result rows from colliding. Timing lives in
distinct runner classes under `vectordb_bench/backend/runner/`: `serial_runner.py` computes
p50/p95/p99 latency plus recall/ndcg/mrr in a closed loop (next query issued only after the last
returns); `concurrent_runner.py` spins up N worker threads, each closed-loop, for a fixed
wall-clock `duration`, and reports QPS = completed / duration; `rate_runner.py` is the one
open-loop path in the whole survey, but only for **insert** throughput, never for query load;
`cold_warm_runner.py` specifically measures cold- vs warm-cache read latency for cloud DBs. Public
leaderboard: <https://zilliz.com/benchmark>.

### MTEB / BEIR / ir_datasets / ir_measures
MTEB code: <https://github.com/embeddings-benchmark/mteb>. Results are **not** stored in the code
repo at all — they live in a sibling repo, <https://github.com/embeddings-benchmark/results>, at
`results/{model_name_with_/_as_double-underscore}/{model_revision}/{task_name}.json`. A concrete
file fetched in this session
(<https://github.com/embeddings-benchmark/results/blob/main/results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a/ARCChallenge.json>)
has top-level `dataset_revision`, `evaluation_time`, `kg_co2_emissions`, `mteb_version`, and
`scores.test[]` — one object per HF subset/language with `main_score` plus every metric variant
(`map_at_10`, `ndcg_at_10`, `mrr_at_10`, `nauc_*`, …). Every result file is thus self-describing
for reproducibility (exact model revision, exact dataset revision, exact package version) without
any process isolation — MTEB doesn't run your model for you, you run it locally and submit the
JSON. Model metadata is a Python object, `ModelMeta` (fields include `loader`, `name`, `revision`,
`n_parameters`, `embed_dim`, `max_tokens`, `license`, `release_date`,
<https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_meta.py>); resolution is
always `mteb.get_model(name, revision)` — **the revision is part of the identity**, exactly like
pinning a git SHA, which is why HF model-card-based result submission was disabled (a model card
can change after the fact; a results-repo PR with a pinned revision cannot). BEIR
(<https://github.com/beir-cellar/beir>) standardizes the *input* side: corpus as JSONL
(`_id`/`title`/`text`), queries as JSONL (`_id`/`text`), qrels as TSV
(query-id, corpus-id, relevance). `ir_datasets` (<https://ir-datasets.com/beir.html>) wraps BEIR
(and dozens of other collections) behind one iteration API and can re-export any of them
(`ir_datasets export beir/arguana qrels --format tsv`). `ir_measures`
(<https://ir-measur.es/en/latest/getting-started.html>, paper
<https://arxiv.org/abs/2111.13466>, building on `pytrec_eval`, <https://arxiv.org/abs/1805.01597>)
exists purely to give every metric one canonical, parseable name across 8 interchangeable scoring
backends: `nDCG@10`, `AP(rel=2)@1000`, `P(rel=2)@5` — `parse_measure('nDCG@20')` and
`parse_trec_measure('ndcg_cut_10')` both resolve to the same object, so downstream code never
special-cases which backend actually computed the number.

### cuVS/RAFT `cuvs_bench` + FAISS `benchs/`
cuVS bench (<https://docs.rapids.ai/api/cuvs/stable/cuvs_bench/>, package
`rapidsai/cuvs::python/cuvs_bench`) is a direct descendant of the ann-benchmarks design applied to
GPU algorithms. Per-algorithm YAML lives at
`python/cuvs_bench/cuvs_bench/config/algos/<name>.yaml`; a fetched example,
<https://github.com/rapidsai/cuvs/blob/branch-25.10/python/cuvs_bench/cuvs_bench/config/algos/hnswlib.yaml>:
```yaml
name: hnswlib
constraints:
  search: cuvs_bench.config.algos.constraints.hnswlib_search
groups:
  base:
    build:
      M: [12, 16, 24, 36]
      efConstruction: [64, 128, 256, 512]
    search:
      ef: [10, 20, 40, 60, 80, 120, 200, 400, 600, 800]
  test:
    build: {M: [12], efConstruction: [64]}
    search: {ef: [10, 20]}
```
— the same build/search split and Cartesian-product-of-value-lists idea as ann-benchmarks'
`run_groups`, plus an explicit `constraints` hook (a Python dotted path) to reject invalid
build×search combinations before running them, and a named `test` group deliberately kept tiny
for smoke-testing the harness itself. Distribution is via prebuilt Docker images,
`rapidsai/cuvs-bench` (GPU) and `cuvs-bench-cpu` (<https://hub.docker.com/r/rapidsai/cuvs-bench>),
each bundling *every* supported algorithm rather than one image per algorithm — isolation is
split only along the GPU/CPU axis, not per-algorithm, because (unlike ann-benchmarks) all the
GPU algorithms here share one CUDA/RAPIDS dependency stack maintained by one team. FAISS's
`benchs/` (<https://github.com/facebookresearch/faiss/blob/main/benchs/bench_all_ivf/bench_all_ivf.py>)
takes the opposite approach: no declarative config at all, no isolation, no shared result schema —
just a Python script per experiment using FAISS's own `ParameterSpace`/autotune machinery, which
explores parameter settings until `min_test_duration`/`n_experiments` stabilizes and keeps only
the (recall, time) Pareto frontier (`OperatingPoints`); results are whatever the script prints or
dumps to an ad hoc JSON (example logs: <https://github.com/facebookresearch/faiss/wiki/bench_all_ivf_logs-bigann100M>).

## §3. Conventions that recur across ≥3 frameworks

1. **Ground truth is a precomputed, versioned artifact, never a per-run computation.**
   ann-benchmarks ships top-100 neighbors inside the dataset HDF5; big-ann-benchmarks ships/downloads
   a static `.bin` (`yfcc100m_query_gt100.bin`) for the filter track and downloads/precomputes
   streaming-checkpoint GT once via `azcopy`/DiskANN CLI; cuVS bench reuses ann-benchmarks' shipped
   GT; MTEB pins `dataset_revision` specifically so qrels can't silently drift between runs.
2. **Sweeps are declared as data (YAML/JSON), Cartesian-producted over per-parameter value lists,
   never hand-enumerated in code.** ann-benchmarks `run_groups.args`, cuVS `groups.build`/`.search`,
   big-ann's "1 build + up to 10 search configs" cap in `config.yml`, VectorDBBench's list-of-dicts
   batch config.
3. **Build-time and query-time parameters are swept independently, and the index is built once per
   build-config, then re-queried for every query-config.** ann-benchmarks `args` vs `query_args`
   (`set_query_arguments` reuses the built index); cuVS `build` vs `search` groups; big-ann's config
   cap ("1 build + up to 10 search"); VectorDBBench separates its capacity/build cases from its
   search-performance cases.
4. **One result record per finest-addressable unit of comparison, written incrementally, doubling
   as the resume/crash-recovery mechanism.** ann-benchmarks/big-ann: one HDF5 file per
   (dataset, algo, args) — a file that already exists is already-done work; MTEB: one JSON per
   (model, revision, task); VectorDBBench: one appended row per (db, case).
5. **Every result record carries its own provenance/environment metadata inline**, rather than in
   a separate run log that has to be joined back. ann-benchmarks HDF5 `attrs` (algo, dataset,
   build_time, count, distance, run_count); MTEB (`mteb_version`, `dataset_revision`,
   `kg_co2_emissions`, `evaluation_time` in every JSON); VectorDBBench (`version`, `test_time`,
   `note` in every row); big-ann inherits ann-benchmarks' attrs.
6. **A dedicated rollup/export step turns many small raw files into one flat table, decoupled from
   raw storage.** `data_export.py` → CSV in both ann-benchmarks and big-ann-benchmarks (and cuVS
   bench, extended with a build/search label split); VectorDBBench's `leaderboard.json` is this
   rollup already, natively; MTEB's public leaderboard app reads across the whole results repo.
7. **Closed-loop timing by default** (issue the next query only after the previous one returns),
   **with an explicitly separate, differently-named mode for batched/concurrent throughput** rather
   than silently mixing the two. ann-benchmarks: per-query loop vs `--batch`; big-ann inherits this;
   VectorDBBench: `serial_runner` vs `concurrent_runner`.
8. **Repeat the timing measurement and report a robust statistic, not one wall-clock sample.**
   ann-benchmarks: best-of-`run_count`; big-ann inherits it; FAISS autotune runs until
   `min_test_duration` is satisfied; VectorDBBench's `concurrent_runner` runs for a fixed `duration`
   and reports percentiles over everything that completed.
9. **One standardized measurement protocol enforced across every compared entry, not trusted
   per-submitter.** big-ann: one reference Azure VM SKU + CI smoke test on every PR;
   ann-benchmarks: exactly one CPU pinned per container; VectorDBBench: identical case config
   applied to every DB; MTEB: identical task/metric code applied to every model, with revision
   pinning standing in for "same measurement conditions."
10. **Metric names are parseable, cutoff-qualified strings used directly as keys**, not bespoke
    per-framework abbreviations. `ir_measures`' grammar (`nDCG@10`, `AP(rel=2)@1000`) is the
    explicit, designed version of the same idea MTEB does informally (`ndcg_at_10`, `map_at_100`)
    and ann-benchmarks/big-ann do implicitly (a "knn" recall column parameterized by `count`).

## §4. Recommendations for `evaluation/`

Grounded against `docs/plans/evaluation-harness-v2.md` (the current design for the rewrite), which
already independently reaches several of the conventions in §3.

**Already aligned — no new work, just note the precedent:**
- The plan's "one JSONL record per cell, appended the moment it finishes, `--resume` = skip keys
  already present" (§3.2 of that plan) **is** convention #4 above, independently converged on; cite
  ann-benchmarks/big-ann/MTEB as validation that this is the right crash-recovery mechanism, not
  an unusual choice worth second-guessing. `[cheap]` — nothing to change.
- The plan's per-record `env` block (gpu, driver, cuda, torch, triton, commit, dirty, `sm_mhz`
  drift check) already matches convention #5 (MTEB's `mteb_version`/`dataset_revision`, VectorDBBench's
  `version`/`test_time`). `[cheap]`
- `config.py`'s "grid → explicit param combos (`itertools.product`)" already matches convention #2
  (ann-benchmarks `run_groups.args`, cuVS `groups.build`/`.search`). `[cheap]`
- The plan's `eager`/`graph` split as two labelled perf variants per cell already matches convention
  #7 (closed-loop default, explicit separately-named batch/replay mode). `[cheap]`
- Existing `"recall@100"`/`"ndcg@100"` quality keys already follow the `ir_measures` grammar
  (convention #10) rather than MTEB's `_at_` style — **keep enforcing this** for every new metric
  key `report.py` emits (e.g. never introduce `recall_at_100` alongside it). `[cheap]`

**Gaps or refinements worth adopting:**

1. **Adopt a coarser-than-per-cell process boundary (subprocess per `(dataset, dim, algo, backend)`,
   not per full cell) because every isolation-providing framework surveyed (ann-benchmarks,
   big-ann-benchmarks) puts a hard process/container wall around exactly the state that's cheapest
   to leak — compiled kernels, allocator arenas, CUDA graph pools — and `evaluation-harness-v2.md`
   itself already documents the failure mode this would fix: "`torch._dynamo.reset()` runs between
   sweeps only… so graph pools accumulate across the cells of a sweep," and separately, "the thesis
   methodology says one process per (dataset, dim, bs, filter, impl); the code isolates per
   (config, algo)." A subprocess boundary at the (dataset, dim, algo, backend) granularity is the
   ann-benchmarks idea (one process = one thing under test) sized for GPU context-init cost (~1–3 s
   × a few hundred boundaries, not × 700 cells). `[medium]`
2. **Make `report.py`'s first output one denormalized CSV/DataFrame from the JSONL+parquet, before
   any custom plotting code**, because every framework surveyed puts exactly one such step between
   raw storage and any chart (`data_export.py`→CSV in ann-benchmarks/big-ann/cuVS; VectorDBBench's
   `leaderboard.json` *is* this step, natively). This is convention #6; the plan already assigns
   this to `report.py` — just make the flat-table export the first thing it does, independent of
   whatever plots come after. `[cheap]`
3. **Give `report.py` a QPS-vs-recall Pareto plot and a "best recall at/above a QPS (or latency)
   threshold" table as its default view**, because this is the one visualization every benchmark in
   this survey converges on for comparing algorithms across operating points (ann-benchmarks/big-ann
   Pareto plots; big-ann's `eval/show_operating_points.py`; VectorDBBench's qps+recall+latency
   triple per case). All the needed fields (`qps`, `recall@k`, `p99_ms`) are already in the planned
   per-cell record — this is a `report.py` function, not new instrumentation. `[cheap]`
4. **Key the oracle/ground-truth cache in `oracle.py` by a content hash of
   (item embeddings, query attrs, filter definition, k_max)** rather than relying on process-local
   fingerprint memoization, because every framework surveyed treats ground truth as a durable,
   shippable artifact (ann-benchmarks' HDF5-embedded GT, big-ann's shipped `.bin` files, MTEB's
   pinned `dataset_revision`) that survives across machines and branches, not just across calls
   within one process. The plan already has a fingerprint cache in `oracle.py` (§2 item 6 of the
   plan) — extending the key to a stable content hash makes the cache portable the way a shipped
   `.bin` file is. `[medium]`
5. **Add a per-entry `disabled: true` kill switch to `suites.yaml`/`dataset.yaml` sweep entries**,
   because ann-benchmarks and cuVS both use exactly this (not commenting out YAML, not deleting the
   entry) to retire a flaky/slow config from a campaign while keeping its definition and its git
   blame. Useful once a `deep` sweep entry needs to be pulled from a nightly run without touching
   the sweep's shape. `[cheap]`
6. **Keep the plan's clock-drift warning (`sm_mhz`/`mem_mhz` recorded, warn at >5% drift from the
   first cell) as the right-sized version of big-ann's T3 power/cost tracking — do not build a
   power/$-per-query metric.** T3's IPMI sampling + $0.10/kWh cost model needs sensor access we
   don't have on a rented A100, and turning power into a *scored* leaderboard axis (as T3 does)
   invites exactly the gaming/measurement-integrity problems a reproducibility paper should avoid.
   This is an explicit **non-adoption**, and it is the right call, not a shortcut: `n/a`
7. **Do not adopt Docker-per-algorithm isolation.** ann-benchmarks/big-ann/cuVS need Docker because
   they each aggregate dozens of third-party libraries, authored by different teams, with
   conflicting native dependencies and no shared Python/CUDA version (this is *why* cuVS bench
   itself only splits GPU/CPU, not per-algorithm, once all algorithms share one dependency stack —
   see §2). We have five algorithms in one repo behind one `uv.lock`. The isolation problem Docker
   solves for them (dependency conflicts across strangers' code) does not exist here; recommendation
   1's subprocess boundary gets the *state-leakage* benefit of their isolation without paying for a
   dependency problem we don't have. This is a correct existing non-adoption, not a mistake: `n/a`
8. **Do not adopt MTEB's results-as-a-separate-repo split.** MTEB needs it because results are
   contributed by thousands of external parties whose submissions must not touch benchmark code
   review. CLAUDE.md already places `evaluation/results/<name>.json` + `.yaml` + `.perkernel/`
   inside the workspace, which is correct for a single-team contributor model — splitting it out
   would only add friction here. `n/a`
9. **Do not build a true open-loop (independent-arrival / queueing) load generator for query QPS.**
   Of every framework surveyed, only VectorDBBench's `rate_runner` is open-loop, and only for
   *insert* throughput — every query-latency/QPS measurement in this entire survey (ann-benchmarks,
   big-ann, cuVS, VectorDBBench's `serial_runner`/`concurrent_runner`) is closed-loop. The plan's
   own choice ("closed-loop single client — the in-process analogue of the papers' client-side
   QPS") matches unanimous community practice; flagging this here so it is not mistaken for a
   shortcut later — building a proper open-loop generator (arrival-process modeling, queueing-delay
   accounting) would be `[expensive]` for a benefit no surveyed framework thought was worth it.

## §5. Could not verify in this session

- **cuVS bench's `--dry-run` flag and its `plot.py`/`data_export.py` output format** are known only
  from a search-engine summary of the docs page and CHANGELOG entries; `docs.rapids.ai`'s Sphinx
  page 404'd on every direct fetch attempt (tried `stable` and pinned-branch URLs) and the
  corresponding README under `python/cuvs_bench/` in the `rapidsai/cuvs` repo wasn't found at the
  paths tried. The one thing directly confirmed from source is the algorithm YAML format
  (`config/algos/hnswlib.yaml`, quoted in full in §2).
- **The exact ground-truth *generation* method for YFCC-10M filtered search** (brute-force over the
  tag-matching subset vs. some approximation) — only the shipped filename and the `datasets.py`
  field layout were confirmed; the generation script that produced `yfcc100m_query_gt100.bin` was
  not located in this session.
- **VectorDBBench's exact composite leaderboard scoring formula** ("(metric/base)×100", the
  half-QPS/double-latency failure penalty, geometric-mean aggregation) came from an AI-generated
  summary of the README's prose, not from reading the scoring module's source directly — treat as
  paraphrase pending a source check.
- **FAISS `benchs/`'s exact autotune algorithm** (`ParameterSpace`, `OperatingPoints`,
  `min_test_duration`, `n_experiments`) came from a search-engine summary of wiki benchmark logs,
  not from reading `AutoTune.cpp`/`contrib/evaluation.py` directly in this session.
- **The streaming track's actual runbook YAML content and result CSV column layout**
  (`res_final_runbook_AzureD8lds_v5.csv`) — the operation semantics (search/insert/delete/replace,
  ~4:4:1 ratio) came from `neurips23/README.md` prose; the file itself was located but not opened.
- **MTEB's stated design rationale** for the separate results repo is inferred from "it was no
  longer possible [to submit via HF model-card metadata]" in the results repo's README; no explicit
  design-rationale document was found confirming *why* (review-load separation vs. provenance
  integrity vs. something else).
