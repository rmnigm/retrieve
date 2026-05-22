<!-- claude code generated file -->

# `evaluation/` retrieval harness

Live reference for the in-process retrieval benchmark used to compare
algorithms across yambda, goodreads, and arxiv. Covers the driver layout, the
YAML config format, the measurement methodology, the output schema, and the
extension points for adding a new algorithm, config, or dataset.

For the algorithm internals (what each `forward(query)` does) see
[kernels.md](kernels.md) and [architecture.md](architecture.md). For the
training pipeline that produces the SASRec checkpoints consumed here, see
[checkpoints.md](checkpoints.md). The correctness-only test suite that
gates kernel changes is documented in [testing.md](testing.md).

## Scope

[`evaluation/retrieval/evaluate.py`](../../evaluation/retrieval/evaluate.py)
is the **single-algo** retrieval benchmark CLI — GPU algos × backends ×
filter kinds across yambda, goodreads, and arxiv. One invocation runs
exactly one algorithm's cells and writes one JSON file. The shell
wrapper [`run_per_algo.sh`](../../evaluation/run_per_algo.sh) reads the
YAML's ``algorithms`` list and loops, writing per-algo JSONs into
``cfg.output`` (now a **directory**, not a file). Downstream analysis
reads them back via
[`retrieval.results_io.load_results`](../../evaluation/retrieval/results_io.py).

Per-algo process isolation is the point: every algo gets a fresh Python
interpreter, which wipes torch.compile / Triton autotune / CUDA-graph
private-pool state that ``_release_algo`` and ``torch._dynamo.reset()``
can't reclaim between algos in the same process. The SASRec encode
pass is cached on disk by
[`queries_cache.py`](../../evaluation/retrieval/queries_cache.py)
(``<ckpt-dir>/encoded_queries_<split>.pt``, keyed on ckpt mtime and
``max_seq_length``), so the N subprocesses don't pay N × encode cost.

The CPU quality baseline ([voyager HNSW](../../evaluation/retrieval/voyager/))
lives in its own subpackage with its own CLI
([`evaluate-voyager`](#cpu-quality-baseline-evaluate-voyager)) and is
NOT subject to the per-algo loop — it writes its own single JSON.

It dispatches by config shape across three datasets:

| Config shape                              | Mode                                   |
|-------------------------------------------|----------------------------------------|
| `checkpoint` set, `query_emb_path` unset  | Encode queries via SASRec (yambda/goodreads) |
| `query_emb_path` set, `checkpoint` unset  | Load pre-encoded text embeddings (arxiv) |
| `filters: null`                           | No filter loop (yambda)                |
| `filters: {none, clause, bloom}`          | Filter sweeps (goodreads, arxiv)       |

For each `(filter_kind, sweep, algo, params, K, batch_size)` cell:

1. Builds the index on top of the dataset's item embeddings.
2. Streams every query through the index in chunks (default
   `QUALITY_BATCH_SIZE = 64`) to compute recall@K and ndcg@K (the
   **quality pass**) — vs the held-out test target on `filter_kind=none`,
   vs the cached filtered-FullScan top-K oracle on filtered cells.
3. Times the index forward at the configured batch size against a
   fixed-seed pool of cached queries (the **perf pass**, default
   `n_pool = 4096`), capturing median / p20 / p80 latency plus peak /
   transient GPU memory.
4. Writes one row per cell to the configured output JSON.

The harness reports numbers and never fails a build; correctness lives
in [`retrieve/tests/`](../../retrieve/tests/) and gates CI separately.

## Files

```
evaluation/
├── run_per_algo.sh            # shell wrapper: loops algos from cfg, one subprocess each
evaluation/retrieval/
├── evaluate.py                # single-algo CLI: --config X --algo Y --output <file>
├── queries_cache.py           # disk cache for the SASRec encode pass
├── results_io.py              # load_results(dir) — concatenate per-algo JSONs
├── algos/                     # one class per GPU algo, duck-typed protocol
│   ├── __init__.py            # build_algorithm + ALGORITHMS + build_filter
│   ├── _helpers.py            # collect_modules
│   ├── filter.py              # build_filter (None/clause/bloom) + make_mask
│   ├── linr_v1.py             # LinrV1Algo — covers triton_knn + linr_v1_filter_mask
│   ├── linr_v2.py             # LinrV2Algo — exact filtered top-K via PrefilterKNN
│   ├── linr_v3.py             # LinrV3Algo — V3 → V2 cascade (1-bit prefilter, fp32 rerank)
│   ├── linr_v4.py             # LinrV4Algo — single-stage int8 dense (Int8SimilarityMasking)
│   ├── silvertorch.py         # SilvertorchAlgo — IVF + INT8 + (none / bloom / exact) filter
│   └── torch_knn.py           # TorchKnnAlgo — FullScanKNN reference
├── voyager/                   # CPU quality baseline, separate CLI
│   ├── baseline.py            # VoyagerHNSW (voyager.Index wrapper)
│   └── evaluate.py            # `evaluate-voyager` driver
├── bench_tools.py             # encode_queries, quality_pass_cached, perf_pass_cached
├── config.py                  # EvalConfig dataclass + yaml loader
├── hf_io.py                   # HF download / upload helpers + console scripts
└── metrics.py                 # accumulate_metrics, finalize_metrics

evaluation/conf/
├── 500m/d{64,128,256}-quality.yaml                # 500M Listen+, SASRec checkpoints
├── 5b/d{64,128}-quality.yaml                      # 5B Listen+, SASRec checkpoints
├── arxiv/d{64,128,256}-{quality,filter}.yaml      # arxiv, nomic-embed-text-v1.5
├── goodreads/d{64,128,256}-{quality,filter}.yaml  # goodreads filter+quality bench
└── voyager/{yambda-500m,goodreads,arxiv}-d{64,128,256}.yaml  # voyager-only CPU baseline
```

## Configuration

Configs are YAML, parsed with `yaml.safe_load` and slammed into the
[`EvalConfig`](../../evaluation/retrieval/config.py) dataclass.
Each file declares a `_defaults: &defaults` anchor block, then merges that
block at the top level with `<<: *defaults` and adds dataset-specific
fields (`checkpoint:` for SASRec datasets, `query_emb_path:` for arxiv).
The loader strips keys starting with `_` so the anchor scratch key doesn't
break the dataclass constructor.

Example yambda config:

```yaml
_defaults: &defaults
  data_dir: data/yambda-500m
  output: null                # null → <ckpt-dir>/evaluate.json
  split: test
  device: cuda
  ks: [100, 200, 400]
  batch_sizes: [1, 4, 8, 16]
  seed: 0
  encode:
    batch_size: 512
    num_workers: 8
    max_seq_length: 200
  algorithms:
    - torch_knn
    - triton_knn
    - linr_v3
    - silvertorch
  algo_params:
    linr_v3:
      - {candidate_pool: 5000, v3_seed: 0}
      - {candidate_pool: 10000, v3_seed: 0}
    silvertorch:
      - {n_lists: 1024, n_probe: 32, n_iter: 10, seed: 0}
      - {n_lists: 2048, n_probe: 64, n_iter: 10, seed: 0}

<<: *defaults
checkpoint: data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/best_model.pt
```

`algo_params[algo]` is a **list of dicts** — each dict is one explicit
combo, taken as-is. Algos with no entry iterate over `[{}]` so the
caller's loop stays uniform. The cross-product is only over `(algos,
combos, ks, batch_sizes)`.

Filter-bench configs (`goodreads-*.yaml`, `arxiv-*.yaml`) add a `filters:`
block declaring `clause` and `bloom` filter kinds, each with attribute
paths and named sweeps; see [filtering.md](filtering.md) for the
filter API.

### Field reference

| Field | Type | Purpose |
|---|---|---|
| `checkpoint` | path | SASRec `best_model.pt` path. Required for yambda/goodreads, omit for arxiv. The trainer writes a sibling `config.json` (via `GSASRecConfig.save`) which the loader reads to pick up `embedding_dim`, `dropout`, `reuse_item_embeddings`, etc. — no hyperparams in the eval YAML. Legacy 500M ckpts that don't ship a `config.json` fall back to `D128_DROP05_DEFAULTS` in `bench_tools.py`. |
| `query_emb_path` | path or `null` | Pre-encoded query tensor for arxiv. Defaults to `<data_dir>/<content_subdir>/query_emb.pt` when unset. |
| `data_dir` | path | Holds `item_id_map.json`, `<split>.parquet`, optional `eval_split.parquet` + filter attrs. |
| `content_subdir` | str | Subdir under `data_dir` for `{text_emb,query_emb}.pt` + meta sidecars (arxiv only). Defaults to `content`. Arxiv ships `content_d64`, `content_d128`, `content` (= d=256). |
| `gt_subdir` | str | Subdir under `data_dir` for `gt_topk_<sweep>.pt` oracle caches. Default `gt`. Must vary with `content_subdir` (oracle scores depend on `item_embs` which depend on dim) — set per-yaml when running multiple dims off the same `data_dir`. |
| `output` | dir path | Directory that holds the per-algo JSONs written by `evaluate`. `run_per_algo.sh` reads it and creates `<output>/<algo>.json` per algo. The voyager CLI still writes a single JSON file at its `output` path (see [voyager configs](#cpu-quality-baseline-evaluate-voyager)). |
| `split` | str | `test` (default) or `val`. |
| `device` | str | `cuda` (only meaningful value today). |
| `ks` | list[int] | K-cutoffs to evaluate. Each emits `recall@K`, `ndcg@K` on its own row. |
| `batch_sizes` | list[int] | Perf-pass batch sizes. Quality is invariant to bs and computed once per `(algo, params, k)`; emitted on every bs row. |
| `seed` | int | Drives `torch.manual_seed`, `torch.cuda.manual_seed_all`, the perf-query-pool generator, and any algo seeds that read from `algo_params`. |
| `encode.batch_size` | int | Forward-pass batch size for the SASRec query encode. Independent of perf bs. |
| `encode.num_workers` | int | DataLoader workers for the encode pass. |
| `encode.max_seq_length` | int | History truncation length; must match the trained checkpoint. |
| `algorithms` | list[str] | Subset of [`ALGORITHMS`](../../evaluation/retrieval/algos/__init__.py). |
| `algo_params` | dict[str, list[dict]] | Per-algo parameter combos. Each dict is one explicit combo; no implicit cross-product. |
| `filters` | dict or `null` | Optional filter-bench block. See [filtering.md](filtering.md). |
| `users_limit` | int or `null` | Optional cap on users for ALL cells (quality and filter alike). Goodreads has 313k test users which makes the bs=1 quality stream the wall-clock bottleneck; cap to e.g. 50000 to speed runs up. Leave `null` to use the full split. |

Two CUDA settings the driver flips at startup, in addition to seeds:

- `torch.backends.cuda.matmul.allow_tf32 = False` and
  `torch.backends.cudnn.allow_tf32 = False` — keeps the oracle (cuBLAS
  `q @ E_t`) and the algos in the same precision so exact-mask paths
  (`linr_v1_filter_mask`) don't show ~1e-3 recall drift on narrow
  filters where top-K boundaries land within TF32's 10-bit mantissa
  band.

### CLI overrides

```
uv run evaluate \
    --config conf/<name>.yaml \
    --algo <name> \
    --output <output-dir>/<algo>.json \
    [--filter-kind {none|clause|bloom} ...] \
    [--sweep <sweep_name>] \
    [--skip-quality]
```

`--algo` is required and runs a single algorithm — the wrapper
[`run_per_algo.sh`](../../evaluation/run_per_algo.sh) is what loops
over the YAML's `algorithms` list. `--filter-kind` is repeatable and
narrows to those kinds; empty = all. `--sweep` narrows to a single
sweep — useful for iterating on one cell without re-encoding queries.
`--skip-quality` drops the quality stream (`recall@K` / `ndcg@K` →
NaN); perf rows still emit. Useful for fast latency/memory sweeps.

To run the whole `algorithms` list:

```
./run_per_algo.sh conf/<name>.yaml [extra flags forwarded to evaluate]
```

## Algorithms

Seven algorithm names are registered (one duplicate alias). The classes
live in [`evaluation/retrieval/algos/`](../../evaluation/retrieval/algos/),
one per file, all duck-typed:

```
algo.modules: list[nn.Module]                 # for memory cleanup
algo.is_cpu: bool                             # CPU baseline marker
algo.forward(q, qa_narrow=None) -> (ids, scores)
```

`build_algorithm(name, item_embs, k, *, filter_kind, filter_mod,
item_attrs_narrow, params)` is a single factory for all cells —
filtered or not. Algos that can't run on the requested `filter_kind`
raise `ValueError` at construction; the driver catches and skips that
cell.

| Name | Class / file | Notes |
|---|---|---|
| `torch_knn` | [`TorchKnnAlgo`](../../evaluation/retrieval/algos/torch_knn.py) | Reference exhaustive IP scan via `FullScanKNN`; mask post-filter on filtered cells (skipped on the goodreads filter suite — equals the oracle). |
| `triton_knn` | [`LinrV1Algo`](../../evaluation/retrieval/algos/linr_v1.py) | Pure-torch full-scan KNN — uses `SimilarityMasking(backend="triton")`, but the original Triton kernel was removed, so both backends run the same `query @ x.T + topk` path. Yambda-config alias. |
| `linr_v1_filter_mask` | [`LinrV1Algo`](../../evaluation/retrieval/algos/linr_v1.py) | Same class as `triton_knn`; canonical name on filter cells. |
| `linr_v3` | [`LinrV3Algo`](../../evaluation/retrieval/algos/linr_v3.py) | V3 → V2 cascade: `OneBitKNN(backend="triton")` produces top-`candidate_pool` at 1-bit precision; `PrefilterKNN(backend="triton")` rescores at fp32. Approximate. |
| `linr_v2` | [`LinrV2Algo`](../../evaluation/retrieval/algos/linr_v2.py) | Exact filtered top-K — candidate set IS the filter (`filter_mod.evaluate_indices`). Recall=1.0 by construction; headline is speed/memory. Filter cells only — raises `ValueError` on `filter_kind="none"`. |
| `linr_v4` | [`LinrV4Algo`](../../evaluation/retrieval/algos/linr_v4.py) | Single-stage int8 dense + optional mask + topk via `Int8SimilarityMasking`. `torch._int_mm` (int8×int8 → int32, IMMA on Ampere+) directly to `torch.topk` with no scale recovery (global per-tensor scale is rank-preserving). Twin of `triton_knn`/`linr_v1_filter_mask` at int8 storage; ≥0.99 recall on unit-norm embeddings at D=128. |
| `silvertorch` | [`SilvertorchAlgo`](../../evaluation/retrieval/algos/silvertorch.py) | IVF + INT8 ANN with `SilverTorch.filter` selected per `filter_kind`: `"bloom"` → bloom-fused IVF (codesigned `codesigned_probe_score`, item signatures over narrow attrs baked in at register time); `"clause"` → exact-AND-of-OR-fused IVF (codesigned `codesigned_probe_score_exact`, narrow attrs baked in); `"none"` → plain IVF + INT8. All three modes routed through the same algo class. |

The CPU quality baseline (`voyager_hnsw`, Spotify HNSW via `voyager.Index`)
is **not** in this registry — it lives in its own subpackage with its own
CLI; see [CPU quality baseline](#cpu-quality-baseline-evaluate-voyager) below.

`make_mask(filter_mod, qa_narrow)` (in [`algos/filter.py`](../../evaluation/retrieval/algos/filter.py))
is the one-line helper most algos call: it routes per-batch query attrs
through `filter_mod.evaluate_mask`, then forces the padding-row sentinel
`mask[:, 0] = False` so reverse clauses don't admit the padding item the
oracle deliberately excludes.

## Measurement methodology

### Quality pass

[`quality_pass_cached`](../../evaluation/retrieval/bench_tools.py)
streams every cached query through the index in chunks of
`QUALITY_BATCH_SIZE = 64` and accumulates per-row recall@K / ndcg@K via
[`accumulate_metrics`](../../evaluation/retrieval/metrics.py). Quality is
invariant to perf batch size (same per-row scoring math), so we run this
**once per `(algo, params, k)` cell** and attach the result to every `bs`
row of the same cell — and the larger chunk amortises CUDA-launch +
Python overhead vs the bs=1 stream that would otherwise dominate
goodreads' 313k test pass.

For filtered cells the targets passed in are the cached filtered-FullScan
top-K oracle (`<data_dir>/<gt_subdir>/gt_topk_<sweep>.pt`); for
`filter_kind=none` and yambda they are the held-out test items.

The oracle is built by
[`compute_filtered_oracle`](../../evaluation/retrieval/evaluate.py) on
first run: brute-force `q @ E_t` with the *exact* filter mask
(ExactAttributeFilter even on bloom cells — bloom's false positives must
not leak into ground truth), then `topk` with the padding-row score
forced to `-inf` and the remaining `-inf` ties mapped back to `-1` so
short-fill rows don't score against the algos' `-1` sentinel. The
oracle is cached at `<data_dir>/<gt_subdir>/gt_topk_<sweep>.pt`; stale
shapes are detected and recomputed.

### Perf pass

The perf pass measures the index forward in isolation, on the device the
algorithm lives on. It is split between two primitives:

- **`measure_forward_cuda`** (the GPU path): runs `WARMUP_ITERS=20`
  calls to settle Triton autotune and JIT, then takes a clean
  `torch.cuda.max_memory_allocated()` window over `mem_reps=5` calls
  (without `do_bench`'s 256 MiB L2-buster polluting peak), then times via
  `triton.testing.do_bench(rep=200ms, warmup=50)`.
  Returns `(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)`.
- **`measure_forward_cpu`** (the CPU path; no current caller in the main
  driver — the voyager baseline runs through its own quality-only CLI
  which doesn't need perf timing): 3 warmup calls, then a
  `time.perf_counter` loop within a 200 ms budget. Returns the same
  tuple shape with peak/transient = 0.

### Multi-query pool — why p20/p80 are over queries

The perf pass times against a **fixed-seed pool of `n_pool=4096` query
batches**, round-robin'd into the timed function. This matters for
IVF-style algorithms (`silvertorch`) and graph-walk ANN (`voyager_hnsw`)
where a single fixed query collapses p20/p80 to one cluster's traversal
cost — degenerate percentiles. With the pool, p20/p80 reflect query
diversity (the intended workload variance), not CUDA scheduling jitter.

The pool itself is built per `(algo, params, k, bs)` cell:

```python
g = torch.Generator(device="cpu").manual_seed(seed)
rows = torch.randint(0, n_users, (n_pool, batch_size), generator=g)
pool = queries[rows.reshape(-1)].reshape(n_pool, batch_size, -1).to(device)
```

so the *same query indices* are sampled across reruns with the same seed.
On filter cells, `qa_narrow` is sampled from the same row indices so the
filter shape matches the queries.

### Memory snapshot ordering

Reading allocator state immediately around `build_algorithm(...)` is
brittle: residue from the previous cell's k-means / topk transients can
inflate `index_mem`. Per-cell ordering is therefore:

```
sync → empty_cache → reset_peak_memory_stats → mem_before
build_algorithm
sync → index_mem = allocated() - mem_before
quality_pass
for bs in batch_sizes: perf_pass
algo_obj.modules.clear(); del algo_obj; empty_cache
```

`modules.clear()` is required — `for m in modules: del m` only drops the
loop variable, not the list's references, leaking the index into the
next cell's `mem_before`.

### Determinism

`main()` calls `torch.manual_seed(cfg.seed)` and
`torch.cuda.manual_seed_all(cfg.seed)` before any allocation. The
perf-query-pool generator uses the same seed. Algo seeds are wired
through `algo_params` (e.g. `silvertorch.seed`, `linr_v3.v3_seed`).
TF32 is disabled at startup so the oracle and the algos compute scores
at the same precision.

Quality columns must be **byte-identical** across reruns with the same
seed. Latency may drift within ~5% due to clock noise, NVML thermal
state, and (for Triton autotune) JIT cache state.

## Output schema

One row per `(filter_kind, sweep, algo, params, k, bs)` cell. Fields:

| Field | Type | Notes |
|---|---|---|
| `suite` | str | `yambda` (no filters) or `filter`. |
| `cell` | str | `<filter_kind>_<sweep>_bs<N>_k<K>` for cross-row joins. |
| `filter_kind` | str | `none`, `clause`, or `bloom`. |
| `sweep` | str | Filter sweep name from the config (e.g. `c0_genre`, `c0_maincat`). |
| `impl` | str | Algorithm name. |
| `device` | str | `"cuda"` or `"cpu"`. |
| `seed` | int | The `cfg.seed` that produced this row. |
| `batch_size` | int | First-class column. |
| `k` | int | Same. |
| `n_users_kept` | int | Number of users this sweep evaluated (skip mask drops users with no surviving narrow clauses). |
| `median_ms`, `p20_ms`, `p80_ms` | float | Latency over the multi-query pool. |
| `peak_mem_mib` | float | `max_memory_allocated()` over the perf window. 0 for CPU rows. |
| `index_mem_mib` | float | `allocated()` delta around `build_algorithm`. 0 for CPU rows. |
| `fwd_scratch_mib` | float | Peak − baseline within the forward call. 0 for CPU rows. |
| `recall@<k>`, `ndcg@<k>` | float | Quality, identical across all bs rows of the same `(algo, params, k)` cell. NaN when `--skip-quality`. |
| `extra.params` | dict[str, str] | Per-algo `algo_params` for traceability. |

Algos that raise `ValueError` at construction (e.g. `linr_v2` with
`filter_kind="none"`, `silvertorch` with `filter_kind="clause"`) silently
skip their cells — no stub row is emitted.

Downstream analysis: filter by `device` to compare GPU rows against each
other separately from CPU baselines; group by `(impl, k)` and span `bs`
to read scaling behavior; group by `(filter_kind, sweep)` to compare
filter shapes; group by `extra.params` to read parameter sweeps for one
algo.

## How to run

The repo is a uv workspace ([root pyproject](../../pyproject.toml));
`evaluation/` shares a single `.venv` with `retrieve/` at the workspace
root. `uv run` from inside `evaluation/` discovers the workspace root
automatically; equivalently use `uv run --directory evaluation …` from
the root.

Yambda 500M sweep:

```bash
cd evaluation
./run_per_algo.sh conf/500m/d128-quality.yaml
```

5B sweep (gated on the 5B checkpoint having been trained):

```bash
./run_per_algo.sh conf/5b/d128-quality.yaml
```

Goodreads filter bench:

```bash
./run_per_algo.sh conf/goodreads/d128-filter.yaml
```

Arxiv filter bench:

```bash
./run_per_algo.sh conf/arxiv/d256-filter.yaml
```

CPU quality baseline (voyager HNSW) — separate CLI, see
[CPU quality baseline](#cpu-quality-baseline-evaluate-voyager) below:

```bash
uv run evaluate-voyager --config conf/voyager/yambda-500m-d128.yaml
```

One filter cell only (faster iteration):

```bash
uv run evaluate \
    --config conf/goodreads/d128-filter.yaml \
    --algo linr_v3 \
    --output results/goodreads/d128-filter/linr_v3.json \
    --filter-kind bloom --sweep c0_genre
```

### Sanity checks to run after a sweep

1. GPU `median_ms(bs=8)` < `8 × median_ms(bs=1)` (sub-linear scaling —
   the proof point of the GPU implementations).
2. `recall@K` and `ndcg@K` columns identical across the bs rows of the
   same `(algo, params, k)` cell.
3. Two reruns with the same seed: quality columns byte-identical;
   latency columns within ~5%.
4. Cross-check against the voyager CPU baseline:
   `evaluate-voyager`'s `recall@K` should land within ~2 pp of
   `silvertorch` at comparable approximation settings (`m`/`ef_query`
   vs `n_lists`/`n_probe`).

## CPU quality baseline: `evaluate-voyager`

Voyager HNSW is the only CPU baseline in the bench. It runs in its own
subpackage with its own CLI because folding it into the unified GPU
driver bought nothing: it shares no kernels, has its own timing path,
doesn't participate in filter sweeps, and forced every cell to carry
an `is_cpu` branch.

```bash
uv run evaluate-voyager --config conf/voyager/yambda-500m-d128.yaml
uv run evaluate-voyager --config conf/voyager/goodreads-d128.yaml
uv run evaluate-voyager --config conf/voyager/arxiv-d128.yaml
```

The CLI reuses the `EvalConfig` schema and the same
`load_item_and_queries` / `quality_pass_cached` helpers as the main
driver — only the sweep machinery is bespoke. Configs declare a single
`voyager_hnsw` entry under `algo_params:`; the script iterates the
cross-product of `algo_params.voyager_hnsw × ks` and emits one row per
combo. No `algorithms:` / `backends:` / `batch_sizes:` / `filters:`
needed.

Example config — yambda 500M, d=128:

```yaml
data_dir: data/yambda-500m
output: results/voyager/yambda-500m-d128.json
split: test
device: cuda                   # for SASRec query encoding; the HNSW itself is CPU
ks: [100, 200, 400]
seed: 0
encode:
  batch_size: 512
  num_workers: 8
  max_seq_length: 200
algo_params:
  voyager_hnsw:
    - {m: 32, ef_construction: 200, ef_query: 1000, num_threads: 16, seed: 0}
checkpoint: data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/best_model.pt
```

Output schema is slim (no `batch_size`, `median_ms`, `peak_mem_mib`,
`backend`, `filter_kind`, `sweep` — none apply to a quality-only CPU
sweep). Each row has `suite: "voyager"`, `impl: "voyager_hnsw"`,
`device: "cpu"`, `seed`, `k`, `n_users`, `recall@<k>`, `ndcg@<k>`, and
`extra.params`. Plotting code that unions the voyager JSON with the
main `evaluate.json` should treat missing fields as not-applicable.

Configs ship for yambda 500m × {d64, d128, d256}, goodreads × {d64,
d128, d256}, arxiv × {d64, d128, d256} — nine total. 5B is omitted
because HNSW build is prohibitively long at that scale; the 5B configs
in `conf/5b/` likewise never listed voyager. Filter sweeps are not
covered: the filter-aware `VoyagerHNSWAlgo` wrapper from before this
refactor would post-filter via mask gather, but the user-facing point
of voyager in this bench is the unfiltered-quality baseline.

## Extending

### Add a new algorithm

1. Add a new file `evaluation/retrieval/algos/<name>.py` with an
   `nn.Module` subclass exposing `algo_modules`, `is_cpu`, and
   `forward(q, qa_narrow=None) -> (ids, scores)`. Wrap a
   `RetrievalModule` from `retrieve` (see
   [interfaces.py:RetrievalModule](../../retrieve/src/retrieve/interfaces.py))
   or roll your own. Use `collect_modules(*base, filter_mod=...)` from
   [`algos/_helpers.py`](../../evaluation/retrieval/algos/_helpers.py)
   to build `self.algo_modules` (single source for the
   optional-filter-append rule the driver iterates for memory cleanup),
   then call `self.compile(dynamic=True, mode="reduce-overhead")` at
   the end of `__init__` so the whole forward (filter + index + any
   cascade stages) gets compiled into one cudagraph capture — applied
   the same way on both retrieval backends.
2. Import it from
   [`algos/__init__.py`](../../evaluation/retrieval/algos/__init__.py),
   add the name to `ALGORITHMS`, and add a branch in `build_algorithm`
   that unpacks the parameters you need (raise `ValueError` for
   incompatible `filter_kind`s; the driver catches and skips). If the
   class threads `backend` through to its underlying `RetrievalModule`,
   also add the name to `BACKEND_CAPABLE_ALGOS` so the driver fans
   out per-backend rows.
3. Add it to the `algorithms:` list in any config that should sweep it,
   plus an `algo_params` entry if it takes knobs.
4. No driver changes needed — the perf primitive is selected by
   `is_cpu`, and the schema columns are written uniformly.

### Add a new config (new checkpoint)

Copy an existing YAML, update `checkpoint:` (or `query_emb_path:`) and
any catalog-size-driven knobs (`silvertorch.n_lists/n_probe`). For a
matching voyager CPU baseline, also drop a YAML into `conf/voyager/`
(set `voyager_hnsw.m/ef_construction/ef_query`). The model loader reads
hyperparams from `<ckpt-dir>/config.json`, so the YAML never carries
`embedding_dim` etc. (legacy 500M checkpoints without `config.json`
get the `D128_DROP05_DEFAULTS` fallback in
[`bench_tools.py`](../../evaluation/retrieval/bench_tools.py)).

### Add a new dataset

The driver dispatches on config shape, not a `dataset:` field, so adding
a new dataset is a question of producing the on-disk artifacts the
driver consumes. The dataset CLIs live in [`evaluation/data/`](../../evaluation/data/)
and produce, for the yambda layout:

```
data/<dataset>/
├── item_id_map.json
├── train.parquet
├── val.parquet
└── test.parquet
```

For the arxiv layout (no SASRec, pre-encoded text):

```
data/<dataset>/
├── item_id_map.json
├── papers.parquet
├── heldout.parquet
├── content/                   # default; varies via cfg.content_subdir
│   ├── text_emb.pt            # item-side, "search_document: " prefix
│   ├── text_emb.meta.json
│   ├── query_emb.pt           # query-side, "search_query: " prefix
│   └── query_emb.meta.json
└── eval_split.parquet         # optional — only needed for filter sweeps
```

For filter sweeps either layout adds:

```
data/<dataset>/
├── item_attrs_narrow.pt
├── item_attrs_wide.pt           # currently unused; kept for future wide bloom sweeps
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
├── ... (per-clause vocab JSONs)
└── <gt_subdir>/                 # auto-built by the driver, oracle cache
```

The CLIs that build all of the above are exposed as console scripts:
`uv run yambda prep ...`, `uv run arxiv all ...`, `uv run goodreads all ...`.

### HuggingFace I/O

[`hf_io.py`](../../evaluation/data/hf_io.py) is the single source
of truth for HF reads / writes. Two entry points:

- `EVAL_REPOS` maps each dataset to a `pinkmeme/eval-<dataset>` HF
  dataset repo holding eval inputs and (under `checkpoints/<ckpt-id>/`)
  model checkpoints.
- `RAW_REPOS` maps each upstream raw source (yambda, arxiv) to its
  upstream HF repo and `repo_type`, downloaded into `data/_raw/<source>/`.

Console scripts (registered in
[`evaluation/pyproject.toml`](../../evaluation/pyproject.toml)):

```bash
uv run eval-fetch yambda-500m            # pull eval inputs to data/yambda-500m/
uv run eval-fetch arxiv-papers --dims d64,d128 --include-checkpoints
uv run eval-publish goodreads-work-id --dry-run
uv run eval-publish-checkpoint yambda-500m gsasrec-d128-drop0.5 --dry-run
```

The local data root resolves to `evaluation/data/` by default; override
with `RETRIEVE_DATA_ROOT=/some/path`.

## See also

- [architecture.md](architecture.md) — package layout and the
  `RetrievalModule` / `FilterModule` contracts.
- [kernels.md](kernels.md) — Triton kernel internals for V1/V2/V3,
  IVF-INT8, IVF-FP32, bloom-match.
- [filtering.md](filtering.md) — filter API and the with-filters story.
- [checkpoints.md](checkpoints.md) — the trainer pipeline that produces
  the checkpoints consumed here.
- [testing.md](testing.md) — the correctness suite (separate from this
  perf harness).
