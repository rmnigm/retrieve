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

`evaluation/retrieval/evaluate.py` is the **single** retrieval benchmark
driver. It dispatches by config shape across three datasets:

| Config shape                              | Mode                                   |
|-------------------------------------------|----------------------------------------|
| `checkpoint` set, `query_emb_path` unset  | Encode queries via SASRec (yambda/gr)  |
| `query_emb_path` set, `checkpoint` unset  | Load pre-encoded text embeddings (arxiv) |
| `filters: null`                           | No filter loop (yambda)                |
| `filters: {none, clause, bloom, combined}`| Filter sweeps (goodreads, arxiv)       |

For each `(filter_kind, sweep, algo, K, batch_size)` cell:

1. Builds the index on top of the dataset's item embeddings.
2. Streams every query through the index at bs=1 to compute recall@K and
   ndcg@K (the **quality pass**) — vs the held-out test target on
   `filter_kind=none`, vs the cached filtered-FullScan top-K oracle on
   filtered cells.
3. Times the index forward at the configured batch size against a
   fixed-seed pool of cached queries (the **perf pass**), capturing
   median / p20 / p80 latency plus peak / transient GPU memory.
4. Writes one row per cell to the configured output JSON.

The harness reports numbers and never fails a build; correctness lives
in [`retrieve/tests/`](../../retrieve/tests/) and gates CI separately.

## Files

```
evaluation/retrieval/
├── evaluate.py                # driver (main): config → loop → JSON
├── algo_registry.py           # build_algorithm + build_filtered_algorithm + ALGORITHMS
├── bench_primitives.py        # encode_queries, quality_pass_cached, perf_pass_cached
├── config.py                  # EvalConfig dataclass + yaml loader
├── voyager_baseline.py        # VoyagerHNSW (CPU)
└── metrics.py                 # accumulate_metrics, finalize_metrics

evaluation/conf/
├── 500m-d64.yaml              # 500M Listen+, d=64 SASRec checkpoint
├── 500m-d128.yaml             # 500M Listen+, d=128 SASRec checkpoint
├── 500m-d256.yaml             # 500M Listen+, d=256 SASRec checkpoint
├── 5b-d64.yaml                # 5B Listen+, d=64 SASRec checkpoint
├── 5b-d128.yaml               # 5B Listen+, d=128 SASRec checkpoint
├── arxiv-d256.yaml            # arxiv filter bench, nomic-embed-text-v1.5 d=256
└── goodreads-d{64,128,256}-drop0.5-id.yaml  # goodreads filter bench
```

## Configuration

Configs are YAML, parsed with `yaml.safe_load` and slammed into the
`EvalConfig` dataclass at [config.py](../../evaluation/retrieval/config.py).
Each file declares a `_defaults: &defaults` anchor block, then merges that
block at the top level with `<<: *defaults` and adds dataset-specific
fields (`checkpoint:` for SASRec datasets, `query_emb_path:` for arxiv).
The loader strips keys starting with `_` so the anchor scratch key doesn't
break the dataclass constructor.

Example yambda config:

```yaml
_defaults: &defaults
  data_dir: data/yambda/500m-listens
  output: null                  # null → <ckpt-dir>/evaluate.json
  split: test
  device: cuda
  ks: [100, 500]
  batch_sizes: [1, 8, 16]
  seed: 0
  encode:
    batch_size: 512
    num_workers: 8
    max_seq_length: 200
  algorithms:
    - torch_fullscan
    - triton_knn
    - linr_v3_then_v2
    - silvertorch
    - voyager_hnsw
  algo_params:
    linr_v3_then_v2: { candidate_pool: 5000, v3_seed: 0 }
    silvertorch:     { n_lists: 1024, n_probe: 16, n_iter: 10, seed: 0 }
    voyager_hnsw:    { m: 16, ef_construction: 200, ef_query: null, num_threads: -1, seed: 0 }

<<: *defaults
checkpoint: checkpoints/gsasrec-500m-listens-d128-drop0.5/best_model.pt
```

Filter-bench configs (`goodreads-*.yaml`, `arxiv-*.yaml`) add a `filters:`
block declaring `clause`, `bloom`, and `combined` filter kinds with their
attribute paths and named sweeps; see [filtering.md](filtering.md) for the
filter API.

### Field reference

| Field | Type | Purpose |
|---|---|---|
| `checkpoint` | path | SASRec `best_model.pt` path. Required for yambda/goodreads, omit for arxiv. The trainer writes a sibling `config.json` (via `GSASRecConfig.save`) which the loader reads to pick up `embedding_dim`, `dropout`, `reuse_item_embeddings`, etc. — no hyperparams in the eval YAML. |
| `query_emb_path` | path or `null` | Pre-encoded query tensor for arxiv. Defaults to `<data_dir>/content/query_emb.pt`. |
| `data_dir` | path | Holds `item_id_map.json`, `<split>.parquet`, optional `eval_split.parquet` + filter attrs. |
| `output` | path or `null` | Output JSON path. `null` → `<ckpt-dir>/evaluate.json` (SASRec datasets) or `<data_dir>/evaluate.json` (arxiv). |
| `split` | str | `test` (default) or `val`. |
| `device` | str | `cuda` (only meaningful value today). |
| `ks` | list[int] | K-cutoffs to evaluate. Each emits `recall@K`, `ndcg@K` on its own row. |
| `batch_sizes` | list[int] | Perf-pass batch sizes. Quality is invariant to bs and computed once per `(algo, k)`; emitted on every bs row. |
| `seed` | int | Drives `torch.manual_seed`, `torch.cuda.manual_seed_all`, the perf-query-pool generator, and any algo seeds that read from `algo_params`. |
| `encode.batch_size` | int | Forward-pass batch size for the SASRec query encode. Independent of perf bs. |
| `encode.num_workers` | int | DataLoader workers for the encode pass. |
| `encode.max_seq_length` | int | History truncation length; must match the trained checkpoint. |
| `algorithms` | list[str] | Subset of [`ALGORITHMS`](../../evaluation/retrieval/algo_registry.py). |
| `algo_params` | dict | Per-algo knobs. Algos with no entry use the registry defaults. |
| `filters` | dict or `null` | Optional filter-bench block. See [filtering.md](filtering.md). |

### CLI overrides

```
uv run evaluate \
    --config conf/<name>.yaml \
    [--algorithms <name> --algorithms <name> ...] \
    [--filter-kind {none|clause|bloom|combined}] \
    [--sweep <sweep_name>] \
    [--output <path>]
```

`--algorithms` *replaces* (does not merge into) the YAML's algorithms
list. `--filter-kind` and `--sweep` narrow the run to one filter cell —
useful for iterating on a single sweep without re-encoding queries for
every other cell.

## Algorithms

Six algorithms are registered. `voyager_hnsw` is the lone CPU baseline;
the others run on GPU. `linr_v2_filter_compact` is the exact filtered
top-K path (recall=1.0 vs the filtered-FullScan oracle); it requires a
filter and is only listed in goodreads/arxiv configs.

| Name | Class | Notes |
|---|---|---|
| `torch_fullscan` | `FullScanKNN` | Reference exhaustive IP scan; mask post-filter on filtered cells (skipped on the goodreads filter suite — equals the oracle). |
| `triton_knn` | `LiNR_V1_Triton` | Pure-torch full-scan KNN — `LiNR_V1_Triton` is a backend-dispatch alias over the same `query @ x.T + topk` path as `LiNR_V1`. |
| `linr_v3_then_v2` | `LiNR_V3_Triton` → `LiNR_V2_Triton` | Quantized V3 pre-filters to top-`candidate_pool`; V2 reranks at full precision. Approximate. |
| `linr_v2_filter_compact` | `LiNR_V2_Triton` | Exact filtered top-K via the filter primitive's native compact `(candidate_ids, counts)` path. Recall=1.0 by construction; headline is speed/memory. Filter suite only. |
| `silvertorch` | `SilverTorch` | Bloom-disabled on yambda/quality; bloom-fused (`m_bits`/`k_hash` set) on wide and combined filter sweeps. Skipped on narrow-only filter sweeps via `SilvertorchSkippedOnNarrow`. Approximate. |
| `voyager_hnsw` | `VoyagerHNSW` | Spotify HNSW (`voyager.Index`, InnerProduct space), multi-threaded by default. Yambda only — filtered configs apply a post-mask. |

The build pipeline lives in [algo_registry.py](../../evaluation/retrieval/algo_registry.py):
`build_algorithm` (unfiltered) and `build_filtered_algorithm` (filter-aware)
both return `(forward, modules, is_cpu)`. The `is_cpu` flag drives:

- which perf-measurement primitive to call (`measure_forward_cuda` vs
  `measure_forward_cpu`),
- whether to report `index_mem_mib` / `peak_mem_mib` / `fwd_scratch_mib`
  (zeroed for CPU rows — mixing GPU and CPU memory deltas in the same
  column would be meaningless),
- the `device` field on the output row (`"cuda"` or `"cpu"`).

## Measurement methodology

### Quality pass

`quality_pass_cached(forward, queries, targets, num_targets, k=...)`
streams every cached query through the index at bs=1, accumulating
per-query recall@K and ndcg@K via
[`accumulate_metrics`](../../evaluation/retrieval/metrics.py). Quality is
invariant to perf batch size (same scoring math, just a different leading
dim), so we run this **once per `(algo, k)` cell** and attach the result
to every `bs` row of the same cell — avoids 3× cost on the bs sweep.

For filtered cells the targets passed in are the cached filtered-FullScan
top-K oracle (`<data_dir>/gt/gt_topk_<sweep>.pt`); for `filter_kind=none`
and yambda they are the held-out test items.

### Perf pass

The perf pass measures the index forward in isolation, on the device the
algorithm lives on. It is split between two primitives:

- **`measure_forward_cuda`** (the GPU path): runs `WARMUP_ITERS=20`
  calls to settle Triton autotune and JIT, then takes a clean
  `torch.cuda.max_memory_allocated()` window over `mem_reps=5` calls
  (without `do_bench`'s 256 MiB L2-buster polluting peak), then times via
  `triton.testing.do_bench(rep=200ms, warmup=50)`.
  Returns `(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)`.
- **`measure_forward_cpu`** (the CPU path, used today by
  `voyager_hnsw`): 3 warmup calls, then a `time.perf_counter` loop
  within a 200 ms budget. Returns the same tuple shape with
  peak/transient = 0.

### Multi-query pool — why p20/p80 are over queries

The perf pass times against a **fixed-seed pool of 64 query batches**,
round-robin'd into the timed function. This matters for IVF-style
algorithms (`silvertorch`) and graph-walk ANN (`voyager_hnsw`) where a
single fixed query collapses p20/p80 to one cluster's / one path's
traversal cost — degenerate percentiles. With the pool, p20/p80
reflect query diversity (the intended workload variance), not CUDA
scheduling jitter.

The pool itself is built per `(algo, k, bs)` cell:

```python
g = torch.Generator(device="cpu").manual_seed(seed)
rows = torch.randint(0, n_users, (n_pool, batch_size), generator=g)
pool = queries[rows.reshape(-1)].reshape(n_pool, batch_size, -1).to(device)
```

so the *same query indices* are sampled across reruns with the same seed.

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
modules.clear(); del; empty_cache
```

`modules.clear()` is required — `for m in modules: del m` only drops the
loop variable, not the list's references, leaking the index into the
next cell's `mem_before`.

### Determinism

`main()` calls `torch.manual_seed(cfg.seed)` and
`torch.cuda.manual_seed_all(cfg.seed)` before any allocation. The
perf-query-pool generator uses the same seed. Algo seeds are wired
through `algo_params` (e.g. `silvertorch.seed`,
`linr_v3_then_v2.v3_seed`, `voyager_hnsw.seed`).

Quality columns must be **byte-identical** across reruns with the same
seed. Latency may drift within ~5% due to clock noise, NVML thermal
state, and (for Triton autotune) JIT cache state.

## Output schema

One row per `(filter_kind, sweep, algo, k, bs)` cell. Fields:

| Field | Type | Notes |
|---|---|---|
| `suite` | str | `yambda` (no filters) or `filter`. |
| `cell` | str | `<filter_kind>_<sweep>_bs<N>_k<K>` for cross-row joins. |
| `filter_kind` | str | `none`, `clause`, `bloom`, or `combined`. |
| `sweep` | str | Filter sweep name from the config (e.g. `c0_genre`, `1shelf`). |
| `impl` | str | Algorithm name. |
| `device` | str | `"cuda"` or `"cpu"`. |
| `seed` | int | The `cfg.seed` that produced this row. |
| `batch_size` | int | First-class column. |
| `k` | int | Same. |
| `n_users_kept` | int | Number of users this sweep evaluated (skip mask drops users with no surviving narrow clauses or empty wide bag). |
| `median_ms`, `p20_ms`, `p80_ms` | float | Latency over the multi-query pool. |
| `peak_mem_mib` | float | `max_memory_allocated()` over the perf window. 0 for CPU rows. |
| `index_mem_mib` | float | `allocated()` delta around `build_algorithm`. 0 for CPU rows. |
| `fwd_scratch_mib` | float | Peak − baseline within the forward call. 0 for CPU rows. |
| `recall@<k>`, `ndcg@<k>` | float | Quality, identical across all bs rows of the same `(algo, k)` cell. |
| `extra.params` | dict[str, str] | Per-algo `algo_params` for traceability. |

Skipped cells emit a one-row stub with `"skipped": true` and a `"reason"`
field instead of the perf/quality columns.

Downstream analysis: filter by `device` to compare GPU rows against each
other separately from CPU baselines; group by `(impl, k)` and span `bs`
to read scaling behavior; group by `(filter_kind, sweep)` to compare
filter shapes.

## How to run

The repo is a uv workspace ([root pyproject](../../pyproject.toml));
`evaluation/` shares a single `.venv` with `retrieve/` at the workspace
root. `uv run` from inside `evaluation/` discovers the workspace root
automatically; equivalently use `uv run --directory evaluation …` from
the root.

Yambda 500M sweep:

```bash
cd evaluation
uv run evaluate --config conf/500m-d128.yaml
```

5B sweep (gated on the 5B checkpoint having been trained):

```bash
uv run evaluate --config conf/5b-d64.yaml
```

Goodreads filter bench:

```bash
uv run evaluate --config conf/goodreads-d128-drop0.5-id.yaml
```

Arxiv filter bench:

```bash
uv run evaluate --config conf/arxiv-d256.yaml
```

CPU baseline only on yambda:

```bash
uv run evaluate \
    --config conf/500m-d128.yaml \
    --algorithms voyager_hnsw
```

### Sanity checks to run after a sweep

1. `device == "cpu"` and `index_mem_mib == 0` for the `voyager_hnsw`
   row.
2. `voyager_hnsw` `recall@K` within ~2 pp of `silvertorch` at
   comparable settings (`m`/`ef_query` vs `n_lists`/`n_probe`) — both
   are approximate, so a wider band than an exact-vs-exact comparison
   is expected.
3. GPU `median_ms(bs=8)` < `8 × median_ms(bs=1)` (sub-linear scaling —
   the proof point of the GPU implementations).
4. `voyager_hnsw` `median_ms(bs=8)` scales near-linearly with `bs`
   (HNSW's per-query graph walk doesn't share work across queries).
5. `recall@K` and `ndcg@K` columns identical across the bs rows of the
   same `(algo, k)` cell.
6. Two reruns with the same seed: quality columns byte-identical;
   latency columns within ~5%.

## Extending

### Add a new algorithm

1. Implement an `nn.Module` (or a `RetrievalModule` subclass — see
   [interfaces.py:RetrievalModule](../../retrieve/src/retrieve/interfaces.py))
   exposing `register_index(item_embs)` + `forward(query) -> (ids, scores)`.
2. Add the name to `ALGORITHMS` and a branch in `build_algorithm` (and
   `build_filtered_algorithm`, if it should support filters) at
   [algo_registry.py](../../evaluation/retrieval/algo_registry.py). Mark
   it in `CPU_ALGOS` if it lives on CPU.
3. Add it to the `algorithms:` list in any config that should sweep it,
   plus an `algo_params` entry if it takes knobs.
4. No driver changes needed — the perf primitive is selected by
   `is_cpu`, and the schema columns are written uniformly.

### Add a new config (new checkpoint)

Copy an existing YAML, update `checkpoint:` (or `query_emb_path:`) and
any catalog-size-driven knobs (`silvertorch.n_lists/n_probe`,
`voyager_hnsw.m/ef_construction/ef_query`). The model loader reads
hyperparams from `<ckpt-dir>/config.json`, so the YAML never carries
`embedding_dim` etc.

### Add a new dataset

The driver dispatches on config shape, not a `dataset:` field, so adding
a new dataset is a question of producing the on-disk artifacts the
driver consumes. The dataset CLIs live in [`evaluation/data/`](../../evaluation/data/)
and produce, for the yambda layout:

```
<data_dir>/
├── item_id_map.json
├── train.parquet
├── val.parquet
└── test.parquet
```

For the arxiv layout (no SASRec, pre-encoded text):

```
<data_dir>/
├── item_id_map.json
├── papers.parquet
├── heldout.parquet
├── content/
│   ├── text_emb.pt           # item-side, "search_document: " prefix
│   ├── text_emb.meta.json
│   ├── query_emb.pt          # query-side, "search_query: " prefix
│   └── query_emb.meta.json
└── eval_split.parquet         # optional — only needed for filter sweeps
```

For filter sweeps either layout adds:

```
<data_dir>/
├── item_attrs_narrow.pt
├── item_attrs_wide.pt
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
├── ... (per-clause vocab JSONs)
└── gt/                        # auto-built by the driver, oracle cache
```

The CLIs that build all of the above are exposed as console scripts:
`uv run yambda prep ...`, `uv run arxiv all ...`, `uv run goodreads all ...`.

## See also

- [architecture.md](architecture.md) — package layout and the
  `RetrievalModule` / `FilterModule` contracts.
- [kernels.md](kernels.md) — Triton kernel internals for V1/V2/V3,
  IVF-INT8, bloom-match.
- [filtering.md](filtering.md) — filter API and the with-filters story.
- [checkpoints.md](checkpoints.md) — the trainer pipeline that produces
  the checkpoints consumed here.
- [testing.md](testing.md) — the correctness suite (separate from this
  perf harness).
