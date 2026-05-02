<!-- claude code generated file -->

# `evaluation/` retrieval harness

Live reference for the in-process retrieval benchmark used to compare
algorithms on the Yambda dataset. Covers the driver layout, the YAML
config format, the measurement methodology, the output schema, and the
extension points for adding a new algorithm, config, or dataset.

For the algorithm internals (what each `forward(query)` does) see
[kernels.md](kernels.md) and [architecture.md](architecture.md). For the
training pipeline that produces the checkpoints consumed here, see
[checkpoints.md](checkpoints.md). The correctness-only test suite that
gates kernel changes is documented in [testing.md](testing.md).

## Scope

`evaluation/retrieval/benchmark.py` is the **sole** retrieval
benchmark driver. It loads a trained GSASRec checkpoint, encodes the eval
split's queries once into a CPU cache, then for each
`(algorithm, K, batch_size)` cell:

1. Builds the index on top of the model's output embeddings.
2. Streams every cached query through the index at bs=1 to compute
   recall@K and ndcg@K (the **quality pass**).
3. Times the index forward at the configured batch size against a
   fixed-seed pool of cached queries (the **perf pass**), capturing
   median / p20 / p80 latency plus peak / transient GPU memory.
4. Writes one row per cell to `<ckpt-dir>/benchmark.json`.

The harness is intentionally narrow: it is the consumer-facing benchmark
for production retrieval modules, not a parity or correctness check.
Correctness lives in [`retrieve/tests/`](../../retrieve/tests/) and gates
CI; this harness reports numbers and never fails a build.

A second harness for filter-bearing datasets is planned but not yet
present.

## Files

```
evaluation/retrieval/
├── benchmark.py               # driver (main)
├── registry.py                # algorithm registry: name → (forward, modules, is_cpu)
├── config.py                  # EvalConfig dataclass + yaml loader
├── faiss_baselines.py         # FaissFlatIP, FaissIVFFlat (CPU)
└── metrics.py                 # accumulate_metrics, finalize_metrics

evaluation/conf/
├── smoke.yaml                 # 50M Listen+ smoke (~2 min)
├── 500m-d64.yaml              # 500M Listen+, d=64 checkpoint
├── 500m-d128.yaml             # 500M Listen+, d=128 checkpoint
├── 500m-d256.yaml             # 500M Listen+, d=256 checkpoint
├── 5b-d64.yaml                # 5B Listen+, d=64 checkpoint
└── 5b-d128.yaml               # 5B Listen+, d=128 checkpoint
```

## Configuration

Configs are YAML, parsed with `yaml.safe_load` and slammed into the
`EvalConfig` dataclass at [config.py](../../evaluation/retrieval/config.py).
Each file declares a `_defaults: &defaults` anchor block holding every
field except the per-checkpoint pointer, then merges that block at the
top level with `<<: *defaults` and adds `checkpoint:`. The loader strips
keys starting with `_` so the anchor scratch key doesn't break the
dataclass constructor.

Example structure (full files in `evaluation/conf/`):

```yaml
_defaults: &defaults
  data_dir: data/yambda/500m-listens
  output: null                  # null → <ckpt-dir>/benchmark.json
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
    - faiss_flat_ip
    - faiss_ivf_flat
  algo_params:
    linr_v3_then_v2: { candidate_pool: 5000, v3_seed: 0 }
    silvertorch:     { n_lists: 1024, n_probe: 16, n_iter: 10, seed: 0 }
    faiss_ivf_flat:  { nlist: 2048, nprobe: 16, seed: 0 }

<<: *defaults
checkpoint: checkpoints/gsasrec-500m-listens-d128-drop0.5/best_model.pt
```

### Field reference

| Field | Type | Purpose |
|---|---|---|
| `checkpoint` | path | Required. `best_model.pt` path. The trainer writes a sibling `config.json` (via `GSASRecConfig.save`) which the loader reads to pick up `embedding_dim`, `dropout`, `reuse_item_embeddings` etc. — no hyperparams in the eval YAML. |
| `data_dir` | path | Holds `item_id_map.json` (catalog size) and `<split>.parquet` (eval set). |
| `output` | path or `null` | Output JSON path. `null` → `<ckpt-dir>/benchmark.json`. |
| `split` | str | `test` (default) or `val`. |
| `device` | str | `cuda` (only meaningful value today). |
| `ks` | list[int] | K-cutoffs to evaluate. Each emits `recall@K`, `ndcg@K` on its own row. |
| `batch_sizes` | list[int] | Perf-pass batch sizes. Quality is invariant to bs and computed once per `(algo, k)`; emitted on every bs row. Smoke uses `[1]` to stay fast. |
| `seed` | int | Drives `torch.manual_seed`, `torch.cuda.manual_seed_all`, the perf-query-pool generator, and any algo seeds that read from `algo_params`. |
| `encode.batch_size` | int | Forward-pass batch size for the one-shot query encode. Independent of perf bs. |
| `encode.num_workers` | int | DataLoader workers for the encode pass. |
| `encode.max_seq_length` | int | History truncation length; must match the trained checkpoint. |
| `algorithms` | list[str] | Subset of [`ALGORITHMS`](../../evaluation/retrieval/registry.py). Order matters only for log readability. |
| `algo_params` | dict | Per-algo knobs. Algos with no entry use the registry defaults. |

### CLI overrides

```
uv run benchmark \
    --config conf/<name>.yaml \
    [--algorithms <name> --algorithms <name> ...] \
    [--output <path>]
```

`--algorithms` *replaces* (does not merge into) the YAML's algorithms
list. Useful for narrowing a sweep to one baseline:

```
uv run benchmark \
    --config conf/smoke.yaml \
    --algorithms faiss_flat_ip --algorithms faiss_ivf_flat
```

## Algorithms

Six algorithms are registered as of this writing. The first four run on
GPU; the last two are CPU-resident faiss baselines.

| Name | Class | Lives where | Notes |
|---|---|---|---|
| `torch_fullscan` | `FullScanKNN` | GPU | Reference exhaustive IP scan. |
| `triton_knn` | `LiNR_V1_Triton` | GPU | Single-pass Triton KNN. |
| `linr_v3_then_v2` | `LiNR_V3_Triton` → `LiNR_V2_Triton` | GPU | Quantized V3 pre-filters to top-`candidate_pool`; V2 reranks at full precision. V1-as-stage-2 is intentionally absent — it's strictly slower than `triton_knn` alone on this workload. |
| `silvertorch` | `SilverTorch` (`m_bits=None`, `k_hash=None`) | GPU | The bloom-fused configuration is *not* used because Yambda has no item attributes; running it against zero signatures would degenerate. With both bloom params left unset, `SilverTorch` skips bloom buffer allocation and runs as a plain IVF + INT8 ANN. |
| `faiss_flat_ip` | `FaissFlatIP` | CPU | Single-thread (`faiss.omp_set_num_threads(1)` at import). Apples-to-apples flat baseline. |
| `faiss_ivf_flat` | `FaissIVFFlat` | CPU | Single-thread IVF baseline. K-means trained on the full catalog with `numpy.random.seed(seed)` for determinism. |

The build pipeline lives in [registry.py:build_algorithm](../../evaluation/retrieval/registry.py),
which returns `(forward, modules, is_cpu)`. The `is_cpu` flag drives:

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

### Perf pass

The perf pass measures the index forward in isolation, on the device the
algorithm lives on. It is split between two primitives:

- **`measure_forward_cuda`** (the GPU path): runs `WARMUP_ITERS=20`
  calls to settle Triton autotune and
  JIT, then takes a clean `torch.cuda.max_memory_allocated()` window over
  `mem_reps=5` calls (without `do_bench`'s 256 MiB L2-buster polluting
  peak), then times via `triton.testing.do_bench(rep=200ms, warmup=50)`.
  Returns `(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)`.
- **`measure_forward_cpu`** (the CPU path for faiss): 3 warmup calls,
  then a `time.perf_counter` loop within a 200 ms budget. Returns the
  same tuple shape with peak/transient = 0.

### Multi-query pool — why p20/p80 are over queries

The perf pass times against a **fixed-seed pool of 64 query batches**,
round-robin'd into the timed function. This matters for IVF-style
algorithms (`silvertorch`, `faiss_ivf_flat`) where a single fixed query
collapses p20/p80 to one cluster's traversal cost — degenerate
percentiles. With the pool, p20/p80 reflect cluster diversity (the
intended workload variance), not CUDA scheduling jitter.

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
`linr_v3_then_v2.v3_seed`, `faiss_ivf_flat.seed`).

Quality columns must be **byte-identical** across reruns with the same
seed. Latency may drift within ~5% due to clock noise, NVML thermal
state, and (for Triton autotune) JIT cache state.

## Output schema

One row per `(algo, k, bs)` cell. Fields:

| Field | Type | Notes |
|---|---|---|
| `suite` | str | `yambda_<split>`. |
| `cell` | str | `bs<N>_k<K>` for cross-row joins. |
| `impl` | str | Algorithm name. |
| `device` | str | `"cuda"` or `"cpu"`. |
| `seed` | int | The `cfg.seed` that produced this row. |
| `batch_size` | int | First-class column (was previously wedged into `extra`). |
| `k` | int | Same. |
| `median_ms`, `p20_ms`, `p80_ms` | float | Latency over the multi-query pool. |
| `peak_mem_mib` | float | `max_memory_allocated()` over the perf window. 0 for CPU rows. |
| `index_mem_mib` | float | `allocated()` delta around `build_algorithm`. 0 for CPU rows. |
| `fwd_scratch_mib` | float | Peak − baseline within the forward call. 0 for CPU rows. |
| `recall@<k>`, `ndcg@<k>` | float | Quality, identical across all bs rows of the same `(algo, k)` cell. |
| `extra.params` | dict[str, str] | Per-algo `algo_params` for traceability. |

Downstream analysis: filter by `device` to compare GPU rows against each
other separately from CPU baselines; group by `(impl, k)` and span `bs`
to read scaling behavior.

## How to run

The repo is a uv workspace ([root pyproject](../../pyproject.toml)); `evaluation/` shares a single `.venv` with `retrieve/` at the workspace root. `uv run` from inside `evaluation/` discovers the workspace root automatically; equivalently use `uv run --directory evaluation …` from the root.

End-to-end smoke (~2 min, single-GPU host):

```bash
cd evaluation
uv run benchmark --config conf/smoke.yaml
```

500M sweep (~1 hour per checkpoint, multiplied by 3 for the bs sweep):

```bash
uv run benchmark --config conf/500m-d128.yaml
```

5B sweep (gated on the 5B checkpoint having been trained):

```bash
uv run benchmark --config conf/5b-d64.yaml
```

Subset baseline check:

```bash
uv run benchmark \
    --config conf/smoke.yaml \
    --algorithms faiss_flat_ip --algorithms faiss_ivf_flat
```

### Sanity checks to run after a sweep

1. `device == "cpu"` and `index_mem_mib == 0` for the two faiss rows.
2. `faiss_flat_ip` recall/ndcg within ~0.5 pp of `torch_fullscan`
   (both are exhaustive IP — should match).
3. `faiss_ivf_flat` recall/ndcg within ~0.5 pp of `silvertorch` at
   comparable `(nlist, nprobe)`.
4. GPU `median_ms(bs=8)` < `8 × median_ms(bs=1)` (sub-linear scaling —
   the proof point of the GPU implementations).
5. CPU faiss `median_ms(bs=8)` ≈ `8 × median_ms(bs=1)` (near-linear —
   single-thread, no SIMD-level batching benefit on flat IP).
6. `recall@K` and `ndcg@K` columns identical across the bs rows of the
   same `(algo, k)` cell.
7. Two reruns with the same seed: quality columns byte-identical;
   latency columns within ~5%.

## Extending

### Add a new algorithm

1. Implement an `nn.Module` (or a `RetrievalModule` subclass — see
   [interfaces.py:RetrievalModule](../../retrieve/src/retrieve/interfaces.py))
   exposing `register_index(item_embs)` + `forward(query) -> (ids, scores)`.
2. Add the name to `ALGORITHMS` and a branch in `build_algorithm` at
   [registry.py](../../evaluation/retrieval/registry.py). Mark it
   in `_CPU_ALGOS` if it lives on CPU.
3. Add it to the `algorithms:` list in any config that should sweep it,
   plus an `algo_params` entry if it takes knobs.
4. No driver changes needed — the perf primitive is selected by
   `is_cpu`, and the schema columns are written uniformly.

### Add a new config (new checkpoint)

Copy an existing YAML, update `checkpoint:` and any catalog-size-driven
knobs (`silvertorch.n_lists/n_probe`, `faiss_ivf_flat.nlist/nprobe`).
The model loader reads hyperparams from `<ckpt-dir>/config.json`, so
the YAML never carries `embedding_dim` etc.

### Add a new dataset (e.g. with-filters)

Out of scope for this driver. The plan is a separate
`<dataset>_benchmark.py` driver that consumes the same
`build_algorithm` + `metrics` building blocks but encodes a different
dataset and threads the per-query filter attributes through the
`forward(q, attrs)` signature for filter-aware algorithms.

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
