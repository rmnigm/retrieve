# Refactor the Yambda retrieval evaluation pipeline + add a faiss-cpu baseline

> Previously: `evaluation/docs/REFACTOR_PLAN.md`.

## Context

`evaluation/retrieval/eval_yambda_retrieval.py` benchmarks four retrieval algorithms (`torch_fullscan`, `triton_knn`, `linr_v3_then_v2`, `silvertorch`) on the Yambda 500M-listens dataset. It loads a trained GSASRec checkpoint, encodes the test split's queries once, then for each `(algorithm, K)` cell builds the index, runs a quality pass at bs=1, and runs a perf pass at bs=1 against a single fixed query.

The user reports the eval results are **somewhat inconsistent** (run-to-run jitter). On audit, several things contribute:

1. **No reproducibility seed** anywhere in the eval driver — `torch.manual_seed` is never called, so any code path that pulls from the default RNG (k-means init in `IVF_INT8_ANN._kmeans_torch` at `silvertorch/ivf.py:131` uses a CPU `Generator` with the configured seed, but `LiNR_V3` uses a different RNG, and Triton autotune cache state is implicit).
2. **`mem_before` is sampled before `empty_cache()`**, so the previous cell's allocator fragmentation pollutes the index-memory delta (`eval_yambda_retrieval.py:264` — the cleanup at line 305 happens *after* `mem_before` is taken on the next cell's first iteration). Confirmed by re-reading the loop: `mem_before` ← then `build_algorithm` ← then `index_mem`. The empty_cache happens at end of *previous* cell, but kmeans/topk transients from the previous build can leave the allocator pool fragmented.
3. **Perf pass uses a single fixed query** (`queries[:1]`), so for cluster-skewed algorithms (IVF / silvertorch) the reported median/p20/p80 reflects only one cluster's traversal cost, not real workload variance. This makes p20/p80 misleading.
4. **`measure()` over-warms**: `eval_yambda_retrieval.py:53` passes `warmup=200` to `do_bench` (claims to mirror `tests/bench/conftest.py`, which actually uses `warmup=50`). Different warmup windows across reruns is one source of jitter on small kernels.
5. **Build-vs-forward measurement is interleaved**: `cuda_allocated_mib() - mem_before` straddles the kmeans build's transient peak, so the reported `index_mem_mib` is not deterministic w.r.t. allocator state.
6. **Driver is monolithic** (~315 lines doing config + load + encode + build + quality + perf + write), and `eval_quality.py` / `eval_perf.py` duplicate parts of it.

**Goal**: Refactor so measurements are deterministic and isolated, add a `faiss_flat_ip` (CPU exhaustive) and `faiss_ivf_flat` (CPU IVF) baseline, extend the perf sweep to bs ∈ {1, 8, 16}, and tighten the YAML configs. **Scope is intentionally tight** — we are not rewriting the bench harness, just plugging the leaks and adding the new baseline + bs sweep.

---

## Plan

### 1. Add reproducibility seeding in the driver

In `eval_yambda_retrieval.py:main`, right after `cfg = load_eval_config(...)`, take a top-level `seed` from the YAML (default `0`) and call:

```python
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
```

Add `seed: int = 0` to `EvalConfig` in [retrieval/config.py](../evaluation/retrieval/config.py). This locks `torch.topk` tie-order and any default-RNG draws.

### 2. Fix the memory-snapshot ordering

In `eval_yambda_retrieval.py:260-305`, restructure the per-cell loop so that **before** taking `mem_before`, we `empty_cache()` and `synchronize()`. New order per cell:

```
synchronize → empty_cache → reset_peak_memory_stats → mem_before
build_algorithm
synchronize → index_mem = allocated - mem_before
quality_pass
perf_pass
clear modules / del / empty_cache
```

This isolates the index's marginal cost from the previous cell's allocator residue. Keep the `modules.clear()` + `del` cleanup at the end of the cell (the existing comment at line 300-302 is correct).

### 3. Align the perf-bench primitive with `tests/bench/conftest.py`

Replace the local `measure()` and `forward_memory_mib()` in `eval_yambda_retrieval.py:43-75` with **a thin call to `tests.bench.conftest.measure_forward`** if the import is reachable, OR an inlined copy with `warmup_iters=20, mem_reps=5` matching the canonical version. Since `evaluation/` does not depend on `tests/`, the cleanest move is to copy the body of `measure_forward` (`tests/bench/conftest.py:281-334`) into `eval_yambda_retrieval.py` (or into a new `retrieval/bench_utils.py`) and delete the local re-implementation. This gives us:

- one warmup window (`warmup_iters` warmup calls, no over-warming),
- separate memory window (without `do_bench`'s 256 MiB L2-buster polluting peak),
- `do_bench(rep_ms=200, warmup=50)` for timing.

### 4. Sample multiple queries for perf p20/p80, and extend to bs=8 and bs=16

Two related changes — both motivated by the same problem (single fixed query → degenerate percentiles):

**(a) Multi-query sampling at each batch size.** Replace the single `queries[:1]` perf query with a small fixed-seed sample (say 64 queries per batch), and call `forward(q)` round-robin in the timed function so each timing iteration sees a different query / cluster. Concretely:

```python
g = torch.Generator(device="cpu").manual_seed(cfg.seed)
def make_perf_pool(batch_size: int, n_pool: int = 64) -> torch.Tensor:
    n = queries.shape[0]
    rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
    return queries[rows.reshape(-1)].reshape(n_pool, batch_size, -1).to(device)

pool = make_perf_pool(bs)
counter = {"i": 0}
def perf_fn():
    q = pool[counter["i"] % pool.shape[0]]
    counter["i"] += 1
    with torch.inference_mode():
        forward(q)
```

This restores cluster diversity in the IVF / silvertorch percentiles. Document that p20/p80 is over **queries**, not over **CUDA scheduling jitter**.

**(b) Extend the perf pass to bs=1, bs=8, bs=16.** Add a `batch_sizes: list[int] = [1, 8, 16]` field to `EvalConfig` (default `[1, 8, 16]` — but `[1]` for the smoke config to keep it short). The driver's per-cell loop becomes triple-nested: `for algo in algorithms: for k in ks: for bs in batch_sizes: …`.

Quality is **not** dependent on batch size for any of these algorithms (same scoring math, just a different leading dim), so the **quality pass runs once per `(algo, k)` cell** at bs=1 (or at the encode batch size — see below) and the recall/ndcg numbers are then attached to every batch-size row of that cell. To avoid a 3× quality-pass blow-up, hoist the quality pass out of the batch-size inner loop:

```python
for algo in cfg.algorithms:
    algo_params = cfg.algo_params.get(algo, {})
    for k in cfg.ks:
        # build once at this k
        forward, modules = build_algorithm(...)
        # quality at bs=1, attach to every bs row
        recall, ndcg = quality_pass_cached(forward, ..., k=k)
        for bs in cfg.batch_sizes:
            med, p20, p80, peak, scratch = perf_pass_cached(forward, queries, batch_size=bs, ...)
            row = {"cell": f"bs{bs}_k{k}", "batch_size": bs, ..., f"recall@{k}": recall, f"ndcg@{k}": ndcg}
            rows.append(row)
        modules.clear(); del forward, modules; torch.cuda.empty_cache()
```

`perf_pass_cached` takes a `batch_size: int` parameter and uses the multi-query pool at that batch size. The cell key `"bs1_k100"` becomes `"bs8_k100"` / `"bs16_k100"` for the new rows.

**Caveats to validate**:
- For algorithms whose forward path branches on shape (e.g. Triton autotune may select a different config for bs=1 vs bs=16), `measure_forward`'s warmup window must be re-run **per batch size**. The simplest implementation is to call `measure_forward` separately for each bs (which we already do — the `pool` and `perf_fn` closure are rebuilt per `bs`).
- For `linr_v3_then_v2` cascade at large batches, V3's candidate_pool=5000 candidates per query × bs=16 = 80K candidate rows that V2 must rerank — make sure `LiNR_V2_Triton` handles this (it should; the kernel supports `[B, C]` candidate_ids).
- For `faiss_flat_ip` and `faiss_ivf_flat` at bs=8 / bs=16 with single-thread, latency scales near-linearly with bs (no SIMD-level batching benefit on flat IP), so the new rows mostly serve to confirm that the GPU baselines amortize bs much better than CPU does.

### 5. Add the `faiss_flat_ip` and `faiss_ivf_flat` baselines

Two new entries in `retrieval/algorithms.py`:

```python
ALGORITHMS = (
    "torch_fullscan",
    "triton_knn",
    "linr_v3_then_v2",
    "silvertorch",
    "faiss_flat_ip",
    "faiss_ivf_flat",
)
```

Implementation lives in a new file [retrieval/faiss_baselines.py](../evaluation/retrieval/faiss_baselines.py) with two `nn.Module` subclasses that conform to the `forward(query) -> (ids, scores)` shape. They keep CPU `numpy` buffers and convert per call:

- `FaissFlatIP(k)`: wraps `faiss.IndexFlatIP(d)`; on `register_index(item_embs)` does `index.add(item_embs.detach().cpu().numpy().astype(np.float32))` and zeros row 0 to mirror the GPU pad treatment. `forward(query)` does `q_np = query.detach().cpu().numpy().astype(np.float32, copy=False)`, calls `index.search(q_np, k)`, and returns `(ids_t.to(query.device), scores_t.to(query.device))`.
- `FaissIVFFlat(k, nlist, nprobe, seed)`: trains `IndexIVFFlat(quantizer=IndexFlatIP, d, nlist, METRIC_INNER_PRODUCT)` on the full catalog (1.87M is small enough — no need to subsample), sets `index.nprobe = nprobe`. Set `faiss.cvar.indexIVF_stats.reset()` and use `faiss.RandomGenerator(seed)` indirectly via `faiss.normalize_L2`-free flow (faiss seeds k-means deterministically when `numpy.random.seed` is set; we set both at module init).

In `algorithms.py`, the new branches:

```python
if name == "faiss_flat_ip":
    idx = FaissFlatIP(k=k)
    idx.register_index(item_embs)
    return (lambda q: idx(q)), [idx]

if name == "faiss_ivf_flat":
    nlist = int(p.get("nlist", 2048))
    nprobe = int(p.get("nprobe", 16))
    seed = int(p.get("seed", 0))
    idx = FaissIVFFlat(k=k, nlist=nlist, nprobe=nprobe, seed=seed)
    idx.register_index(item_embs)
    return (lambda q: idx(q)), [idx]
```

**Threading**: at module-import time call `faiss.omp_set_num_threads(1)` for an apples-to-apples bs=1 single-thread baseline. Document this in the algorithm's docstring; multi-thread can be added later behind a config knob if the user wants it.

### 6. Make the perf pass device-aware

The CUDA `do_bench` path doesn't time CPU work. In `perf_pass_cached` (eval_yambda_retrieval.py:188-205), branch on `torch.cuda.is_available()` AND on whether the algorithm is CPU-resident. Simplest cut: have `build_algorithm` also return an `is_cpu: bool` flag (or detect via `next(modules[0].buffers()).is_cuda` after a register), and route to either the CUDA `measure_forward` or a new `measure_forward_cpu`:

```python
def measure_forward_cpu(fn, *, rep_ms=200.0):
    for _ in range(3): fn()
    times = []
    deadline = time.perf_counter() + rep_ms/1000
    while time.perf_counter() < deadline:
        t0 = time.perf_counter(); fn(); times.append((time.perf_counter()-t0)*1000)
    times.sort()
    n = len(times)
    return times[n//2], times[max(0, n*20//100)], times[min(n-1, n*80//100)]
```

For CPU rows, `peak_mib` / `index_mem_mib` / `fwd_scratch_mib` are reported as `0` and the JSON row gets a new `device: "cpu"` field so downstream analysis isn't surprised. Mixing GPU and CPU memory numbers in the same column is the easy way to report meaningless deltas.

### 7. Capture the result schema cleanly

Add `device`, `seed`, and `batch_size` (now a real column, no longer wedged into `extra`) fields to each row written to the output JSON in `eval_yambda_retrieval.py:276-294` so the artifact is self-describing:

```python
row = {
    "suite": f"yambda_{cfg.split}",
    "cell": f"bs{bs}_k{k}",
    "impl": algo,
    "device": "cuda" if is_gpu else "cpu",
    "seed": cfg.seed,
    "batch_size": bs,
    "k": k,
    "median_ms": med, "p20_ms": p20, "p80_ms": p80,
    "peak_mem_mib": peak, "index_mem_mib": index_mem, "fwd_scratch_mib": scratch,
    f"recall@{k}": recall, f"ndcg@{k}": ndcg,
    "extra": {"params": {...}},
}
```

### 8. DRY the YAML configs

The four configs (`smoke.yaml`, `500m-d64.yaml`, `500m-d128.yaml`, `500m-d256.yaml`) are nearly identical. Use YAML anchors and add `batch_sizes`:

```yaml
_defaults: &defaults
  data_dir: data/yambda/500m-listens
  output: null
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

`smoke.yaml` overrides `batch_sizes: [1]` and `ks: [100]` to stay fast.

This brings each per-checkpoint file to ~3 lines (checkpoint path + any per-checkpoint overrides) without changing the loader. (No code change in `config.py` is required — `yaml.safe_load` resolves anchors.)

### 9. Add the dependency

Append to `evaluation/pyproject.toml` `[project].dependencies`:

```toml
"faiss-cpu>=1.8",
"numpy>=1.26",  # already implied transitively, but pin floor
```

`faiss-cpu` is pure CPU, ships its own OpenMP, and does not conflict with the `pytorch-cu128` indexed torch.

### 10. (NOT doing) Things deliberately out of scope

- Not rewriting `eval_perf.py` / `eval_quality.py` — those are pre-export code paths and don't run as part of `eval_yambda_retrieval.py`. Touching them now would balloon the diff.
- Not adding HNSW (build cost is hours on CPU at 1.87M items; not worth it for a baseline row).
- Not adding multi-threaded faiss row (single-thread is the apples-to-apples comparison; can be added later behind a knob).
- Not adding GPU faiss (we'd be adding a third install path; `faiss-gpu` requires older CUDA, doesn't co-exist cleanly with `pytorch-cu128`).
- Not validating `candidate_ids >= 1` in metrics; on inspection, the existing pad-zeroing + `-inf` masking in IVF is sufficient and the auditor's concern doesn't manifest in practice with `item_embs[0] = 0.0`.

---

## Files to modify

| File | Change |
|---|---|
| [evaluation/retrieval/eval_yambda_retrieval.py](../evaluation/retrieval/eval_yambda_retrieval.py) | seed at start of `main`; reorder mem snapshot; replace `measure`/`forward_memory_mib` with the conftest-aligned `measure_forward`; multi-query perf pass with bs∈{1,8,16}; hoist quality pass out of the bs loop; CPU-vs-CUDA branch in `perf_pass_cached`; add `device`/`seed`/`batch_size`/`k` fields to result row |
| [evaluation/retrieval/config.py](../evaluation/retrieval/config.py) | add `seed: int = 0` and `batch_sizes: list[int] = [1, 8, 16]` to `EvalConfig` |
| [evaluation/retrieval/algorithms.py](../evaluation/retrieval/algorithms.py) | add `faiss_flat_ip` and `faiss_ivf_flat` to `ALGORITHMS` and `build_algorithm` |
| [evaluation/retrieval/faiss_baselines.py](../evaluation/retrieval/faiss_baselines.py) | NEW — `FaissFlatIP`, `FaissIVFFlat` `nn.Module` wrappers |
| [evaluation/conf/smoke.yaml](../evaluation/conf/smoke.yaml) | refactor with `&defaults` anchor; add faiss baselines |
| [evaluation/conf/500m-d64.yaml](../evaluation/conf/500m-d64.yaml) | same |
| [evaluation/conf/500m-d128.yaml](../evaluation/conf/500m-d128.yaml) | same |
| [evaluation/conf/500m-d256.yaml](../evaluation/conf/500m-d256.yaml) | same |
| [evaluation/pyproject.toml](../evaluation/pyproject.toml) | add `faiss-cpu>=1.8` |

## Reused existing utilities

- `RetrievalModule` interface (`retrieve/src/retrieve/interfaces.py:22`) — `FaissFlatIP` / `FaissIVFFlat` subclass it; the `(forward, modules)` contract in `algorithms.py:46` is the only shape we have to match.
- `accumulate_metrics` / `finalize_metrics` ([metrics.py:124](../evaluation/retrieval/metrics.py)) — the new baselines feed the same `(ids, scores)` tensors into them.
- `measure_forward` body (`retrieve/tests/bench/conftest.py:281`) — the canonical bench primitive we copy into the eval driver to replace the over-warmed local version.

## Verification

1. **Smoke run** (≤2 minutes, ckpt is small):
   ```bash
   uv run python -m retrieval.eval_yambda_retrieval --config conf/smoke.yaml
   ```
   Confirm one row per `(algo, k, bs)` including `faiss_flat_ip` and `faiss_ivf_flat`. Smoke uses `batch_sizes: [1]` so the count is `n_algos × n_ks × 1`. Eyeball that recall/ndcg for the two faiss rows are within ~0.5pp of `torch_fullscan` (flat) and `silvertorch` (IVF) respectively.

2. **Batch-size sanity** (run on smoke with the default `[1, 8, 16]` once):
   - GPU rows: `median_ms(bs=8)` should be < 8 × `median_ms(bs=1)` (sub-linear scaling — the proof we want from this benchmark).
   - CPU faiss rows: `median_ms(bs=8)` should be ≈ 8 × `median_ms(bs=1)` (near-linear, single-thread, no SIMD batching benefit on flat IP).
   - Quality columns are identical across the three bs rows of the same `(algo, k)` cell.

3. **Reproducibility check**: run smoke twice with the same seed; diff the two output JSONs. Quality columns must be byte-identical; latency columns may drift within ~5%.

4. **Subset run**: `--algorithms faiss_flat_ip faiss_ivf_flat` to validate the new baselines in isolation.

5. **Memory sanity**: in the smoke output, `index_mem_mib` for `faiss_flat_ip` and `faiss_ivf_flat` must be `0` (CPU-resident), and the `device` field must be `"cpu"` for those rows.

6. **Full 500M run** (gated by user, ~hour each on the d128 / d64 / d256 checkpoints, multiplied by 3 for the bs sweep).
