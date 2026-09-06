# Modules: selection guide & API reference

All retrieval modules share the `register_index(item_embs) → forward(queries) → (topk_ids,
topk_scores)` lifecycle described in [`getting-started.md`](getting-started.md). This page covers
which one to reach for and the exact signatures.

## Choosing a module

Two families:

- **LiNR** (`FullScanKNN`, `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`, `OneBitKNN`,
  `SimHashKNN`) — score the corpus directly (full scan or over a candidate set). Simple, exact or
  near-exact, no index-build step. Best at small-to-medium `N` or when you already have a
  candidate set from an upstream filter.
- **SilverTorch** — IVF (clustered) + INT8 ANN. Adds an index-build (k-means) step and approximate
  recall, but scales to large `N` by only probing a few clusters per query.

| Module | Scoring | Memory vs fp16 | Notes |
| --- | --- | --- | --- |
| `FullScanKNN` | exact dot product | 1× (fp32) | Reference / small-N. |
| `PostfilterKNN` | fp16 dot product | 1× (fp16) | Dense scan + optional boolean mask. |
| `PostfilterKNNInt8` | INT8 dot product | 0.5× | Dense scan, int32 end-to-end. |
| `PrefilterKNN` | fp16 dot product | 1× (fp16) | Scores only a candidate set. |
| `OneBitKNN` | Hamming (1-bit) | ~1/16× | Sign-OPORP quantization. |
| `SimHashKNN` | Hamming (1-bit) | ~1/16× | SimHash; `k_bits` can exceed `D`. |
| `SilverTorch` | INT8 ANN over IVF | 0.5× + centroids | Scales to large `N`; optional fused filter. |

### Backend

Most modules accept `backend="triton"` (default) or `backend="torch"`. The Triton path runs fused
kernels; the torch path is pure PyTorch (still GPU) and is `torch.compile`-friendly. Results are
equivalent. `PostfilterKNN` / `PostfilterKNNInt8` accept the flag for API symmetry but always run
the same code (cuBLAS already covers their case). `SilverTorch` alone also accepts
`backend="cuda"` (CUDA C++, JIT-built on first forward, needs `nvcc` + `ninja`) and
`backend="cute"` (the same kernels in the CuTe DSL, `pip install "torchretrieve[cute]"`); both
are bit-identical to the Triton path in every `filter_mode`. `SilverTorch` also accepts
`backend="official"`: Meta's own `meta-recsys/silvertorch` kernels (`torch.ops.st.*`, the
`official` extra — built from source, needs a CUDA toolkit matching your torch) for the scoring
and bloom phases, with our k-means and quantization in front; it is the reference the Triton
kernels are checked against, eager-only (`torch.compile` raises), and its bloom mode is Meta's
bloom index rather than ours (`m_bits` is optional; `official=OfficialConfig(...)` carries
`b_multiplier`, `hash_k`, the `"int32"` bit-exact vs `"fp16"` serving score path, and the
partial-vs-full bloom path). Any other module given `"cuda"`, `"cute"` or `"official"` runs its
torch path.

## LiNR modules

Constructor → `register_index` → `forward`. Unless noted, `item_embs` is `[N, D]`, `query` is
`[B, D]`, and the return is `([B, k] int64 ids, [B, k] scores)`. The score dtype follows the
module's arithmetic: `FullScanKNN` returns the input dtype (fp32 for fp32 inputs), `OneBitKNN` /
`SimHashKNN` return fp32, `PostfilterKNN` and `PostfilterKNNInt8` return fp16, and `PrefilterKNN`
returns fp16 on the `torch` backend and fp32 on the `triton` backend (and for an empty candidate
set). Ranking is what the layers promise; cast at the boundary if you need one dtype.

### `FullScanKNN(k)`
- `forward(query, mask=None, candidate_ids=None)`
- Exhaustive matmul + top-K. Optional `mask: [B, N] bool` (filtered ids become `-1`) or
  `candidate_ids: [B, P]` to score only a candidate set (`-1` entries are padding: never scored,
  never returned; a row with fewer than `min(k, P)` real candidates ends in `-1` / `-inf`).

### `PostfilterKNN(k, backend="triton")`
- `forward(query, mask=None)`
- Dense fp16 dot product + top-K, with an optional `mask: [B, N] bool` applied before selection.

### `PostfilterKNNInt8(k, backend="triton")`
- `forward(query, mask=None)`
- Same as `PostfilterKNN` but INT8 (one global scale), int32 end-to-end. Half the index memory.

### `PrefilterKNN(k, backend="triton")`
- `forward(query, candidate_ids=None, counts=None)`
- Scores only `candidate_ids: [B, P]` (per-row valid width bounded by `counts: [B]`, default all
  `P`), then top-Ks back to global ids. Without `candidate_ids` it falls back to a dense full
  matmul. Pair it with a `FilterModule` (see the filtering guide).

### `OneBitKNN(k, seed=0, backend="triton", k_bits=0)`
- `forward(query, candidate_ids=None, counts=None)`
- Sign-OPORP 1-bit quantization; Hamming similarity. `k_bits=0` resolves to `D`. Optional
  `candidate_ids` / `counts` restrict scoring to a candidate set.

### `SimHashKNN(k, k_bits, seed=0, backend="triton")`
- `forward(query, candidate_ids=None, counts=None)`
- SimHash 1-bit quantization (Gaussian projection). `k_bits` is required and **may exceed `D`** to
  trade memory for recall — the one knob the OPORP family can't reach.

## SilverTorch

```python
SilverTorch(k, n_lists, n_probe, filter_mode="none",
            m_bits=None, k_hash=None, n_iter=10, seed=0, backend="triton")
```

- `register_index(item_embs, item_clause_attrs=None, clause_is_reverse=None)`
- `forward(query, query_clause_attrs=None, candidate_ids=None)` — `candidate_ids: [B, P]`
  (original ids, `-1` = padding) switches to a pure re-rank of those ids with no filter; pads are
  never scored or returned, a row with fewer than `min(k, P)` real candidates ends in
  `-1` / `-inf`, and passing `query_clause_attrs` alongside raises.

Parameters:

- `n_lists` — number of IVF clusters (k-means runs `n_iter` iterations at register time).
- `n_probe` — clusters scanned per query (≤ `n_lists`). Higher = more recall, more work.
- `filter_mode` — `"none"`, `"bloom"`, or `"exact"`. `"bloom"` requires `m_bits` (power of 2,
  multiple of 64) and `k_hash`. The filter is fused into the probe+score kernel — see the
  [filtering guide](filtering-and-quantization.md).

Constraint: `n_probe * max_cluster_size >= k` (raised at `register_index` otherwise).

`build_silvertorch(item_embs, k, *, n_lists, n_probe, filter_mode="none", ...)` is a convenience
that constructs the module and calls `register_index` in one step.

## KMeansTorch

```python
KMeansTorch(n_lists, n_iter=10, seed=0)
centroids, assignments = KMeansTorch(n_lists=1024).fit(item_embs)
```

Lloyd's k-means with chunked assignment; returns `(centroids [n_lists, D], assignments [N])`.
`SilverTorch` uses it internally — call it directly only if you want the clustering for your own
index build.
