# Modules: selection guide & API reference

All retrieval modules share the `register_index(item_embs) → forward(queries) → (topk_ids,
topk_scores)` lifecycle described in [`getting-started.md`](getting-started.md). This page covers
which one to reach for and the exact signatures.

## Choosing a module

Two families, and the paper's variants of the first:

- **LiNR** — score the corpus directly (full scan or over a candidate set). Simple, exact or
  near-exact, no index-build step. Best at small-to-medium `N` or when you already have a
  candidate set from an upstream filter. The paper's four variants ship as modules
  (`LiNRV1`–`LiNRV4`), each composing the primitives (`PostfilterKNN`, `PostfilterKNNInt8`,
  `PrefilterKNN`, `OneBitKNN`, `SimHashKNN`, `FullScanKNN`) with an optional filter; the
  primitives are public for your own compositions.
- **SilverTorch** — IVF (clustered) + INT8 ANN. Adds an index-build (k-means) step and approximate
  recall, but scales to large `N` by only probing a few clusters per query.

| Module | Scoring | Memory vs fp16 | Notes |
| --- | --- | --- | --- |
| `LiNRV1` | fp16 inputs, fp32 dot | 1× (fp16) | Dense scan, filter as a mask. |
| `LiNRV2` | fp16 inputs, fp32 dot | 1× (fp16) | Filter → candidates → exact rescoring; filter required. |
| `LiNRV3` | Hamming, then fp16-input fp32 dot | ~1/16× + 1× | 1-bit top-`candidate_pool`, then exact rescoring. |
| `LiNRV4` | INT8 dot product | 0.5× | Dense int8 scan, filter as a mask. |
| `SilverTorch` | INT8 ANN over IVF | 0.5× + centroids | Scales to large `N`; optional fused filter. |
| `FullScanKNN` | exact dot product | 1× (fp32) | Reference / small-N. |
| `PostfilterKNN` | fp16 inputs, fp32 dot | 1× (fp16) | Dense scan + optional boolean mask. |
| `PostfilterKNNInt8` | INT8 dot product | 0.5× | Dense scan, int32 end-to-end. |
| `PrefilterKNN` | fp16 inputs, fp32 dot | 1× (fp16) | Scores only a candidate set. |
| `OneBitKNN` | Hamming (1-bit) | ~1/16× | Sign-OPORP quantization. |
| `SimHashKNN` | Hamming (1-bit) | ~1/16× | SimHash; `k_bits` can exceed `D`. |

The public surface, in one import:

```python
from retrieve import (
    LinrBackend, SilverTorchBackend, RetrievalModule, FilterModule,   # interfaces
    SilverTorch, SilverTorchBuilder, OfficialConfig,                  # Algorithm 1
    LiNRV1, LiNRV2, LiNRV3, LiNRV4, LiNRBuilder,                      # LiNR paper variants
    PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, FullScanKNN,      # dense / sparse primitives
    OneBitKNN, SimHashKNN,                                            # 1-bit primitives
    BloomFilter, ExactAttributeFilter,                                # filters
)
import retrieve.modules.official   # Meta's BloomIndexSearchModule(+Builder), FilterQueryParserModule(+Builder)
```

### Backend

Most modules accept `backend="triton"` (default) or `backend="torch"`. The Triton path runs fused
kernels; the torch path is pure PyTorch (still GPU) and is `torch.compile`-friendly. Results are
equivalent. `PostfilterKNN` / `PostfilterKNNInt8` accept the flag for API symmetry but always run
the same code (cuBLAS already covers their case). `SilverTorch` alone also accepts
`backend="official"`: Meta's own `meta-recsys/silvertorch` kernels (`torch.ops.st.*`, the
`official` extra — built from source, needs a CUDA toolkit matching your torch) for the scoring
and bloom phases, with our k-means and quantization in front; it is the reference the Triton
kernels are checked against, eager-only (`torch.compile` raises), and its bloom mode is Meta's
bloom index rather than ours (`m_bits` is optional; `official=OfficialConfig(...)` carries
`b_multiplier`, `n_stored_hashes`, the `"int32"` bit-exact vs `"fp16"` serving score path, the
partial-vs-full bloom path, and `cache_plans` — set it `False` when timing so every forward pays
the expression parse). Any other module given `"official"` — or any string outside its
backend literal — raises `ValueError` at construction.

## LiNR variants

Each holds its filter (a `BloomFilter` or `ExactAttributeFilter`, see the
[filtering guide](filtering-and-quantization.md)) as the `filter` submodule, so `buffers()` and
`state_dict()` cover index and filter (the filter's buffers under `filter.`). All four:

- `register_index(item_embs, item_clause_attrs=None, clause_is_reverse=None)` — registers the
  index and, when attributes are given, the attached filter.
- `forward(query, query_clause_attrs=None) -> (ids [B, k] int64, scores [B, k])`; without
  `query_clause_attrs` the filter is skipped (V2 requires it: the filter is its candidate
  source).
- `k` is settable after `register_index`; `capturable` is `True` (a class attribute).

### `LiNRV1(k, *, filter=None, backend="triton")`
- `PostfilterKNN` + the filter's mask: dense dot product (fp16 inputs, fp32 scores), masked, top-k.

### `LiNRV2(k, *, filter, backend="triton")`
- `PrefilterKNN` over `filter.evaluate_indices`: the filter's compact candidate list, rescored
  exactly. `query_clause_attrs` is required.

### `LiNRV3(k, *, candidate_pool=5000, seed=0, filter=None, backend="triton")`
- `OneBitKNN(k=candidate_pool)` → `PrefilterKNN(k)`: 1-bit Hamming top-`candidate_pool` (over
  the filter's candidates when there is one), then exact rescoring of the survivors (fp16 inputs, fp32 scores).
  `set_query_params(candidate_pool=...)` changes the pool later (must be `<= N`).

### `LiNRV4(k, *, filter=None, backend="triton")`
- `PostfilterKNNInt8` + the filter's mask: dense int8 dot product, masked, top-k.

### `LiNRBuilder(variant, **kwargs)`

`variant` is `"v1"` … `"v4"`, `kwargs` the variant's constructor keywords. Then:

```python
v3 = (LiNRBuilder("v3", k=100, candidate_pool=8000)
      .set_item_embeddings(item_embs)
      .set_filter(ExactAttributeFilter(backend="triton"), item_attrs, clause_is_reverse)
      .set_backend("triton")
      .set_device("cuda")
      .build())                                   # == construct + register_index + .to(device)

same = (LiNRBuilder("v3", k=100, candidate_pool=8000)
        .set_filter(ExactAttributeFilter())        # a home for the saved filter.* buffers
        .set_state_dict(torch.load("v3.pt"))
        .build())                                  # prebuilt: no quantization, the buffers are loaded
```

`set_item_embeddings` and `set_state_dict` are mutually exclusive; `build()` without either
raises.

## LiNR primitives

Constructor → `register_index` → `forward`. Unless noted, `item_embs` is `[N, D]`, `query` is
`[B, D]`, and the return is `([B, k] int64 ids, [B, k] scores)`. The score dtype follows the
module's arithmetic: `FullScanKNN` returns the input dtype (fp32 for fp32 inputs), `OneBitKNN` /
`SimHashKNN` return fp32, `PostfilterKNN` and `PrefilterKNN` return fp32 on every backend (fp16
inputs, fp32 accumulation), and `PostfilterKNNInt8` returns fp16. Ranking is what the layers promise; cast at the boundary if you need one dtype.

### `FullScanKNN(k)`
- `forward(query, mask=None, candidate_ids=None)`
- Exhaustive matmul + top-K. Optional `mask: [B, N] bool` (filtered ids become `-1`) or
  `candidate_ids: [B, P]` to score only a candidate set (`-1` entries are padding: never scored,
  never returned; a row with fewer than `min(k, P)` real candidates ends in `-1` / `-inf`).

### `PostfilterKNN(k, backend="triton")`
- `forward(query, mask=None)`
- Dense dot product (fp16 inputs, fp32 scores) + top-K, with an optional `mask: [B, N] bool` applied before selection.

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
SilverTorch(k, n_lists, n_probe, filter_mode="none", m_bits=None, k_hash=None,
            n_iter=10, seed=0, kmeans_init="random", backend="triton", official=None)
```

- `register_index(item_embs, item_clause_attrs=None, clause_is_reverse=None)`
- `forward(query, query_clause_attrs=None, candidate_ids=None)` — `candidate_ids: [B, P]`
  (original ids, `-1` = padding) switches to a pure re-rank of those ids with no filter; pads are
  never scored or returned, a row with fewer than `min(k, P)` real candidates ends in
  `-1` / `-inf`, and passing `query_clause_attrs` alongside raises.

Parameters:

- `n_lists` — number of IVF clusters (k-means runs `n_iter` iterations at register time;
  `kmeans_init="kmeans++"` seeds it by D² sampling instead of random rows — opt-in, the
  default is `"random"`).
- `n_probe` — clusters scanned per query (≤ `n_lists`). Higher = more recall, more work.
  `set_query_params(n_probe=...)` changes it after `register_index`, with the same validation.
- `filter_mode` — `"none"`, `"bloom"`, or `"exact"`. `"bloom"` requires `m_bits` (power of 2,
  multiple of 64) and `k_hash`. The filter is fused into the probe+score kernel — see the
  [filtering guide](filtering-and-quantization.md).

Constraint: the `n_probe` largest clusters must hold at least `k` items together (the
scorer's static probe width; raised at `register_index` and by `set_query_params`
otherwise). `k` is a plain attribute, settable at any time.

After `register_index`, `build_timings` holds the seconds of the four build phases
(`kmeans_s`, `assemble_s`, `quantize_s`, `filter_s`; `{}` before it and on a prebuilt module).
`capturable` is `True` on `triton` / `torch` and `False` on `official` (eager-only).

### `SilverTorchBuilder(**kwargs)`

`kwargs` are `SilverTorch`'s. Then:

```python
ann = (SilverTorchBuilder(k=100, n_lists=1024, n_probe=24, filter_mode="bloom", m_bits=1024, k_hash=5)
       .set_item_embeddings(item_embs)
       .set_item_attributes(item_attrs, clause_is_reverse=None)
       .set_backend("official", official=OfficialConfig(cache_plans=False))
       .set_device("cuda")
       .build())                                   # == construct + register_index + .to(device)

same = SilverTorchBuilder(k=100, n_lists=1024, n_probe=24).set_state_dict(torch.load("st.pt")).build()
```

The `set_state_dict` path runs no k-means: the saved buffers are loaded and the load hook
re-derives the cached scalars, so `same` scores exactly like the module that was saved (the
state dict must come from the same backend — see the portability note in the system docs).
`set_item_embeddings` and `set_state_dict` are mutually exclusive; `build()` without either
raises. `OfficialConfig` is exported at the top level.

`KMeans`, the IVF layouts and the quantizers are documented in
[`indexing-and-ops.md`](indexing-and-ops.md).
