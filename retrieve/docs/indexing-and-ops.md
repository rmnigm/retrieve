# Indexing helpers, functional glue and the ops

Below the modules sit three namespaces. `retrieve.indexing` is index-*build* math (what
`register_index` runs); `retrieve.functional` is query-time torch glue that is not a kernel;
`retrieve.ops.<backend>` are the registered kernels the modules call, one namespace per backend
with the same op names and signatures. Reach for them when assembling your own pipeline; the
modules use them for you.

## `retrieve.indexing`

| name | signature | used by |
| --- | --- | --- |
| `KMeans` | `KMeans(n_lists, n_iter=10, seed=0, init="random")`; `.fit(embs) -> (centroids [n_lists, D], assignments [N])`; `KMeans.assign(embs, centroids) -> [N]` | `SilverTorch` (`kmeans_init`) |
| `csr_layout` | `(assignments, n_lists) -> (sort_perm, inv_perm, cluster_offsets [n_lists + 1], cluster_sizes)` — one stable argsort; `item_codes[sort_perm]` is the cluster-sorted table | `SilverTorch` (every backend) |
| `probe_width` | `(cluster_sizes, n_probe) -> int` — the sum of the `n_probe` largest clusters, the scorer's output width | `SilverTorch` |
| `quantize_int8` | `(embs) -> (codes [N, D] int8, scales [N] fp32)` — per-row symmetric; `embs ≈ codes.float() * scales[:, None]` | `SilverTorch` (queries) |
| `quantize_int8_global` | `(embs) -> (codes [N, D] int8, scale: float)` — one global scale | `SilverTorch` (items) |
| `quantize_oporp_1bit` | `(embs, seed=0, k_bits=0) -> (bits [N, k_bits // 64] int64, signs [D] int8, perm [D] int64)` — Sign-OPORP; `k_bits=0` → `D` | `OneBitKNN` |
| `project_oporp_1bit_query` | `(query, signs, perm, k_bits) -> [B, k_bits // 64] int64` | `OneBitKNN` |
| `quantize_simhash_1bit` | `(embs, k_bits, seed=0) -> (bits [N, k_bits // 64] int64, r [k_bits, D] fp32)` — SimHash; `k_bits` may exceed `D` | `SimHashKNN` |
| `project_simhash_1bit_query` | `(query, r) -> [B, k_bits // 64] int64` | `SimHashKNN` |


`KMeans.fit` is deterministic: same `seed` and same input give bit-identical centroids and
assignments on repeated calls, on CPU and on CUDA, with either `init`. The centroid update sums
each cluster with a float64 one-hot GEMM rather than `index_add_`, whose float atomics reduce in
scheduling order and are not reproducible on a GPU. `init="kmeans++"` seeds by D² sampling (one
pass over the index per centroid, no host sync per step) instead of `n_lists` random rows; it is
opt-in — `SilverTorch(kmeans_init="kmeans++")`.

For the 1-bit schemes, the query is projected with the **same** `(signs, perm, k_bits)` or `r`,
and similarity is `k_bits - 2 * popcount(query_bits ^ item_bits)`.

```python
import torch
from retrieve.indexing import KMeans, csr_layout, probe_width, quantize_oporp_1bit

embs = torch.randn(100_000, 128, device="cuda")
centroids, assignments = KMeans(n_lists=1024).fit(embs)
sort_perm, inv_perm, offsets, sizes = csr_layout(assignments, 1024)
width = probe_width(sizes, n_probe=16)          # the scorer's [B, width] output
bits, signs, perm = quantize_oporp_1bit(embs)   # bits: [100000, 2] int64 at k_bits=128
```

## `retrieve.functional`

| name | what it does |
| --- | --- |
| `two_level_topk(scores, k)` | `torch.topk` up to ties, in two levels on wide rows (block maxima, then the k best blocks' items) |
| `masked_topk(scores, k, valid=None, gather_ids=None, pad_to_k=True)` | the shared top-k epilogue: mask to `-inf`, top-k (`two_level_topk`), map local → global ids, `-1` where the score is not finite |
| `counts_to_valid(counts, p)` | `[B]` counts → `[B, P]` prefix mask |
| `compact_mask(mask)` | `[B, N]` bool → `(ids [B, N], counts [B])`: the first `counts[b]` ids of each row are the passing items |
| `combine_masks(*masks)` | element-wise AND of `[B, N]` masks; `None` inputs ignored |
| `combine_indices(filters, query_clause_attrs)` | sparse cascade across several filters; order them most-selective first |
| `post_filter_topk(topk_ids, post_mask)` | apply a filter **after** top-k: `(ids, counts)` with filtered positions set to `-1` |
| `popcount_int64(x)` | per-element popcount of an int64 tensor |
| `clause_subset_match` | the exact predicate over gathered candidates (what `ExactAttributeFilter.evaluate_subset` runs) |

## `retrieve.ops`

```python
import retrieve.ops.triton              # registers torch.ops.retrieve.* (Triton)
retrieve.ops.triton.clause_mask(item_clause_attrs, clause_is_reverse, query_clause_attrs)
retrieve.ops.reference.clause_mask(...)   # same signature, pure torch, torch.equal to the above
retrieve.ops.official.ensure_loaded(); retrieve.ops.official.st.fused_kmean_ann(...)   # Meta's op
retrieve.ops.available_backends()       # ("triton", "torch"), plus "official" / "official-fork" where each loads
```

| op | `triton` / `reference` | consumer |
| --- | --- | --- |

| `clause_mask`, `clause_mask_scores`, `clause_compact` | exact clause predicate → `[B, N]` bool / masked scores / compact candidates | `ExactAttributeFilter` |
| `clause_mask_packed` | the clause predicate packed into Meta's `filtering_bit_mask` | `SilverTorch(backend="official", filter_mode="exact")` |
| `gemv_scores` | one query's dense fp32 scores in cuBLAS's order | `PostfilterKNN` at `B = 1` |
| `fused_masked_knn_topk` | gather + fp32-accumulated dot over candidate ids + top-k | `PrefilterKNN` |
| `oporp_1bit_match_topk_full`, `oporp_1bit_match_topk_indirect` | XOR + popcount + top-k, full scan / through candidate ids | `OneBitKNN`, `SimHashKNN` |

`retrieve.ops.official` is the adapter over Meta's `torch.ops.st.*` and the adapted
`torch.ops.stfork.*` (`official_probe_score`, `fork_search`, the bloom index and
expression-parser helpers, `OfficialConfig`, `OfficialMissing`); the ops themselves are reachable
verbatim as `retrieve.ops.official.st.<name>` / `.stfork.<name>` once `ensure_loaded(backend)`
ran. `retrieve.modules.official` re-exports Meta's four `silvertorch.modules` classes the same
lazy way. Both need the `official` extra. `retrieve.ops.tune` is the `tune-kernels` autotune CLI.
