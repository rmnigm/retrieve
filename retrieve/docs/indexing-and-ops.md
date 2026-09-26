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
| `generate_seeds`, `generate_clause_salt`, `build_signatures`, `build_query_signatures` | the bloom hash: `k_hash` seed pairs, the per-clause salt, item signatures `[N, m_bits // 64]` and query signatures `[B, m_bits // 64]` | `BloomFilter`, `SilverTorch(filter_mode="bloom")` |
| `build_transposed_sigs`, `build_query_bit_positions` | the transposed bloom index `[m_bits, ceil(N / 64)]` of row-wise signatures, and a query's set-bit positions `[B, C * k_hash]` (`-1` = inactive clause) | `SilverTorch(filter_mode="bloom")` on `triton` / `torch` |

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
| `masked_topk(scores, k, valid=None, gather_ids=None, pad_to_k=True)` | the shared top-k epilogue: mask to `-inf`, top-k, map local → global ids, `-1` where the score is not finite |
| `counts_to_valid(counts, p)` | `[B]` counts → `[B, P]` prefix mask |
| `compact_mask(mask)` | `[B, N]` bool → `(ids [B, N], counts [B])`: the first `counts[b]` ids of each row are the passing items |
| `combine_masks(*masks)` | element-wise AND of `[B, N]` masks; `None` inputs ignored |
| `combine_indices(filters, query_clause_attrs)` | sparse cascade across several filters; order them most-selective first |
| `post_filter_topk(topk_ids, post_mask)` | apply a filter **after** top-k: `(ids, counts)` with filtered positions set to `-1` |
| `popcount_int64(x)` | per-element popcount of an int64 tensor |
| `clause_subset_match`, `bloom_subset_match` | the two subset predicates over gathered candidates (what `evaluate_subset` runs) |

## `retrieve.ops`

```python
import retrieve.ops.triton              # registers torch.ops.retrieve.* (Triton)
retrieve.ops.triton.codesigned_probe_score(query, probe_ids, cluster_offsets, item_codes, sort_perm, global_scale, k, width)
retrieve.ops.reference.codesigned_probe_score(...)   # same signature, pure torch, torch.equal to the above
retrieve.ops.official.ensure_loaded(); retrieve.ops.official.st.fused_kmean_ann(...)   # Meta's op
retrieve.ops.available_backends()       # ("triton", "torch") or ("triton", "torch", "official")
```

| op | `triton` / `reference` | consumer |
| --- | --- | --- |
| `codesigned_probe_score`, `codesigned_probe_score_bloom`, `codesigned_probe_score_exact` | fused IVF probe + INT8 dot (+ bloom / exact predicate) + top-k | `SilverTorch` |
| `bloom_match`, `bloom_compact` | bloom subset test → `[B, N]` bool / compact candidates | `BloomFilter` |
| `clause_mask`, `clause_compact` | exact clause predicate → `[B, N]` bool / compact candidates | `ExactAttributeFilter` |
| `fused_masked_knn_topk` | gather + fp16 dot over candidate ids + top-k | `PrefilterKNN` |
| `oporp_1bit_match_topk_full`, `oporp_1bit_match_topk_indirect` | XOR + popcount + top-k, full scan / through candidate ids | `OneBitKNN`, `SimHashKNN` |

`retrieve.ops.official` is the adapter over Meta's `torch.ops.st.*` (`official_probe_score`,
the bloom index and expression-parser helpers, `OfficialConfig`, `OfficialMissing`); the ops
themselves are reachable verbatim as `retrieve.ops.official.st.<name>` once `ensure_loaded()`
ran. `retrieve.modules.official` re-exports Meta's four `silvertorch.modules` classes the same
lazy way. Both need the `official` extra. `retrieve.ops.tune` is the `tune-kernels` autotune CLI.
