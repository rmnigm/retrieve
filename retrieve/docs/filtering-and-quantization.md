# Filtering & quantization

Two topics that cut across the modules: attribute-filtered retrieval (only return items matching a
structured predicate) and the standalone quantization utilities.

## Attribute filtering

### Clause data layout

A filter evaluates a conjunction of **clauses** (attribute keys) against each item:

- `item_clause_attrs: [N, C, A_max]` int64 — for each item, up to `A_max` values per clause,
  `-1`-padded. `C` is the number of clauses.
- `query_clause_attrs: [B, C]` int64 — one queried value per clause per query; `-1` means the
  clause is **inactive** (always passes).
- `clause_is_reverse: [C]` bool — invert (NOT) a clause. Only `ExactAttributeFilter` /
  `SilverTorch(filter="exact")` support reverse; `BloomFilter` is paper-strict and rejects it.

Semantics: a query passes an item iff, for **every** active clause, the query value appears among
that item's values (OR over the `A_max` slot, AND across clauses).

### Two ways to filter

1. **SilverTorch inline** — the predicate is fused into the probe+score kernel; nothing
   intermediate touches HBM.
2. **LiNR decoupled** — a standalone `FilterModule` computes a mask or candidate set, which you
   pass into a LiNR module.

### SilverTorch inline filtering

Pass the item attributes at `register_index` and the query attributes at `forward`:

```python
import torch
from retrieve import SilverTorch

item_embs = torch.randn(N, 128, device="cuda")
item_attrs = ...  # [N, C, A_max] int64, -1 padded

# Bloom: false-positive-tolerant, needs m_bits (power of 2, mult. of 64) and k_hash.
ann = SilverTorch(k=10, n_lists=1024, n_probe=16,
                  filter="bloom", m_bits=1024, k_hash=4).cuda()
ann.register_index(item_embs, item_clause_attrs=item_attrs)

query = torch.randn(4, 128, device="cuda")
query_attrs = ...  # [B, C] int64, -1 = inactive clause
topk_ids, topk_scores = ann(query, query_clause_attrs=query_attrs)
```

For exact-clause filtering (no false positives, supports reverse):

```python
ann = SilverTorch(k=10, n_lists=1024, n_probe=16, filter="exact").cuda()
ann.register_index(item_embs, item_clause_attrs=item_attrs,
                   clause_is_reverse=clause_is_reverse)   # [C] bool, optional
topk_ids, topk_scores = ann(query, query_clause_attrs=query_attrs)
```

`bloom` trades hash flexibility for a small false-positive rate; `exact` is bandwidth-cheaper at
small `C × A_max` and exact. Calling `forward` with `query_clause_attrs=None` skips filtering and
runs plain ANN.

### LiNR decoupled filtering

A `FilterModule` (`BloomFilter`, `ExactAttributeFilter`) exposes three evaluation paths over a
registered item set:

- `evaluate_mask(query_clause_attrs) -> [B, N]` bool — dense.
- `evaluate_indices(query_clause_attrs) -> ([B, P] int64, [B] int64)` — a compact candidate set
  `(ids, counts)`; within-row order is unspecified.
- `evaluate_subset(query_clause_attrs, candidate_ids) -> [B, P]` bool — re-check an existing
  candidate set.

Build a filter, then feed its output to a LiNR module:

```python
from retrieve import ExactAttributeFilter, PostfilterKNN, PrefilterKNN

filt = ExactAttributeFilter().cuda()
filt.register_index(item_attrs, clause_is_reverse=clause_is_reverse)

# Dense path: mask → PostfilterKNN (or FullScanKNN)
mask = filt.evaluate_mask(query_attrs)               # [B, N] bool
knn = PostfilterKNN(k=10).cuda(); knn.register_index(item_embs)
ids, scores = knn(query, mask=mask)

# Sparse path: candidate set → PrefilterKNN (or OneBitKNN)
cand_ids, counts = filt.evaluate_indices(query_attrs)   # ([B, P], [B])
pre = PrefilterKNN(k=10).cuda(); pre.register_index(item_embs)
ids, scores = pre(query, candidate_ids=cand_ids, counts=counts)
```

`BloomFilter(m_bits, k_hash)` registers via `register_index(item_clause_attrs)` (use keyword args —
its parameter order differs from `ExactAttributeFilter`). Bloom raises if `clause_is_reverse` has
any `True`.

Composition helpers (exported at top level):

- `combine_masks(*masks)` — element-wise AND of `[B, N]` masks; `None` inputs ignored.
- `combine_indices(filters, query_clause_attrs)` — sparse cascade across multiple filters; order
  them most-selective first.
- `post_filter_topk(topk_ids, post_mask)` — apply a filter **after** top-K; returns
  `(ids, counts)` with filtered positions set to `-1`.

For deeper filter internals and kernel behavior, see the repo-level
`docs/system/filtering.md`.

## Quantization utilities

The 1-bit and INT8 modules quantize internally, but the building blocks are exported for custom
index builds:

| Function | Returns | Used by |
| --- | --- | --- |
| `quantize_int8(embs)` | `(codes [N, D] int8, scales [N] fp32)` — per-row symmetric; `embs ≈ codes.float() * scales[:, None]` | INT8 ANN |
| `quantize_oporp_1bit(embs, seed=0, k_bits=0)` | `(bits [N, k_bits//64] int64, signs [D] int8, perm [D] int64)` — Sign-OPORP; `k_bits=0` → `D` | `OneBitKNN` |
| `quantize_simhash_1bit(embs, k_bits, seed=0)` | `(bits [N, k_bits//64] int64, r [k_bits, D] fp32)` — SimHash; `k_bits` may exceed `D` | `SimHashKNN` |

For the 1-bit schemes, the query is projected with the **same** `(signs, perm, k_bits)` or `r`,
and similarity is `k_bits - 2 * popcount(query_bits ^ item_bits)`. The modules handle this for you;
use the standalone functions only when assembling your own pipeline.

```python
import torch
from retrieve import quantize_oporp_1bit

embs = torch.randn(100_000, 128, device="cuda")
bits, signs, perm = quantize_oporp_1bit(embs)   # bits: [100000, 2] int64 at k_bits=128
```
