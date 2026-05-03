<!-- claude code generated file -->

# Filtering in `retrieve`

This repo reproduces two retrieval papers — SilverTorch (IVF + INT8 + Bloom)
at [retrieve/src/retrieve/layers/silvertorch/](../../retrieve/src/retrieve/layers/silvertorch/)
and LiNR (V1 dense / V2 sparse pre-filter / V3 1-bit OPORP) at
[retrieve/src/retrieve/layers/linr/](../../retrieve/src/retrieve/layers/linr/).
Filtering is its own subpackage: [retrieve/src/retrieve/layers/filters/](../../retrieve/src/retrieve/layers/filters/).
This brief covers **filtering only**; KNN scoring, quantization, and training
are out of scope here.

## What each paper specifies

### SilverTorch (paper §4.1, §4.4) — Bloom index, runtime DSL

```
ANN_Index(user_emb)
  AND item_country = "US"
  AND (item_lang = "EN" OR item_lang = "ES")
```

Arbitrary nested AND/OR/NOT over `feature = value` predicates. Each item
carries an M-bit Bloom signature built by hashing every `(feature, value)`
pair K times; the leaf test against a per-query bitset `QB` is the subset
relation `(qb & sig) == qb`. Compound queries are parsed to RPN and
host-walked over a stack of per-leaf Bloom masks.

**Approximate** (hash collisions, accepted because OverArch filters them
downstream), schema-free, equality only.

### LiNR (paper §3.1) — fixed clause schema, no DSL

- `C` clauses pre-declared at index time, each with a reverse flag.
- Item: `[N, C, A_max]` int64, padded with `-1`.
- Query: `[B, C]` int64 (one attribute per clause; `-1` = inactive).
- Match: `AND_c (any item.attr_c == query.attr_c) [^reverse_c]`.

**Exact**, schema-bound, equality only. No nesting, no parens, no per-query
boolean composition.

## Implementation status

| component | path | notes |
|---|---|---|
| `FilterModule` ABC | [interfaces.py](../../retrieve/src/retrieve/interfaces.py) | three native paths: `evaluate_mask`, `evaluate_indices`, `evaluate_subset`; `forward` aliases `evaluate_mask` |
| `ClauseIndex` (exact, supports reverse) | [layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py) | `FilterModule` subclass; native `evaluate_mask` via `clause_mask` kernel; native `evaluate_indices` via `clause_compact` kernel; `evaluate_subset` via gather + broadcast |
| `BloomFilter` (approximate, conjunctive) | [layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | `FilterModule` subclass; paper-strict (no reverse, no DSL); native `evaluate_mask` via `bloom_match` kernel; native `evaluate_indices` via `bloom_compact` kernel; `evaluate_subset` via gathered subset test |
| LiNR clause Triton kernels | [kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py), [clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) | fused eval + stream compaction (compact); fused eval emitting `[B, N]` bool (mask). No `[B, N, C, A_max]` intermediate either way |
| Bloom Triton kernels | [kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py), [kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py) | `(qb & sigs) == qb` → `[B, N]` bool (match); fused subset-test + stream compaction (compact). Consumed by `BloomFilter.evaluate_mask` / `evaluate_indices` on CUDA |
| Bloom fused into score kernel | [kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py) | conjunctive only (single `QB`), part of co-designed Algorithm 1; **separate path** from `BloomFilter` |
| `combine_masks` / `combine_indices` | [layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) | mask-AND composition; sparse cascade via `evaluate_subset` |

## Key observation: LiNR hosts *both* filter types

LiNR's `forward` is decoupled from the filter — it accepts `mask: [B, N]`
or `candidate_ids: [B, P]` as input
([v1.py](../../retrieve/src/retrieve/layers/linr/v1.py),
[v2.py](../../retrieve/src/retrieve/layers/linr/v2.py),
[v3.py](../../retrieve/src/retrieve/layers/linr/v3.py)). Any `FilterModule`
that produces those shapes plugs in. Both `ClauseIndex` and `BloomFilter`
do, and they compose via `combine_masks` / `combine_indices`.

Asymmetry: SilverTorch fuses Bloom *into* its score kernel
(`codesigned_probe_score`), so swapping its filter would require a new
fused kernel, not a wrapping. The natural extension only goes one
direction — bring Bloom into the LiNR side as an alternative
`FilterModule`. That is what `BloomFilter` is.

## Caller patterns

```python
from retrieve import (
    BloomFilter, ClauseIndex,
    LiNR_V1_Triton, LiNR_V2_Triton, LiNR_V3_Triton,
    combine_indices, combine_masks,
)

ci = ClauseIndex().to("cuda")
ci.register_index(item_attrs, clause_is_reverse=is_reverse)

bf = BloomFilter(m_bits=1024, k_hash=5).to("cuda")
bf.register_index(item_attrs)

# V1 / V3 mask path — combine exact + approximate.
mask = combine_masks(ci.evaluate_mask(qa), bf.evaluate_mask(qa))
ids, scores = linr_v1(query, mask=mask)

# V2 / V3 candidate-id path — sparse cascade, most-selective filter first.
cand_ids, counts = combine_indices([ci, bf], [qa, qa])
ids, scores = linr_v2(query, candidate_ids=cand_ids, counts=counts)
```

## Out of scope

- DSL / nested AND/OR/NOT / RPN walker — kept out of the filter API. If
  full predicate-tree support is ever needed, it would be a host-side
  parser feeding multiple `evaluate_mask` calls into `combine_masks`.
- NOT inside Bloom — exact reverse semantics live in `ClauseIndex` and
  stay there.
- Range / prefix / numeric predicates — neither paper supports them; both
  are equality-on-int64-hash.
- Replacing Bloom with `ClauseIndex` inside `codesigned_probe_score` —
  would require a new fused kernel, not a wrapping.
