<!-- claude code generated file -->

# Filtering in `retrieve` — brief for another agent

> Previously: `retrieve/docs/filtering.md` (originally `retrieve/docs/FILTERING.md`).

This repo reproduces two retrieval papers — SilverTorch (IVF + INT8 + Bloom)
at [retrieve/src/retrieve/layers/silvertorch/](../../retrieve/src/retrieve/layers/silvertorch/)
and LiNR (V1 dense / V2 sparse pre-filter / V3 1-bit OPORP) at
[retrieve/src/retrieve/layers/linr/](../../retrieve/src/retrieve/layers/linr/). This brief
covers **filtering only**; KNN scoring, quantization, and training are out of
scope here.

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

## Implementation status in this repo

| component | path | notes |
|---|---|---|
| LiNR clause filter | [layers/utils/filters.py](../../retrieve/src/retrieve/layers/utils/filters.py) | `evaluate_mask → [B, N]` bool; `evaluate_indices → (pos_idx, counts)` |
| LiNR clause Triton kernel | [kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) | fused eval + stream compaction; richer than what the paper describes |
| Bloom build helpers | [layers/silvertorch/bloom.py](../../retrieve/src/retrieve/layers/silvertorch/bloom.py) | `_build_signatures`, `_generate_seeds`; private to the silvertorch package |
| Bloom standalone Triton kernel | [kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py) | `(qb & sigs) == qb` → `[B, N]` bool; **no consumer in the serving path** |
| Bloom fused into score kernel | [kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py) | conjunctive only (single `QB`), part of co-designed Algorithm 1 |
| `FilterModule` ABC | [interfaces.py:8-19](../../retrieve/src/retrieve/interfaces.py#L8-L19) | defined; no concrete filter subclasses it yet |
| `combine_masks(clause_mask, external_mask)` | proposed in [filtering-api.md](../plans/filtering-api.md) | missing |

## Key observation: LiNR can host *both* filter types

LiNR's `forward` is decoupled from the filter — it accepts `mask: [B, N]`
or `candidate_ids: [B, P]` as input
([v1.py](../../retrieve/src/retrieve/layers/linr/v1.py),
[v2.py](../../retrieve/src/retrieve/layers/linr/v2.py),
[v3.py](../../retrieve/src/retrieve/layers/linr/v3.py)). Any filter that produces
those shapes plugs in.

Asymmetry: SilverTorch fuses Bloom *into* its score kernel
(`codesigned_probe_score`), so swapping its filter would require a new
fused kernel, not a wrapping. The natural extension only goes one
direction — bring Bloom into the LiNR side as an alternative
`FilterModule`.

## Proposed work — single PR

### `BloomFilter` as a `FilterModule` for LiNR (~50 LoC + test)

**Scope: conjunctive only.** AND of required `(feature, value)` predicates
via `(qb & sigs) == qb`. **No DSL, no RPN walker, no NOT inside Bloom.**
Reverse / NOT remains the `ClauseIndex` job; the two filters are
complementary, not overlapping.

Lift the existing helpers out of the silvertorch private namespace and
wrap the standalone `bloom_match` kernel:

```python
class BloomFilter(FilterModule):
    def register_index(self, item_attrs, item_embs=None):
        seeds = _generate_seeds(self.k_hash, device=item_attrs.device)
        self.register_buffer("hash_seeds", seeds)
        self.register_buffer(
            "bloom_sigs",
            _build_signatures(item_attrs, seeds,
                              self.m_bits, self.k_hash, self.W),
        )

    def forward(self, query_attrs):                      # [B, t] int64
        qb = _build_signatures(
            query_attrs.unsqueeze(-1),
            self.hash_seeds, self.m_bits, self.k_hash, self.W,
        )
        return bloom_match(qb, self.bloom_sigs)           # [B, N] bool
```

Caller pattern (no LiNR kernel changes):

```python
mask = bloom_filter(query_features)         # OR
mask = clause_filter.evaluate_mask(query_clause_attrs)
ids, scores = linr_v2(query, mask=mask)
```

Side benefits:

- Retires the "`bloom_match` is dead code" footnote in
  [kernels.md](kernels.md) — it's the right primitive, just had no consumer.
- Make `ClauseIndex` officially subclass `FilterModule` (it almost does).
- `combine_masks(clause_mask, external_mask)` — the missing helper —
  becomes the natural way to AND two filter outputs together.

## What a thesis bench should show

Run both filters against any LiNR backend (V1 / V2 / V3) on two predicate
shapes:

1. **Narrow, exact** (geo + company + title, ≤ 4 attrs/clause) — expect
   `ClauseIndex` wins on latency, exact recall.
2. **Wide, free-form** (item-has-any-of-N tags, no fixed schema) — expect
   `BloomFilter` wins on latency, recall asterisk for hash collisions.

Neither paper does this head-to-head. It's the natural empirical
contribution for "two filter types under one retrieval engine."

## Out of scope

- DSL / nested AND/OR/NOT / RPN walker — explicit decision, keeps the PR
  small. If full predicate-tree support is ever needed, it would be a
  host-side parser feeding multiple `bloom_match` calls.
- NOT inside Bloom — exact reverse semantics live in `ClauseIndex` and
  stay there.
- Range / prefix / numeric predicates — neither paper supports them; both
  are equality-on-int64-hash.
- Replacing Bloom with `ClauseIndex` inside `codesigned_probe_score` —
  would require a new fused kernel, not a wrapping.
- Filter quality / training-time concerns.
