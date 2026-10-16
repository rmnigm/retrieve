---
title: filtering
created: 2026-09-26
updated: 2026-10-15
type: concept
tags: [filtering, library]
sources: [retrieve/src/retrieve/modules/filters.py, retrieve/src/retrieve/functional.py]
---

# Filtering in `retrieve`

This repo reproduces two retrieval papers: SilverTorch (IVF + INT8 ANN with a bloom or exact
attribute filter inline) at
[retrieve/src/retrieve/modules/silvertorch.py](../../retrieve/src/retrieve/modules/silvertorch.py),
on Meta's official code, adapted (`backend="cuda"`; its bloom is Meta's bundle bloom with a configured width and an
exact recheck, its exact mode runs inside Meta's search op;
[decisions](../../research/decisions.md#goal-and-scheduling)); and LiNR (V1 fp16 dense / V2 sparse
pre-filter / V3 1-bit OPORP) at
[retrieve/src/retrieve/modules/knn.py](../../retrieve/src/retrieve/modules/knn.py) and
[bit_knn.py](../../retrieve/src/retrieve/modules/bit_knn.py). The standalone filters are
[retrieve/src/retrieve/modules/filters.py](../../retrieve/src/retrieve/modules/filters.py); their
kernels are `retrieve.ops.triton` (with pure-torch twins in `retrieve.ops.reference`), the
composition helpers `retrieve.functional`. This brief
covers **filtering only**; KNN scoring, quantization, and training are out of scope here.

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
| `FilterModule` ABC | [interfaces.py](../../retrieve/src/retrieve/interfaces.py) | three native paths: `evaluate_mask`, `evaluate_indices`, `evaluate_subset`, plus `mask_scores(scores, q)` (dense scores with failing items at `-inf`; the ABC's default is `where(evaluate_mask, scores, -inf)`, `ExactAttributeFilter` runs it fused as `clause_mask_scores`); `forward` aliases `evaluate_mask`. Unified signature `register_index(item_clause_attrs, *, clause_is_reverse=None)` — keyword-only after the first arg on the ABC and both subclasses |
| `ExactAttributeFilter` (exact, supports reverse) | [modules/filters.py](../../retrieve/src/retrieve/modules/filters.py) | `FilterModule` subclass; native `evaluate_mask` via `clause_mask` kernel; native `evaluate_indices` via `clause_compact` kernel; `evaluate_subset` via gather + `clause_subset_match` (the module-level torch predicate SilverTorch's eager exact path also uses) |
| LiNR clause Triton kernels | [ops/triton/clause_compact.py](../../retrieve/src/retrieve/ops/triton/clause_compact.py), [clause_mask.py](../../retrieve/src/retrieve/ops/triton/clause_mask.py) | fused eval + stream compaction (compact); fused eval emitting `[B, N]` bool (mask). No `[B, N, C, A_max]` intermediate either way |
| `combine_masks` / `combine_indices` | [functional.py](../../retrieve/src/retrieve/functional.py) | mask-AND composition; sparse cascade via `evaluate_subset` |
| SilverTorch's filters (Meta's ops, `filter_mode`) | [ops/official/adapter.py](../../retrieve/src/retrieve/ops/official/adapter.py) | `filter_mode="bloom"` is **Meta's bloom index** (`bloom_index_build` over `(clause, value)` features, murmur3, bundles of 2048 docs; on the adapted code at the configured `m_bits` width, its survivors rechecked exactly so no false positive is returned) queried through Meta's expression DSL (`"0:v AND 1:w"`, `NOT`, `""` = all): a different hash from ours, never bit-compared, matched by FPR / memory instead. `filter_mode="exact"` is the clause test inside the adapted search op, or our `clause_mask_packed` as the published scorer's `filtering_bit_mask`. See [kernels.md](kernels.md#silvertorch-on-metas-ops) |

## Item-range masks

`evaluate_mask(query_clause_attrs, start=0, end=None) → [B, end - start]
bool` evaluates items `[start, end)` only; the default is the whole catalog. The filter hands the
op a row slice of its item table (`item_clause_attrs[start:end]`): a contiguous view, so
`clause_mask` is unchanged and no `[B, N]` mask exists for a range on either backend (`torch`
materializes `[B, end - start, C, A_max]`). The predicate is per item, so the range result is
bit-identical to `evaluate_mask(q)[:, start:end]`, empty ranges included (an empty grid launches
nothing). The consumer is the harness's chunked oracle, which walks the catalog in ranges where a
`[B, N]` mask does not fit (10M items × 768). Gate: `test_evaluate_mask_range_*` in
[`test_filters.py`](../../retrieve/tests/correctness/test_filters.py).

## LiNR hosts the exact filter; SilverTorch's filters are Meta's

LiNR's `forward` is decoupled from the filter: it accepts `mask: [B, N]` or `candidate_ids: [B,
P]` as input ([modules/knn.py](../../retrieve/src/retrieve/modules/knn.py),
[modules/bit_knn.py](../../retrieve/src/retrieve/modules/bit_knn.py)), and the `LiNRV1`–`LiNRV3`
modules hold a `FilterModule` that produces them. `ExactAttributeFilter` is that filter; our own
bloom filter for the LiNR side (the paper's bloom option) was deleted with the Triton SilverTorch
([kernels.md](kernels.md#deleted-backends)): the study runs LiNR with the exact filter only, and
the bloom it measures is Meta's, inside SilverTorch. SilverTorch's predicates are Meta's,
selected by `filter_mode` (the bloom index and its expression DSL; the clause test inside the
adapted search op), so a new predicate there is a change to Meta's ops; on the LiNR side a new
predicate is a new `FilterModule` subclass.

## Caller patterns

```python
from retrieve import ExactAttributeFilter, PostfilterKNN, PrefilterKNN
from retrieve.functional import combine_indices, combine_masks

ef = ExactAttributeFilter(backend="triton").to("cuda")
ef.register_index(item_attrs, clause_is_reverse=is_reverse)
q = ef.prepare_queries(qa)

# Mask path: feed PostfilterKNN / OneBitKNN (combine_masks ANDs several filters' masks).
ids, scores = postfilter_knn(query, mask=ef.evaluate_mask(q))

# Candidate-id path: feed PrefilterKNN (combine_indices cascades several filters, most-selective
# first).
cand_ids, counts = ef.evaluate_indices(q)
ids, scores = prefilter_knn(query, candidate_ids=cand_ids, counts=counts)
```

**When a filter leaves fewer than K survivors** the returned top-K row is
padded: the dead slots carry score `-inf` and id `-1`, on every backend, on the
standalone-filter paths above and on SilverTorch's `filter_mode="bloom" / "exact"` path
([kernels.md](kernels.md#score-conventions)).
Callers must drop those slots rather than treat `-1` as an item: a hit-based
metric that only compares ids would otherwise score a padded slot against a
target id of `-1`.

**Candidate order is part of the contract.** `evaluate_indices` returns each
row's survivors in **ascending item order** on every backend: the
`compact_mask` (stable argsort) order on `torch`, and the same order from the fused
`clause_compact` kernel, which counts per tile (stashing each tile's survivors in its
own slot range), scans, and moves every run to its fixed row offset
([kernels.md](kernels.md#clause_compact--fused-clause-eval--stream-compaction)).
A row base claimed with an atomic would order a row by tile completion,
and the downstream tie-breakers — `PrefilterKNN`'s top-k, and
`OneBitKNN`'s heavily tied Hamming ranking, which decides pool membership
— would make `linr_v2` / `linr_v3` on `triton` irreproducible run to run
(2e-6 / 7e-5 in quality). The same candidate order every call is what
makes a quality number repeatable.

## Out of scope

- DSL / nested AND/OR/NOT / RPN walker — kept out of the filter API. If
  full predicate-tree support is ever needed, it would be a host-side
  parser feeding multiple `evaluate_mask` calls into `combine_masks`.
- Our own bloom filter for LiNR: deleted ([kernels.md](kernels.md#deleted-backends)); the
  bloom in the study is Meta's, inside SilverTorch.
- Range / prefix / numeric predicates — neither paper supports them; both
  are equality-on-int64-hash.
