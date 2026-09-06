# Filtering in `retrieve`

This repo reproduces two retrieval papers — SilverTorch (IVF + INT8 ANN
with bloom or exact attribute filter fused inline) at
[retrieve/src/retrieve/layers/silvertorch/](../../retrieve/src/retrieve/layers/silvertorch/)
and LiNR (V1 fp16 dense / V2 sparse pre-filter / V3 1-bit OPORP, plus a
V4 int8 dense variant added beyond the paper) at
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
| `FilterModule` ABC | [interfaces.py](../../retrieve/src/retrieve/interfaces.py) | three native paths: `evaluate_mask`, `evaluate_indices`, `evaluate_subset`; `forward` aliases `evaluate_mask`. Unified signature `register_index(item_clause_attrs, *, clause_is_reverse=None)` — keyword-only after the first arg on the ABC and both subclasses |
| `ExactAttributeFilter` (exact, supports reverse) | [layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | `FilterModule` subclass; native `evaluate_mask` via `clause_mask` kernel; native `evaluate_indices` via `clause_compact` kernel; `evaluate_subset` via gather + `clause_subset_match` (the module-level torch predicate SilverTorch's eager exact path also uses) |
| `BloomFilter` (approximate, conjunctive) | [layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | `FilterModule` subclass; paper-strict (no reverse, no DSL) — `register_index` accepts an optional keyword-only `clause_is_reverse` and raises `ValueError` if any entry is `True`; native `evaluate_mask` via `bloom_match` kernel; native `evaluate_indices` via `bloom_compact` kernel; `evaluate_subset` via gather + `bloom_subset_match`. Hashes `(clause_idx, value)` pairs, **not** raw values — see ["Bloom hash keys"](#bloom-hash-keys-clause_idx-value) |
| Bloom hash math | [layers/filters/bloom_hash.py](../../retrieve/src/retrieve/layers/filters/bloom_hash.py) | the single home of the signature builders: public `generate_seeds` / `build_signatures` (index-side, chunked) / `build_query_signatures` (query-side, loop-free) plus `bloom_subset_match`; consumed by `BloomFilter` **and** `SilverTorch`'s fused bloom mode (no private cross-module imports). Hash math is pinned — changing it invalidates every persisted `bloom_sigs` buffer |
| LiNR clause Triton kernels | [kernels/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/filters/clause_compact.py), [clause_mask.py](../../retrieve/src/retrieve/kernels/filters/clause_mask.py) | fused eval + stream compaction (compact); fused eval emitting `[B, N]` bool (mask). No `[B, N, C, A_max]` intermediate either way |
| Bloom Triton kernels | [kernels/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py), [kernels/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py) | `(qb & sigs) == qb` → `[B, N]` bool (match); fused subset-test + stream compaction (compact). Consumed by `BloomFilter.evaluate_mask` / `evaluate_indices` on CUDA |
| Filter fused into SilverTorch score kernel | [kernels/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py), [codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) | Selected by `SilverTorch.filter_mode`. `"bloom"` → conjunctive bloom subset test (single `QB`), part of co-designed Algorithm 1; `"exact"` → exact AND-of-OR over `[N, C, A_max]` narrow attrs (reverse clauses supported), same `common.clause_pass` inner loop as `clause_mask` but fused into the probe-and-score path. Both are **separate paths** from the standalone `BloomFilter` / `ExactAttributeFilter` — SilverTorch shares the predicate math and hash builders, not the module classes. On `backend="official"` the `"exact"` predicate is our `clause_mask` packed into the official scorer's bit mask and `"bloom"` is Meta's own bloom index (see [kernels.md](kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend)). |
| `combine_masks` / `combine_indices` | [layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) | mask-AND composition; sparse cascade via `evaluate_subset` |
| Official SilverTorch filters (`backend="official"`) | [kernels/silvertorch/official.py](../../retrieve/src/retrieve/kernels/silvertorch/official.py) | `filter_mode="bloom"` is **Meta's bloom index** (`bloom_index_build` over `(clause, value)` features, murmur3, bundles of 2048 docs, width from `OfficialConfig.b_multiplier`) queried through their expression DSL (`"0:v AND 1:w"`, `NOT`, `""` = all) — a different hash from ours, never bit-compared, matched by FPR / memory instead; `filter_mode="exact"` is our `clause_mask` packed into the official scorer's `filtering_bit_mask`. See [kernels.md](kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend) |

## Key observation: LiNR hosts *both* filter types

LiNR's `forward` is decoupled from the filter — it accepts `mask: [B, N]`
or `candidate_ids: [B, P]` as input
([postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py),
[prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py),
[one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)). Any
`FilterModule` that produces those shapes plugs in. Both
`ExactAttributeFilter` and `BloomFilter` do, and they compose via
`combine_masks` / `combine_indices`.

Asymmetry: SilverTorch fuses its predicate *into* the probe-and-score
kernel, so swapping its filter requires a new fused kernel — not a
wrapping. Two such kernels ship today: `codesigned_probe_score` (bloom
predicate) and `codesigned_probe_score_exact` (exact AND-of-OR
predicate over the same `[N, C, A_max]` narrow attrs as
`ExactAttributeFilter`); `SilverTorch.filter_mode` selects between them.
The LiNR side, by contrast, decouples the filter entirely — any
`FilterModule` plugs in. The natural extension along the LiNR axis is
to add new `FilterModule` subclasses; along the SilverTorch axis it is
to add new fused predicate variants.

## Caller patterns

```python
from retrieve import (
    BloomFilter, ExactAttributeFilter,
    PostfilterKNN, PrefilterKNN, OneBitKNN,
    combine_indices, combine_masks,
)

ef = ExactAttributeFilter(backend="triton").to("cuda")
ef.register_index(item_attrs, clause_is_reverse=is_reverse)

bf = BloomFilter(m_bits=1024, k_hash=5, backend="triton").to("cuda")
bf.register_index(item_attrs)

# Mask path — combine exact + approximate, feed PostfilterKNN / OneBitKNN.
mask = combine_masks(ef.evaluate_mask(qa), bf.evaluate_mask(qa))
ids, scores = postfilter_knn(query, mask=mask)

# Candidate-id path — sparse cascade (most-selective filter first), feed PrefilterKNN.
cand_ids, counts = combine_indices([ef, bf], [qa, qa])
ids, scores = prefilter_knn(query, candidate_ids=cand_ids, counts=counts)
```

**When a filter leaves fewer than K survivors** the returned top-K row is
padded: the dead slots carry score `-inf` and id `-1`, on every backend and
on both the standalone-filter paths above and SilverTorch's fused
`filter_mode="bloom" / "exact"` path (the two Triton epilogues apply the
sentinel themselves — see [kernels.md](kernels.md#score-conventions)).
Callers must drop those slots rather than treat `-1` as an item: a hit-based
metric that only compares ids would otherwise score a padded slot against a
target id of `-1`.

## Bloom hash keys: `(clause_idx, value)`

The paper says *"for each feature, we apply K hash functions"* — and a
feature is a `(key, value)` pair. The implementation reflects this:
[`bloom_hash._signature_batch`](../../retrieve/src/retrieve/layers/filters/bloom_hash.py)
(the shared core both builders wrap)
mixes a per-clause salt (`_mix64(clause_id, …)`) into the post-hash bits
before the position mask, so value `V` in clause C0 lands on different
bits than the same `V` in clause C3. Without this, single-clause queries
on overlapping value vocabularies (common in real schemas — year buckets,
version counts, license codes all share small integer ranges) leak
~25–30% of non-matching items as false positives via cross-clause value
collision. The salt lives in the shared core, so item-side
`build_signatures` and query-side `build_query_signatures` are
symmetric by construction. It is a pure function of the clause index
(`generate_clause_salt(C, device)` → `[C]` int64, no seed), and since
B5 (2026-09-06) `BloomFilter` and `SilverTorch(filter_mode="bloom")`
register it once as the **`clause_salt` buffer** at `register_index`
and pass it to every builder call; before that the two splitmix64
constants were materialised per call with `torch.tensor(_SALT,
device=cuda)` — a pageable host→device copy on every forward that
inflated the eager bloom path by ~0.4 ms and broke raw CUDA-graph
capture. The bits are identical either way (`build_signatures(...,
clause_salt=None)` still derives the salt on the fly for standalone
callers — the parity tests, the tuner — and
[`test_bloom_hash.py`](../../retrieve/tests/correctness/test_bloom_hash.py)
pins the buffer path against the old inline computation). A
`SilverTorch` bloom index registered *without* attributes stores an
empty `clause_salt` (the clause count is unknown) and derives it at
query time — device-side, from `arange` and the two constants applied
as Python ints (a wrapped scalar rides along the kernel arguments), so
that path copies nothing either. No
runtime cost worth measuring (one extra elementwise XOR inside an already
chunked loop) and zero kernel impact —
[`bloom_match`](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py),
[`bloom_compact`](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py),
and the bloom branch of
[`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)
all consume opaque `[N, W]` / `[B, W]` int64 buffers and are unchanged.

**Index vs query build paths.** The item index goes through
`build_signatures` (chunk loop over `_BUILD_SIGS_BATCH` rows,
bandwidth-bound, ~128 ms for N=3M, paid once at `register_index`). The
per-forward query build is the separate loop-free
`build_query_signatures`, written as pure tensor flow so the
outer `torch.compile(dynamic=True, mode="reduce-overhead")` wrapped
around each algo in `evaluation/retrieval/algos/` captures it into one
cudagraph. Eager standalone (no algo wrapper) is launch-overhead-bound
(~0.4 ms flat in B from ~15 small CUDA kernels); under the algo-level
cudagraph_trees capture it collapses to ~0.09 ms — ~4× at all batch
sizes, ~80% of `SilverTorch.forward` at bs=1. Both builders wrap the
same `_signature_batch` core, so outputs are bit-equal (asserted by
[`test_bloom_hash.py`](../../retrieve/tests/correctness/test_bloom_hash.py)).

`bloom_sigs` snapshots persisted before this keying was added are stale
and must be rebuilt; the bench harness rebuilds on every run.

## Out of scope

- DSL / nested AND/OR/NOT / RPN walker — kept out of the filter API. If
  full predicate-tree support is ever needed, it would be a host-side
  parser feeding multiple `evaluate_mask` calls into `combine_masks`.
- NOT inside Bloom — exact reverse semantics live in `ExactAttributeFilter`
  and stay there.
- Range / prefix / numeric predicates — neither paper supports them; both
  are equality-on-int64-hash.
