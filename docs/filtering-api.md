# Unified attribute-filter API

## Why this exists

Two filtering paths grew up independently in `retrieve/`:

- `ClauseIndex` ([retrieve/src/retrieve/layers/utils/filters.py:9-66](../retrieve/src/retrieve/layers/utils/filters.py#L9-L66)) —
  exact, supports reverse clauses, `[N, C, A_max]` int64 items + `[B, C]`
  query. Caller composes mask / indices and threads them into LiNR.
- Silvertorch's bloom — encoding private to the silvertorch package, inlined
  inside `SilverTorch.forward` ([silvertorch/main.py:138-161](../retrieve/src/retrieve/layers/silvertorch/main.py#L138-L161)).
  The standalone `bloom_match` Triton kernel ([kernels/triton/silvertorch/bloom_match.py](../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py))
  works but has no consumer ([kernels.md](./kernels.md)).

Today neither implementation subclasses the existing `FilterModule` ABC
([retrieve/src/retrieve/interfaces.py:8-19](../retrieve/src/retrieve/interfaces.py#L8-L19)),
the bloom helpers can't be used outside silvertorch, the eval harness
declares `--use-attrs` but `algorithms.py` wraps every algorithm as
`lambda q: idx(q)` so attrs never reach the index, and `combine_masks` is
docs-only ([architecture.md](./architecture.md)).

This doc specifies the unified API. Scope is the [filtering.md](./filtering.md)
single-PR sketch: paper-strict bloom (conjunctive only, no DSL, no NOT,
no reverse), reverse stays a `ClauseIndex`-only feature, and SilverTorch's
fused in-cluster bloom is untouched.

## Design constraint: mask vs indices is not a free round-trip

Within a single filter, multi-clause AND is already one kernel:

- `clause_compact` ([clause_compact.py:61](../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L61)) —
  `for c in tl.static_range(C)` evaluates all clauses against `BLOCK_N`
  items, AND'ing per-item; the same kernel does cumsum + atomic_add to
  emit compact `(positive_indices[B, P], counts[B])` directly. No
  `[B, N]` ever materialized.
- `bloom_match` ([bloom_match.py:39-41](../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py#L39-L41)) —
  all clauses fold into one bloom signature at register time; subset
  test `(qb & sigs) == qb` is one masked-AND-reduce per item. Outputs
  a `[B, N]` bool today (no compact path yet).

So the inefficiency is **not** within a single filter; it is at the
boundary between filter output format and downstream consumer:

| consumer | wants | filter native cheapest output |
|---|---|---|
| `LiNR_V1` | `mask: [B, N]` | clause: pure-torch broadcast; bloom: `bloom_match` |
| `LiNR_V2` | `(candidate_ids[B, P], counts[B])` | clause: `clause_compact` (direct); bloom: needs compaction |
| `LiNR_V3` (mask) | `mask: [B, N]` | same as V1 |
| `LiNR_V3` (cand) | `candidate_ids` | same as V2 |
| `SilverTorch` | `query_clause_attrs: [B, C]` | not via this API — silvertorch has its own fused path |

Two corollaries:

1. `evaluate_indices` must not be implemented as `evaluate_mask` →
   `compact_mask` for filters that have a direct compact kernel.
   `ClauseIndex` already does the right thing
   ([filters.py:48-66](../retrieve/src/retrieve/layers/utils/filters.py#L48-L66)).
   `BloomFilter`'s first cut may use `bloom_match` + `compact_mask`,
   but the API leaves room for a future fused `bloom_compact` kernel
   without touching callers.
2. Cross-filter combination must not always go through `[B, N] & [B, N]`.
   That works fine when both filters are loose (high pass-rate) and the
   consumer wants a mask, but it's wasteful when the consumer wants
   indices and one filter is highly selective. The API exposes both
   patterns and lets the harness pick.

## Module layout

```
retrieve/src/retrieve/layers/filters/
    __init__.py            # re-exports + combiner helpers
    clause.py              # ClauseIndex (moved from utils/filters.py)
    bloom.py               # BloomFilter (new) + hashing helpers
                           # (lifted from silvertorch/bloom.py)
```

`layers/utils/filters.py` becomes a one-line re-export of `ClauseIndex`
for back-compat (existing tests / callers unchanged). The
silvertorch-package `BloomIndex` stays as a thin shim that delegates to
`BloomFilter`. Helpers (`_build_signatures`, `_generate_seeds`, `_mix64`)
move from `silvertorch/bloom.py` to `layers/filters/bloom.py`;
`silvertorch/main.py` re-imports them from the new location, so its
fused kernel path is unchanged.

## The contract

```python
class FilterModule(nn.Module, ABC):
    """Boolean predicate over an item index, paper-faithful semantics."""

    def register_index(
        self,
        item_clause_attrs: Tensor,        # [N, C, A_max] int64, -1 = pad
        item_embs: Tensor | None = None,  # currently unused; reserved
    ) -> None: ...

    # Native paths — at least one must be efficient (no inter-format
    # round-trip). Subclasses override the others if a fused kernel
    # exists for them.
    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
        """[B, N] bool. Default: gather from evaluate_indices."""

    def evaluate_indices(
        self, query_clause_attrs: Tensor
    ) -> tuple[Tensor, Tensor]:
        """([B, P] int64, [B] int64). Default: compact_mask(evaluate_mask(.))."""

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,            # [B, P] int64
    ) -> Tensor:
        """[B, P] bool — apply this filter only to candidate_ids.
        Default: gather columns of evaluate_mask. Subclasses override
        when a per-row check is much cheaper than full evaluation."""
```

Query format stays `[B, C]` int64 with `-1` deactivation, matching both
papers. Multi-value query (`[B, C, Q_max]`) is out of scope for this PR.

### `ClauseIndex(FilterModule)` — exact, reverse-supporting

Body is the current code at
[filters.py:9-66](../retrieve/src/retrieve/layers/utils/filters.py#L9-L66),
moved to `clause.py` and declared as `FilterModule`. Native fast paths:

- `evaluate_mask` — pure-torch broadcast + AND across clauses.
- `evaluate_indices` — `clause_compact` Triton kernel, no `[B, N]`.
- `evaluate_subset` — gather `item_clause_attrs[candidate_ids]` →
  `[B, P, C, A_max]` → broadcast equality + AND. Cheap when P ≪ N
  because no full-N scan.

Reverse clauses supported via `clause_is_reverse: [C] bool`.

### `BloomFilter(FilterModule)` — approximate, conjunctive

New ~50 LoC. Constructor takes `(m_bits, k_hash)`. `register_index`
calls `_build_signatures` to get `bloom_sigs: [N, W]` int64. Native:

- `evaluate_mask` — `bloom_match` Triton kernel
  ([bloom_match.py:51-85](../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py#L51-L85)),
  the now-no-longer-dead primitive.
- `evaluate_indices` — first cut: `compact_mask(evaluate_mask(.))`.
  Doc-noted follow-up: a `bloom_compact` Triton kernel modelled on
  `clause_compact` (subset-test + cumsum + atomic_add) eliminates the
  `[B, N]` materialisation. Caller-API unchanged when that lands.
- `evaluate_subset` — gather `bloom_sigs[candidate_ids]` →
  `[B, P, W]` → `(qb & sigs) == qb` per word, all-reduce. Cheap when
  P ≪ N, no kernel needed.

Paper-strict: no reverse, no DSL, no nesting. `clause_is_reverse` is
**rejected** at register time; the harness routes reverse-touching
queries through `ClauseIndex` instead.

## Cross-filter composition

`layers/filters/__init__.py` exposes two helpers:

```python
def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional [B, N] masks. None-tolerant.
    Returns None if all inputs are None. Single torch op (no kernel)."""

def combine_indices(
    filters: Sequence[FilterModule],
    query_clause_attrs: Sequence[Tensor],   # one per filter
) -> tuple[Tensor, Tensor]:
    """Sparse cascade. The first (most-selective) filter produces
    indices via its native compact path; each subsequent filter is
    invoked via evaluate_subset on those indices and the survivors
    are re-compacted. No [B, N] from the cascade itself."""
```

Composition table:

| consumer wants | filters available | use |
|---|---|---|
| `mask` | clause only | `clause.evaluate_mask(qa)` |
| `mask` | bloom only | `bloom.evaluate_mask(qa)` |
| `mask` | both, plus an external business mask | `combine_masks(clause.evaluate_mask, bloom.evaluate_mask, biz)` |
| `indices` | clause only | `clause.evaluate_indices(qa)` |
| `indices` | bloom only | `bloom.evaluate_indices(qa)` |
| `indices` | both | `combine_indices([clause, bloom], [qa_c, qa_b])` — order matters; harness puts the most-selective filter first |

Cross-compatibility (filter producer → retrieval consumer):

| producer → consumer | works? | format flow |
|---|---|---|
| `BloomFilter` → `LiNR_V1` | ✅ | bloom mask → `mask=` arg |
| `BloomFilter` → `LiNR_V2` | ✅ | bloom indices (via compact, future kernel) → `candidate_ids=` |
| `BloomFilter` → `LiNR_V3` | ✅ | mask or indices — V3 takes either |
| `ClauseIndex` → `LiNR_V*` | ✅ (already works) | exact path; reverse supported |
| `combine_masks(clause, bloom)` → `LiNR_V1` | ✅ | exact AND approximate; FPs survive into top-K and get scored down |
| `combine_indices([clause, bloom])` → `LiNR_V2` / `LiNR_V3` | ✅ | sparse cascade; cheapest when both filters are tight |
| `ClauseIndex` → `SilverTorch` (mask kwarg) | ✅ | exact override; bypasses bloom-fused speedup |
| any → `IVF_INT8_ANN` | ❌ | no mask kwarg today; out of scope |
| `BloomFilter` → `SilverTorch` | possible but pointless | SilverTorch already filters via its own bloom |

## Why filters live external to retrieval modules

LiNR's `forward` is decoupled — it accepts `mask` (V1, V3) or
`candidate_ids + counts` (V2, V3) ([v1.py:27-36](../retrieve/src/retrieve/layers/linr/v1.py#L27-L36),
[v2.py:33-41](../retrieve/src/retrieve/layers/linr/v2.py#L33-L41),
[v3.py:54-62](../retrieve/src/retrieve/layers/linr/v3.py#L54-L62)). Any
filter that produces those shapes plugs in. No LiNR kernel changes are
required for either filter to work with any LiNR variant.

SilverTorch is asymmetric: bloom is fused into `codesigned_probe_score`
([silvertorch/main.py:152-161](../retrieve/src/retrieve/layers/silvertorch/main.py#L152-L161))
because the silvertorch paper's whole co-design point is in-cluster
filtering. Replacing that bloom with `ClauseIndex` would require a new
fused kernel, not a wrapping. Out of scope for this PR. The natural
extension only goes one way: bring `BloomFilter` to LiNR.

## Concrete file changes

Create:
- `retrieve/src/retrieve/layers/filters/__init__.py`
- `retrieve/src/retrieve/layers/filters/clause.py`
- `retrieve/src/retrieve/layers/filters/bloom.py`
- `retrieve/tests/correctness/test_bloom_filter.py` —
  bloom-is-superset-of-clause parity (no false negatives), FPR by
  `(m_bits, k_hash)`, conjunctive AND across clauses, query `-1`
  deactivation, `evaluate_subset` parity with mask-then-gather.
- `retrieve/tests/correctness/test_combine_filters.py` —
  `combine_masks` None-tolerance + AND identity; `combine_indices`
  cascade equivalence to mask-AND-then-compact on small inputs.

Modify:
- `retrieve/src/retrieve/interfaces.py:8-19` — extend `FilterModule`
  with the three abstract methods above; existing `forward` becomes
  a thin alias to `evaluate_mask`.
- `retrieve/src/retrieve/layers/utils/filters.py` — one-line re-export
  of `ClauseIndex` from the new location (no behaviour change).
- `retrieve/src/retrieve/layers/silvertorch/bloom.py` — `BloomIndex`
  becomes a thin compat shim that delegates to `BloomFilter`; helpers
  re-imported from `layers/filters/bloom.py`.
- `retrieve/src/retrieve/__init__.py` and `layers/__init__.py` —
  expose `BloomFilter`, `combine_masks`, `combine_indices`,
  `FilterModule`.
- [filtering.md](./filtering.md) — update §"Implementation status";
  cross-link this design doc.
- [architecture.md](./architecture.md) — replace docs-only
  `combine_masks` description with code links.

## Verification

1. `pytest retrieve/tests/correctness/test_bloom_filter.py` — bloom is
   strict superset of clause on a synthetic 100k corpus; FPR ≤ paper
   numbers ([silvertorch.md, line 336](../articles/silvertorch.md))
   for `(1024 bits, 5 hashes)`.
2. `pytest retrieve/tests/correctness/test_combine_filters.py` —
   `combine_indices` equals `compact_mask(combine_masks(...))` on small
   inputs.
3. Existing `tests/correctness/test_filters.py`, `test_linr.py`,
   `test_silvertorch.py`, and `tests/parity/test_bloom_match.py` pass
   unchanged (back-compat re-export verified).
4. New cross-compat smoke in `tests/correctness/test_filters.py` —
   build a `LiNR_V3_Triton` index over a 50k-row toy embedding,
   build both filters from a synthetic `[N, 4, 4]` attr tensor, and
   confirm `linr_v3(q, mask=combine_masks(clause_mask, bloom_mask))`
   returns ids whose attrs satisfy the conjunction.
5. Latency micro-bench in `tests/bench/bench_filters.py` (new) —
   measure `evaluate_mask` and `evaluate_indices` for both filters at
   N = 1M, 10M; confirm `clause_compact` indices path is faster than
   `compact_mask(bloom_match(.))` when P ≪ N. Numbers feed
   [bench.md](./bench.md).

## Out of scope (explicit non-goals)

- Multi-value query (`[B, C, Q_max]`) — both papers describe it on the
  item side; query side is paper-side single-value. Defer.
- DSL / RPN walker — explicit per [filtering.md](./filtering.md).
- NOT inside bloom — explicit per [filtering.md](./filtering.md).
- Range / prefix / numeric predicates — neither paper supports them.
- Replacing bloom inside `codesigned_probe_score` — would need a new
  fused kernel.
- IVF-aware ClauseIndex (filter-while-probing without bloom) — research
  follow-up.
- A `bloom_compact` Triton kernel — first PR ships compact via
  `compact_mask(bloom_match)`. Add the fused kernel only if profiling
  on YFCC-5M shows the compact pass is the bottleneck for V2 / V3.
