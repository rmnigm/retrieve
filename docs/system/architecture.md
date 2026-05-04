<!-- claude code generated file -->

# `retrieve` architecture

> Previously: `retrieve/docs/architecture.md` (originally `retrieve/docs/ARCHITECTURE.md`).

This is the high-level map of the `retrieve` package: what each retrieval
module does, how filters compose, and which Triton kernels back which paths.
For per-kernel detail (launch grids, tile shapes, autotune keys, numerics)
see [kernels.md](kernels.md).

## Module families

Two retrieval families live side by side, both implementing the
[`RetrievalModule`](../../retrieve/src/retrieve/interfaces.py) interface:

- **LiNR** ([`layers/linr/`](../../retrieve/src/retrieve/layers/linr/)) — three
  variants (`SimilarityMasking` dense, `PrefilterKNN` sparse pre-filter,
  `OneBitKNN` 1-bit OPORP) with separate pure-torch and Triton-fused
  backends. Filtering is **decoupled**: each forward takes a mask or a
  candidate-id buffer the caller computed via a
  [`FilterModule`](../../retrieve/src/retrieve/interfaces.py)
  ([`ExactAttributeFilter`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) /
  [`BloomFilter`](../../retrieve/src/retrieve/layers/filters/bloom.py)),
  [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py), or
  any upstream cascade composed via
  [`combine_masks` / `combine_indices`](../../retrieve/src/retrieve/layers/filters/__init__.py).
- **SilverTorch** ([`layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/)) —
  co-designed IVF + INT8 ANN + Bloom attribute filter, all fused in one
  Triton kernel. Filtering is **inline** (Bloom signatures live on the
  module); no standalone `ExactAttributeFilter` is wired in.

Filter modules live in [`layers/filters/`](../../retrieve/src/retrieve/layers/filters/):
`ExactAttributeFilter` (exact, supports reverse), `BloomFilter` (approximate,
conjunctive), and the `combine_masks` / `combine_indices` composition
helpers. Both filters subclass [`FilterModule`](../../retrieve/src/retrieve/interfaces.py).
Other utilities live in [`layers/utils/`](../../retrieve/src/retrieve/layers/utils/):
`compact_mask`, `FullScanKNN`, `DotProductScorer`, and the quantizers
(`quantize_int8`, `quantize_oporp_1bit`).

## Clause / attribute data layout

`ExactAttributeFilter` stores items with `C` clauses, each holding up to
`A_max` int64 attribute IDs padded with `-1`. Two buffers:

- `item_clause_attrs` — `[N, C, A_max]` int64.
- `clause_is_reverse` — `[C]` bool, `True` marks a *reverse* clause.

At query time a query supplies one attribute ID per clause in
`query_clause_attrs[B, C]`.

Semantics:

- A clause **passes** for an item if **any** of the item's attribute IDs for
  that clause equals the query attribute (OR within a clause). Padded `-1`
  slots never equal a valid query attribute, so they are inert.
- For a **reverse** clause the outcome is inverted: the item passes iff it
  does *not* match.
- A query attribute of `-1` marks the clause **inactive** — it always passes.
- An item passes overall when **all** clauses pass (AND between clauses).

## Filter composition

The [`FilterModule`](../../retrieve/src/retrieve/interfaces.py) contract
exposes three native evaluators. `evaluate_mask` is `@abstractmethod`
and must be overridden by every concrete filter; `evaluate_indices` and
`evaluate_subset` ship cheap defaults
(`compact_mask(evaluate_mask(...))` and
`evaluate_mask(...).gather(1, candidate_ids)` respectively) and are
overridden only when a fused kernel beats the default:

- `evaluate_mask(query_clause_attrs) → [B, N] bool` — dense path, used by
  `SimilarityMasking` (which masks scores in place) and `OneBitKNN`'s mask
  path. `ExactAttributeFilter` routes to the fused `clause_mask` Triton
  kernel on CUDA (no `[B, N, C, A_max]` intermediate), pure-torch broadcast
  on CPU. `BloomFilter` routes to the `bloom_match` Triton kernel on CUDA,
  pure-torch subset test on CPU.
- `evaluate_indices(query_clause_attrs) → (positive_indices[B, P] int64,
  counts[B] int64)` — sparse path, used by `PrefilterKNN` and `OneBitKNN`'s
  masked-Triton path. `ExactAttributeFilter` routes to the fused
  `clause_compact` Triton kernel on CUDA; `BloomFilter` routes to the fused
  `bloom_compact` kernel on CUDA. Both fall back to
  `compact_mask(evaluate_mask)` on CPU.
- `evaluate_subset(query_clause_attrs, candidate_ids) → [B, P] bool` —
  apply a filter only to the given candidate ids. Both filters override
  this for cheap-when-P-small paths via gather + broadcast/word-wise
  subset test; default is `evaluate_mask(...).gather(1, candidate_ids)`.

Callers that already have a bool mask from some other source (a hand-rolled
predicate, an external mask passed through the API) use
[`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py)
directly to convert it to the `(positive_indices, counts)` form V2 wants.

Two composition helpers ship in
[`layers/filters/__init__.py`](../../retrieve/src/retrieve/layers/filters/__init__.py):

- `combine_masks(*masks)` — element-wise AND of N optional `[B, N]` masks,
  None-tolerant; one torch op, no kernel.
- `combine_indices(filters, queries)` — sparse cascade. The first filter
  produces `(ids, counts)` via its native compact path; each subsequent
  filter is invoked via `evaluate_subset` on those ids and the survivors
  are re-compacted. No `[B, N]` from the cascade itself. Caller orders
  filters most-selective first.

## LiNR variants

All three return `(ids[B, K], scores[B, K])`. Backends ship in pairs: a
torch reference and a `*Triton` subclass. `PrefilterKNN` and `OneBitKNN`
have real fused kernels; `SimilarityMasking`'s `*Triton` subclass is a
backend-dispatch alias only — its dense matmul + top-K runs in pure torch
(cuBLAS + CUB already do the right thing; there's no fusion to win).

- **`SimilarityMasking` — dense similarity, optional mask.** Full
  `query @ item_embs.T`, masked scores set to `-inf`, then top-K. Forward
  takes `(query, mask=None)`. Wins when the mask is dense or absent. Both
  backends share the same pure-torch implementation.
- **`PrefilterKNN` — sparse pre-filter.** Forward takes
  `(query, candidate_ids=None, counts=None)` — passing item ids per query,
  precompacted by the caller (typically
  `ExactAttributeFilter.evaluate_indices`). The module itself does **no**
  mask handling: it gathers the passing rows, runs a reduced `bmm`, top-K's
  locally, and maps back to global ids. Without `candidate_ids` it falls
  back to a `SimilarityMasking`-style exhaustive matmul. Triton backend uses
  `fused_masked_knn_topk` for the sparse path and the parent's pure-torch
  dense matmul for the unmasked fallback.
- **`OneBitKNN` — 1-bit Sign-OPORP.** [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)
  builds three buffers at index time: `item_bits[N, W]` int64
  (`W = D / 64`), `oporp_signs[D]` int8, `oporp_perm[D]` int64. Scoring
  is `D - 2 * popcount(query_bits ^ item_bits)`, so retrieval is purely
  bitwise — `D/8` bytes per item, ~16× smaller than fp16. Forward takes
  `(query, mask=None, candidate_ids=None)` and resolves three paths:
  1. `candidate_ids` provided — gather the passing `item_bits`, score,
     top-K, map back. `mask` is ignored on this path.
  2. `mask` provided — score the full corpus, mask scores to `-inf`,
     top-K. (The Triton backend instead compacts the mask first via
     `compact_mask` and uses the indirect-load kernel path — popcount is
     cheap enough that the gather always wins.)
  3. Neither provided — full exhaustive popcount scan.

  All three paths in the Triton backend share the same kernel
  (`oporp_1bit_match_topk`) — only the addressing changes.

  The pure-torch reference (`OneBitKNN`, used by tests and as a fallback
  when the Triton backend isn't selected) wraps two hot bodies with
  `torch.compile(dynamic=True, mode='reduce-overhead')`:
  [`project_oporp_1bit_query`](../../retrieve/src/retrieve/layers/utils/quantize.py)
  (per-call query projection) and `OneBitKNN._score_full`'s xor + popcount
  + reduce body. Both follow the same dispatch pattern as the bloom query
  build — CUDA-only, eager fallback on CPU. The `_score_full` win is
  driven by Inductor fusing the 6-op SWAR popcount chain into one elementwise
  pass over the `[B, N, W]` xor tensor; the projection win is launch-overhead
  elision.

## SilverTorch

[`SilverTorch`](../../retrieve/src/retrieve/layers/silvertorch/main.py)
implements the SilverTorch paper's Algorithm 1: IVF clustering over INT8-
quantized item codes, with a per-item Bloom signature for attribute
filtering, all fused into a single Triton kernel
[`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py).
Constructed via `build_silvertorch(item_embs, k, *, n_lists, n_probe,
m_bits=None, k_hash=None, n_iter=10, seed=0, item_clause_attrs=None)`
— everything after `k` is keyword-only. Forward takes
`(query, query_clause_attrs=None, mask=None, candidate_ids=None)` —
`query_clause_attrs=None` is a documented fast path that skips bloom
evaluation entirely.

The bloom filter is private to SilverTorch: signatures are derived from
`item_clause_attrs` at `register_index` time and stored as
`bloom_sigs[N, W]` plus per-row `hash_seeds`. There is no
`ExactAttributeFilter` on this path. Constructing `SilverTorch` without
`m_bits`/`k_hash`
yields the bloom-free variant — `register_index` skips bloom buffer
allocation, and `forward` requires `query_clause_attrs=None`.

## Utility modules

- [`FullScanKNN`](../../retrieve/src/retrieve/layers/utils/retrieval.py) —
  exhaustive `query @ item_embs.T` + top-K with optional post-mask or
  candidate_ids, used as a baseline / sanity check.
- [`DotProductScorer`](../../retrieve/src/retrieve/layers/utils/scorers.py) —
  `ScorerModule` for candidate-set rerank (`bmm` over gathered item rows).
- [`post_filter_topk`](../../retrieve/src/retrieve/layers/utils/retrieval.py) —
  applies a post-filter to already-computed top-K results.
- Three abstract bases in [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py):
  `RetrievalModule`, `FilterModule`, `ScorerModule`.

## Triton kernels

Kernels live under [`kernels/triton/`](../../retrieve/src/retrieve/kernels/triton/)
in three subtrees by domain. One launch per `forward()`, host-side
`torch.topk` over the score buffer (CUB beats anything we can write in
pure Triton). Full per-kernel detail in [kernels.md](kernels.md).

| subtree                                                                                              | kernel                                                                                                                                | consumer                          |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| [`linr/`](../../retrieve/src/retrieve/kernels/triton/linr/)                                             | [`fused_masked_knn_topk`](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py) — gather + dot over `positive_indices` | `PrefilterKNNTriton` (masked path) |
| [`linr/`](../../retrieve/src/retrieve/kernels/triton/linr/)                                             | [`oporp_1bit_match_topk`](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py) — XOR + popcount, all `OneBitKNN` paths | `OneBitKNNTriton`                 |
| [`filters/`](../../retrieve/src/retrieve/kernels/triton/filters/)                                       | [`clause_compact`](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) — fused clause eval + stream compaction          | `ExactAttributeFilter.evaluate_indices` |
| [`filters/`](../../retrieve/src/retrieve/kernels/triton/filters/)                                       | [`clause_mask`](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) — fused clause eval emitting `[B, N]` bool             | `ExactAttributeFilter.evaluate_mask` |
| [`filters/`](../../retrieve/src/retrieve/kernels/triton/filters/)                                       | [`bloom_compact`](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py) — fused subset-test + stream compaction            | `BloomFilter.evaluate_indices`    |
| [`silvertorch/`](../../retrieve/src/retrieve/kernels/triton/silvertorch/)                               | [`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py) — fused IVF + INT8 + Bloom    | `SilverTorch`                     |
| [`silvertorch/`](../../retrieve/src/retrieve/kernels/triton/silvertorch/)                               | [`bloom_match`](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py) — bool subset test (standalone)                     | `BloomFilter.evaluate_mask`       |

## Testing

Tests live in [`retrieve/tests/`](../../retrieve/tests/) and split by purpose,
not by module: [`correctness/`](../../retrieve/tests/correctness/) for module-
level semantics against torch baselines, [`parity/`](../../retrieve/tests/parity/)
for Triton kernel vs pure-torch agreement. The suite is GPU-only and skipped
without CUDA via the root [`conftest.py`](../../retrieve/tests/conftest.py)
gate. Performance characterization (latency, memory, recall sweeps) lives in
[`evaluation/`](../../evaluation/), not in `tests/`. Full reference in
[testing.md](testing.md).

## Module layout

- LiNR variants ship one file per class, paired with a `_triton.py`
  sibling that subclasses the torch reference. The torch references live
  in [`similarity_masking.py`](../../retrieve/src/retrieve/layers/linr/similarity_masking.py),
  [`prefilter_knn.py`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py),
  and [`one_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py);
  the Triton subclasses mirror them as `*_triton.py`. Construct a module
  directly (`SimilarityMaskingTriton(k=...)` etc.) — there's no version-
  numbered builder; the v1/v2/v3 naming lives only at the evaluation-algo
  layer.
- The SilverTorch builder (`build_silvertorch`) lives next to its class
  in [`layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/).
- Quantizers ship from [`layers/utils/quantize.py`](../../retrieve/src/retrieve/layers/utils/quantize.py):
  `quantize_int8` (consumed by `SilverTorch`) and
  `quantize_oporp_1bit` (consumed by `OneBitKNN`). They share the file but
  no caller; LiNR never sees INT8, SilverTorch never sees OPORP.
