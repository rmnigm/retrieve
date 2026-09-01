# `retrieve` architecture

This is the high-level map of the `retrieve` package: what each retrieval
module does, how filters compose, and which Triton kernels back which paths.
For per-kernel detail (launch grids, tile shapes, autotune keys, numerics)
see [kernels.md](kernels.md).

## Module families

Two retrieval families live side by side, both implementing the
[`RetrievalModule`](../../retrieve/src/retrieve/interfaces.py) interface
and selecting their compute path via a `backend=` flag on `__init__`
(the literal alias is exported as `Backend` from
[`interfaces.py`](../../retrieve/src/retrieve/interfaces.py)):

| `backend=` | meaning | who accepts it |
|---|---|---|
| `"triton"` | fused Triton kernels. The default everywhere. | every module |
| `"torch"` | pure-torch eager equivalent, same semantics, larger intermediates | every module |
| `"cuda"` | hand-written CUDA C++, JIT-compiled on first forward | **`SilverTorch` only** |

`"cuda"` is not a universal third path: it exists solely for
`SilverTorch`'s probe-scoring kernel. Every other class accepts the flag
for API symmetry, but its dispatch is `if backend == "triton": … else:
<torch>`, so passing `"cuda"` to a LiNR module or a standalone filter
silently runs the **torch** path. See
[Backend dispatch](#backend-dispatch) below.

- **LiNR** ([`layers/linr/`](../../retrieve/src/retrieve/layers/linr/)) — five
  variants (`PostfilterKNN` dense fp16, `PostfilterKNNInt8` dense
  int8, `PrefilterKNN` sparse pre-filter, `OneBitKNN` 1-bit Sign-OPORP,
  `SimHashKNN` 1-bit SimHash; the two bit-KNNs share the
  [`_PackedBitsKNN`](../../retrieve/src/retrieve/layers/linr/_bit_knn.py)
  base, which owns scoring/dispatch while subclasses own bit production).
  Filtering is **decoupled**: each forward takes a mask or a candidate-id
  buffer the caller computed via a
  [`FilterModule`](../../retrieve/src/retrieve/interfaces.py)
  ([`ExactAttributeFilter`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) /
  [`BloomFilter`](../../retrieve/src/retrieve/layers/filters/bloom.py)),
  [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py), or
  any upstream cascade composed via
  [`combine_masks` / `combine_indices`](../../retrieve/src/retrieve/layers/filters/__init__.py).
- **SilverTorch** ([`layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/)) —
  co-designed IVF + INT8 ANN with an inline attribute filter selected by
  a `filter_mode ∈ {"none", "bloom", "exact"}` flag. `"bloom"` fuses Bloom
  subset tests into [`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py);
  `"exact"` fuses an exact AND-of-OR attribute predicate into
  [`codesigned_probe_score_exact`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py);
  `"none"` runs plain IVF + INT8. All three modes share the IVF probe and
  INT8 dot-product path; only the predicate inside the fused kernel
  changes. With `backend="torch"` the same semantics run in pure torch
  (phase 1 IVF probe + predicate + INT8 dequant + dot + topk),
  materializing a `[B, P, D]` intermediate; with `backend="cuda"` they run
  the paper's two-kernel CUDA C++ design (`filter_mode` `"none"` /
  `"bloom"` only). Filtering is **inline**: bloom
  signatures or narrow clause attrs live on the module; no standalone
  `ExactAttributeFilter` / `BloomFilter` instance is wired in.

Filter modules live in [`layers/filters/`](../../retrieve/src/retrieve/layers/filters/):
`ExactAttributeFilter` (exact, supports reverse), `BloomFilter` (approximate,
conjunctive), the `combine_masks` / `combine_indices` composition
helpers, and [`bloom_hash.py`](../../retrieve/src/retrieve/layers/filters/bloom_hash.py) —
the single home of the bloom hash math (`generate_seeds`,
`build_signatures`, `build_query_signatures`, `bloom_subset_match`),
consumed by both `BloomFilter` and `SilverTorch`'s fused bloom mode.
Both filters subclass [`FilterModule`](../../retrieve/src/retrieve/interfaces.py),
whose `register_index(item_clause_attrs, *, clause_is_reverse=None)`
signature is keyword-only after the first argument.
Other utilities live in [`layers/utils/`](../../retrieve/src/retrieve/layers/utils/):
`compact_mask`, `post_filter_topk`, `FullScanKNN`, `KMeansTorch`, the
shared masked top-K epilogue
([`topk.py`](../../retrieve/src/retrieve/layers/utils/topk.py):
`masked_topk`, `counts_to_valid`), and the quantizers (`quantize_int8` /
`quantize_int8_global`, `quantize_oporp_1bit`, `quantize_simhash_1bit`).

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

- `evaluate_mask(query_clause_attrs) → [B, N] bool` — dense path, fed to
  the mask-taking dense layers (`PostfilterKNN` / `PostfilterKNNInt8`,
  which mask scores before top-K) or compacted via `compact_mask` for
  the candidates-taking layers. `ExactAttributeFilter` routes to the
  fused `clause_mask` Triton kernel on the `"triton"` backend (no
  `[B, N, C, A_max]` intermediate), pure-torch broadcast on `"torch"`.
  `BloomFilter` routes to the `bloom_match` Triton kernel on
  `"triton"`, pure-torch subset test on `"torch"`.
- `evaluate_indices(query_clause_attrs) → (positive_indices[B, N] int64,
  counts[B] int64)` — sparse path, used by the callers that feed
  `PrefilterKNN` / the bit-KNNs' candidates path. The returned index
  buffer is **full-width** `[B, N]`: only the first `counts[b]` entries
  of each row are meaningful (`-1`-filled tails on the kernel path,
  arbitrary argsort tails on the `compact_mask` fallback), and within-row
  order is unspecified. `ExactAttributeFilter` routes to the fused
  `clause_compact` Triton kernel on the `"triton"` backend; `BloomFilter`
  routes to the fused `bloom_compact` kernel. The `"torch"` backend falls
  back to `compact_mask(evaluate_mask)`.
- `evaluate_subset(query_clause_attrs, candidate_ids) → [B, P] bool` —
  apply a filter only to the given candidate ids. Both filters override
  this with gather + the shared torch-side predicate helpers
  (`clause_subset_match` in
  [`exact_attribute.py`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py),
  `bloom_subset_match` in
  [`bloom_hash.py`](../../retrieve/src/retrieve/layers/filters/bloom_hash.py) —
  the same functions SilverTorch's eager backend uses); default is
  `evaluate_mask(...).gather(1, candidate_ids)`.

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

All five return `(ids[B, K], scores[B, K])`; `-1` / `-inf` are the
"no item" sentinels for masked-out or short rows. Each module takes a
`backend=` arg in `__init__` (`"cuda"` resolves to the torch path here —
see [Backend dispatch](#backend-dispatch)); `"triton"` is the default
(the eval harness and the original paper experiments target Triton). The
torch backend is eager — callers wanting Inductor fusion or cudagraph
capture wrap the module with `torch.compile` themselves. The library does
not bind compile internally. The torch-side "mask to `-inf` → topk →
map ids → `-1`-sentinel non-finite winners → pad to k" epilogue is the
shared [`masked_topk`](../../retrieve/src/retrieve/layers/utils/topk.py)
utility, not per-class copies.

- **`PostfilterKNN` — dense similarity, optional mask.** Full
  `query @ item_embs.T`, masked scores set to `-inf`, then top-K. Forward
  takes `(query, mask=None)`. The original Triton kernel was removed
  because cuBLAS + CUB already deliver the same memory traffic; the
  `backend=` flag is accepted for API symmetry but is a no-op on this
  class.
- **`PostfilterKNNInt8` — dense int8 similarity, optional mask.**
  Single-stage int8 dense matmul + optional mask + top-K, int32
  end-to-end. Items and queries are int8-quantized with one global scale
  each (SilverTorch §3.2); the matmul runs through `torch._int_mm`
  (cuBLAS LtGemm, IMMA tensor cores on Ampere+) and the int32 result
  feeds `torch.topk` directly — no rescale to fp32, no scale recovery,
  because two global scalars are a positive monotonic transform of the
  true dot product (topk ordering exact modulo int8 rounding). Storage
  is one `[D, N]` int8 buffer — half the memory of `PostfilterKNN`'s
  fp16 layout. Forward takes `(query, mask=None)`. The `backend=` flag
  is accepted for API symmetry but is a no-op (cuBLAS LtGemm runs the
  same code on both paths).
- **`PrefilterKNN` — sparse pre-filter.** Forward takes
  `(query, candidate_ids=None, counts=None)` — passing item ids per query,
  precompacted by the caller (typically
  `ExactAttributeFilter.evaluate_indices`, or `compact_mask` over an
  external mask). The layer has **no** mask parameter and does no mask
  handling itself: it gathers the passing rows, runs a reduced `bmm`,
  top-K's locally, and maps back to global ids. Without `candidate_ids`
  it falls back to a `PostfilterKNN`-style exhaustive matmul. With
  `backend="triton"` the sparse path uses `fused_masked_knn_topk`; the
  unmasked path runs the pure-torch dense matmul on either backend
  (nothing to fuse over cuBLAS). Items are stored fp16 (queries cast to
  fp16 on entry); rows with fewer than `k` survivors pad `-1` / `-inf`.
- **`OneBitKNN` / `SimHashKNN` — 1-bit Hamming KNNs.** Both subclass
  [`_PackedBitsKNN`](../../retrieve/src/retrieve/layers/linr/_bit_knn.py):
  score = `64·W − 2·popcount(query_bits ^ item_bits)` over packed int64
  sign bits — purely bitwise, `k_bits/8` bytes per item, ~16× smaller
  than fp16. The base owns `register_index` (asserts `k <= N`), backend
  dispatch, and the two forward paths; subclasses own only the index
  quantizer and the query projection:
  - `OneBitKNN(k, seed=0, backend="triton", k_bits=0)` —
    [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)
    builds `item_bits[N, W]` int64, `oporp_signs[D]` int8,
    `oporp_perm[D]` int64. `k_bits=0` is a sentinel resolved to `D` at
    `register_index` (re-resolved from the pristine constructor arg on
    every registration).
  - `SimHashKNN(k, k_bits, seed=0, backend="triton")` — fixed Gaussian
    projection `simhash_R[k_bits, D]` then sign-pack
    ([`quantize_simhash_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py));
    `k_bits` may exceed `D` for a recall-vs-memory trade.

  Forward takes `(query, candidate_ids=None, counts=None)` — there is
  no mask parameter (callers with a mask compact it first, e.g. the
  eval algos) — and resolves two paths:
  1. `candidate_ids` provided — score only those rows (per-row valid
     width bounded by `counts`, default all `P`), top-K, map back.
  2. Neither provided — full exhaustive popcount scan.

  Both paths in the Triton backend share the same kernel
  (`oporp_1bit_match_topk`) — only the addressing changes. The torch
  backend runs the eager xor + popcount + reduce chain over the same
  packed bits and produces bit-equal scores (the projection is shared,
  so both backends see byte-identical bits).

## SilverTorch

[`SilverTorch`](../../retrieve/src/retrieve/layers/silvertorch/main.py)
implements the SilverTorch paper's Algorithm 1: IVF clustering over INT8-
quantized item codes (global per-tensor scale, paper §3.2), with an
inline attribute predicate. Two axes are chosen at construction and are
independent apart from one excluded combination: `filter_mode` picks the
predicate, `backend` picks the implementation.

The predicate, `filter_mode ∈ {"none", "bloom", "exact"}`:

- `"none"` — plain IVF + INT8 ANN, no attribute filter
  ([`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)
  with `HAS_QB=False`).
- `"bloom"` — paper's bloom subset test fused into
  [`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py);
  one false-positive-tolerant filter shared across all clauses
  (`m_bits`, `k_hash` required).
- `"exact"` — exact AND-of-OR predicate fused into
  [`codesigned_probe_score_exact`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py);
  no false positives, bandwidth-cheaper per item at small `C × A_max`,
  trades the bloom hash flexibility for exact-value match.

The implementation, `backend ∈ {"triton", "torch", "cuda"}`:

- `"triton"` (default) — phases 2+3 fused into one Triton launch, no
  probe intermediate on HBM.
- `"torch"` — the same semantics eager, materializing `[B, P, D]`; large
  `P·B·D` needs one of the other two.
- `"cuda"` — the paper's two-kernel CUDA C++ design (transposed bloom
  index → 1-bit-per-item masks → masked `__dp4a` scoring), JIT-compiled
  on first forward. Results are bit-identical to `"triton"`. **Rejects
  `filter_mode="exact"` at construction** — that combination raises
  `ValueError`; use `"triton"` or `"torch"` for exact clauses. Details in
  [kernels.md](kernels.md#codesigned_probe_score_cuda--the-cuda-c-backend).

Constructed via `build_silvertorch(item_embs, k, *, n_lists, n_probe,
filter_mode="none", m_bits=None, k_hash=None, n_iter=10, seed=0,
item_clause_attrs=None, clause_is_reverse=None, backend="triton")` —
everything after `k` is keyword-only. Forward takes
`(query, query_clause_attrs=None, candidate_ids=None)` — there is no
mask parameter. `query_clause_attrs=None` is a documented fast path
that skips predicate evaluation entirely; passing `query_clause_attrs`
together with `candidate_ids` raises `ValueError` (the candidates path
scores the given candidates *without* the fused filter, so accepting
both would silently drop the predicate). `register_index` is split into
validate → `_build_ivf` (k-means) → `_quantize_items` →
`_register_filter_buffers` phases with a frozen buffer-registration
order (state-dict key order): `centroids`, `item_codes`,
`global_scale`, `padded_cluster_items`, `cluster_sizes`, then the
filter buffers.

The filter is private to SilverTorch — no standalone `FilterModule`
instance is wired in; the module shares the ten-line predicate *math*
(`bloom_subset_match`, `clause_subset_match`) and the bloom hash
builders with the filters package, not the module classes:

- For `"bloom"`, signatures are derived from `item_clause_attrs` at
  `register_index` time via `bloom_hash.build_signatures` and stored
  alongside `hash_seeds[k_hash, 2]`. **Which signature buffer is
  registered depends on the backend**: `"triton"` / `"torch"` store the
  row-wise `bloom_sigs[N, W]`; `"cuda"` stores only the transposed
  `bloom_sigs_t[m_bits, n_lists · wpc]` that its phase-2 kernel reads
  (registering both would double the filter index for nothing).
- For `"exact"`, the narrow `[N, C, A_max]` attribute tensor is stored
  as the `item_clause_attrs` buffer (plus `clause_is_reverse[C]` bool)
  and consumed directly by the exact kernel — reverse clauses are
  supported on this mode only. This pair is backend-independent: the
  cuda clause-mask kernel reads the same two buffers in place, so it
  registers nothing extra.
- For `"none"`, both attribute buffers are skipped and `forward`
  requires `query_clause_attrs=None`.

> **State dicts are not portable across backends for `filter_mode="bloom"`
> — and only for that mode.**
> A checkpoint saved under `"triton"` has a `bloom_sigs` key; the same
> module built with `backend="cuda"` expects `bloom_sigs_t`, of a
> different shape. `load_state_dict` across the two fails. Rebuild the
> index with `register_index` rather than trying to load across
> backends. Every other buffer is backend-independent: `centroids`,
> `item_codes`, `global_scale`, `padded_cluster_items`, `cluster_sizes`,
> and the `filter_mode="exact"` pair `item_clause_attrs` /
> `clause_is_reverse` — so an exact-mode checkpoint *is* portable across
> all three backends.

## Utility modules

- [`FullScanKNN`](../../retrieve/src/retrieve/layers/utils/retrieval.py) —
  exhaustive `query @ item_embs.T` + top-K with optional post-mask or
  candidate_ids, used as a baseline / sanity check. The mask implements
  **post-filter** semantics (the LiNR baseline): top-K is selected over
  the full corpus first, then masked winners are tombstoned to `-1` —
  not backfilled — so recall vs a pre-filter oracle is < 1 by design
  (documented on the class docstring).
- [`post_filter_topk`](../../retrieve/src/retrieve/layers/utils/retrieval.py) —
  applies a post-filter to already-computed top-K results, returning
  `(ids, counts)`; `FullScanKNN.forward` discards the counts.
- [`masked_topk` / `counts_to_valid`](../../retrieve/src/retrieve/layers/utils/topk.py) —
  the shared torch-side masked top-K epilogue (pure tensor-flow, traces
  cleanly under the eval harness's `torch.compile(dynamic=True,
  mode="reduce-overhead")`).
- [`KMeansTorch`](../../retrieve/src/retrieve/layers/utils/kmeans.py) —
  pure-torch Lloyd's k-means used by `SilverTorch` for IVF index
  building (not a `RetrievalModule`).
- Two abstract bases plus the `Backend` literal alias in
  [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py):
  `RetrievalModule` (a minimal lifecycle ABC — `k` attr + abstract
  `register_index`, called exactly once; forward signatures deliberately
  unconstrained), `FilterModule`, and
  `Backend = Literal["torch", "triton", "cuda"]`. All retrieval layers
  (`PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`,
  `_PackedBitsKNN` and its two subclasses, `SilverTorch`, `FullScanKNN`)
  subclass `RetrievalModule`.

## Kernels

Kernels live under [`kernels/`](../../retrieve/src/retrieve/kernels/)
in three subtrees by domain, plus
[`kernels/common.py`](../../retrieve/src/retrieve/kernels/common.py) —
shared `@triton.jit` building blocks (`popcount_int64`,
`bloom_subset_pass`, `clause_pass`, `compact_store`, `or_combine`)
called from the kernel bodies; each kernel keeps its own launch grid,
loads, and epilogue policy. Everything is Triton except one leaf,
[`silvertorch/cuda/`](../../retrieve/src/retrieve/kernels/silvertorch/cuda/),
which holds the single `.cu` translation unit behind
`SilverTorch(backend="cuda")`.

Each Triton path is **one launch per `forward()`** followed by a
host-side `torch.topk` over the score buffer (CUB beats anything we can
write in pure Triton). The CUDA backend is the one exception: both of its
*filtered* paths launch two kernels (a phase-2 mask — partial bloom or
exact clause — then the shared masked scorer) before the same host-side
top-K; the unfiltered path is a single launch. Full per-kernel detail in
[kernels.md](kernels.md).

| subtree                                                                                              | kernel                                                                                                                                | consumer                          |
|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| [`linr/`](../../retrieve/src/retrieve/kernels/linr/)                                             | [`fused_masked_knn_topk`](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py) — gather + dot over `positive_indices` | `PrefilterKNN(backend="triton")` (candidates path) |
| [`linr/`](../../retrieve/src/retrieve/kernels/linr/)                                             | [`oporp_1bit_match_topk`](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py) — XOR + popcount, both bit-KNN paths   | `OneBitKNN` / `SimHashKNN` (`backend="triton"`) |
| [`filters/`](../../retrieve/src/retrieve/kernels/filters/)                                       | [`clause_compact`](../../retrieve/src/retrieve/kernels/filters/clause_compact.py) — fused clause eval + stream compaction          | `ExactAttributeFilter.evaluate_indices` |
| [`filters/`](../../retrieve/src/retrieve/kernels/filters/)                                       | [`clause_mask`](../../retrieve/src/retrieve/kernels/filters/clause_mask.py) — fused clause eval emitting `[B, N]` bool             | `ExactAttributeFilter.evaluate_mask` |
| [`filters/`](../../retrieve/src/retrieve/kernels/filters/)                                       | [`bloom_compact`](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py) — fused subset-test + stream compaction            | `BloomFilter.evaluate_indices`    |
| [`silvertorch/`](../../retrieve/src/retrieve/kernels/silvertorch/)                               | [`codesigned_probe_score`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py) — fused IVF + INT8 + Bloom    | `SilverTorch(filter_mode="none" \| "bloom")` |
| [`silvertorch/`](../../retrieve/src/retrieve/kernels/silvertorch/)                               | [`codesigned_probe_score_exact`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) — fused IVF + INT8 + exact AND-of-OR | `SilverTorch(filter_mode="exact")`     |
| [`silvertorch/`](../../retrieve/src/retrieve/kernels/silvertorch/)                               | [`bloom_match`](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py) — bool subset test (standalone)                     | `BloomFilter.evaluate_mask`       |
| [`silvertorch/cuda/`](../../retrieve/src/retrieve/kernels/silvertorch/cuda/)                     | [`codesigned_probe_score_cuda`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py) — CUDA C++ transposed bloom + dp4a scoring (2 launches) | `SilverTorch(backend="cuda")` |

### Backend dispatch

Only `SilverTorch` branches three ways. Everywhere else the dispatch is
binary, so a `"cuda"` request lands on the torch path:

| module | `"triton"` | `"torch"` | `"cuda"` |
|---|---|---|---|
| `SilverTorch` | fused Triton | eager torch | CUDA C++ |
| `PrefilterKNN` | `fused_masked_knn_topk` | eager | → torch |
| `OneBitKNN` / `SimHashKNN` | `oporp_1bit_match_topk` | eager | → torch |
| `ExactAttributeFilter` | `clause_mask` / `clause_compact` | eager | → torch |
| `BloomFilter` | `bloom_match` / `bloom_compact` | eager | → torch |
| `PostfilterKNN` / `PostfilterKNNInt8` | cuBLAS (flag is a no-op) | same | same |

This matters when benchmarking: a harness cell labelled `backend="cuda"`
for anything other than `SilverTorch` is measuring the **torch** path.
The eval harness works around it by building filter modules for `"cuda"`
cells with `backend="triton"` — see
[evaluation.md](evaluation.md#the-cuda-backend-in-sweeps).

## Testing

Tests live in [`retrieve/tests/`](../../retrieve/tests/) and split by purpose,
not by module: [`correctness/`](../../retrieve/tests/correctness/) for module-
level semantics against torch baselines, [`parity/`](../../retrieve/tests/parity/)
for Triton kernel vs pure-torch agreement, and
[`compile/`](../../retrieve/tests/compile/) for `torch.compile`
graph-break and `torch.export` kernel-reference gates. The suite is GPU-only and skipped
without CUDA via the root [`conftest.py`](../../retrieve/tests/conftest.py)
gate. Performance characterization (latency, memory, recall sweeps) lives in
[`evaluation/`](../../evaluation/), not in `tests/`. Full reference in
[testing.md](testing.md).

## Module layout

- LiNR variants ship one file per class — no separate `_triton.py`
  siblings. Each module
  ([`postfilter_knn.py`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py),
  [`postfilter_knn_int8.py`](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py),
  [`prefilter_knn.py`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py),
  [`one_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py),
  [`simhash_knn.py`](../../retrieve/src/retrieve/layers/linr/simhash_knn.py))
  takes a `backend=` flag in `__init__`. The two
  bit-KNN files are thin subclasses of the shared
  [`_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/_bit_knn.py)
  base. Construct directly (`OneBitKNN(k=..., backend="triton")`);
  there's no version-numbered builder, and v1/v2/v3/v4 naming lives only
  at the evaluation-algo layer.
- The SilverTorch builder (`build_silvertorch`) lives next to its class
  in [`layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/).
- Quantizers ship from [`layers/utils/quantize.py`](../../retrieve/src/retrieve/layers/utils/quantize.py):
  `quantize_int8` / `quantize_int8_global` (consumed by `SilverTorch`),
  `quantize_oporp_1bit` + `project_oporp_1bit_query` (consumed by
  `OneBitKNN`; both wrap the shared `_oporp_project` core), and
  `quantize_simhash_1bit` + `project_simhash_1bit_query` (consumed by
  `SimHashKNN`). `PostfilterKNNInt8` keeps a private codes-only
  `_quantize_int8_global` variant in its own file. No caller crosses
  families: the bit-KNNs never see INT8, `SilverTorch` never sees
  OPORP/SimHash.
