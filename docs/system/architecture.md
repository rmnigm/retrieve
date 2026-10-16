---
title: architecture
created: 2026-09-26
updated: 2026-10-15
type: entity
tags: [library]
sources: [retrieve/src/retrieve/]
---

# `retrieve` architecture

This is the high-level map of the `retrieve` package (distribution
`torchretrieve`, not yet on PyPI): what each retrieval module does, how filters compose, and
which kernels back which paths. For per-kernel detail (launch grids, tile
shapes, autotune keys, numerics) see [kernels.md](kernels.md).

**What ships** ([decisions](../../research/decisions.md#goal-and-scheduling)): the
release carries LiNR V1-V3 with their filters and kernels, and `SilverTorch` on Meta's official
SilverTorch code, adapted (`backend="official-fork"`, the default; the `official-fork/` workspace
member, to be renamed). `backend="official"` (Meta's code as published) stays as the
parity reference. Our earlier Triton reimplementation of SilverTorch and its eager reference are
deleted ([kernels](kernels.md#deleted-backends)).

## Package layout

Two public layers, mirrored from Meta's `silvertorch` package — `modules`
(the `nn.Module`s you instantiate) and `ops` (registered kernels, one
namespace per backend) — plus two small helper namespaces Meta has no
equivalent for: `indexing` (index-*build*-time math) and `functional`
(query-time torch glue that is not a kernel).

```
retrieve/src/retrieve/
├── __init__.py            modules.* re-exported; LinrBackend, SilverTorchBackend; RetrievalModule, FilterModule
├── interfaces.py          the two backend literals, check_backend, the two ABCs, ops_for(backend), DISPATCH, load_prebuilt
├── functional.py          two_level_topk, masked_topk, counts_to_valid, compact_mask, combine_masks,
│                          combine_indices, post_filter_topk, popcount_int64, clause_subset_match
├── modules/
│   ├── silvertorch.py     SilverTorch (Algorithm 1 on Meta's ops; official-fork | official), SilverTorchBuilder, OfficialConfig
│   ├── linr.py            LiNRV1–LiNRV3 (the paper variants over the primitives + a filter), LiNRBuilder
│   ├── knn.py             PostfilterKNN, PrefilterKNN, FullScanKNN
│   ├── bit_knn.py         OneBitKNN, SimHashKNN (+ the _PackedBitsKNN base)
│   ├── filters.py         ExactAttributeFilter
│   └── official.py        Meta's silvertorch.modules.* re-exported lazily (OfficialMissing without the extra)
├── ops/
│   ├── __init__.py        triton / reference / official as lazy submodules; available_backends()
│   ├── triton/            __init__.py (imports every kernel file → registers retrieve::*), _host.py (shared
│   │                      launch scaffold), common.py (@triton.jit helpers), one file per kernel
│   ├── reference/         the same op names and signatures in pure torch: the "torch" backend + parity oracle
│   ├── official/          __init__.py (loader per backend, OfficialConfig, constants, `st` / `stfork`), adapter.py (the adapter)
│   └── tune.py            the autotune CLI (`tune-kernels`)
└── indexing/              kmeans.py (KMeans, init random | kmeans++), ivf.py (csr_layout, probe_width), quantize.py
```

`import retrieve` imports no kernel: a module resolves its backend's op
namespace through [`interfaces.ops_for`](../../retrieve/src/retrieve/interfaces.py)
at construction (which is when `retrieve.ops.triton` registers the eight
`torch.ops.retrieve.*` ops, or `retrieve.ops.official` loads Meta's
extension) and again in `forward` (a dict hit — the namespace is never
stored on the instance, which keeps modules deep-copyable and picklable).
`tests/test_public_api.py` pins `retrieve.__all__`, `retrieve.modules.__all__`
and the no-kernel-on-import property.

## Module families

Two retrieval families live side by side, both implementing the
[`RetrievalModule`](../../retrieve/src/retrieve/interfaces.py) interface
and selecting their compute path via a `backend=` flag on `__init__`
(two literal aliases are exported from
[`interfaces.py`](../../retrieve/src/retrieve/interfaces.py):
`LinrBackend = Literal["torch", "triton"]` for the LiNR layers and the
standalone filters, `SilverTorchBackend = OfficialBackend = Literal["official", "official-fork"]`
(its tuple `OFFICIAL_BACKENDS`) for `SilverTorch`; every constructor validates its own):

| `backend=` | meaning | who accepts it |
|---|---|---|
| `"triton"` | fused Triton kernels. The default. | every module but `SilverTorch` |
| `"torch"` | pure-torch eager equivalent, same semantics, larger intermediates | every module but `SilverTorch` |
| `"official"` | Meta's own `meta-recsys/silvertorch` ops (`torch.ops.st.*`) for phases 2+3 — the reference backend; eager-only; needs the `official` extra | **`SilverTorch` only** |
| `"official-fork"` | Meta's code adapted ([`official-fork/`](../../official-fork/CHANGES.md), package `silvertorch_fork`, `torch.ops.stfork.*`) through the same adapter; `SilverTorch`'s default; loads beside `"official"` in one process; CUDA-graph capturable and compilable ([validation](../../research/validation.md#meta-fork-backendofficial-fork)); needs the `official` extra | **`SilverTorch` only** |

The `"official"` backend needs an optional extra: `torchretrieve[official]`
(`uv sync --extra official`) installs Meta's own `meta-recsys/silvertorch`,
pinned by sha in the workspace root's `[tool.uv.sources]`, which builds a
CUDA extension and registers nine `torch.ops.st.*` ops. The adapter is the
`retrieve.ops.official` package; without the package,
`SilverTorch(backend="official")` raises `OfficialMissing`. Installing it
needs an nvcc that matches the torch wheel (`CUDA_HOME=/usr/local/cuda-12.8`
for the cu128 wheel — upstream's README insists on the match), `ninja` and
`setuptools`; see
[../artifacts/official-silvertorch/README.md](../../research/artifacts/official-silvertorch/README.md)
for the pin, the build record and the upstream-suite result.

`"official"` / `"official-fork"` are not universal paths: they exist solely for
`SilverTorch`, which takes nothing else. Every other class takes `LinrBackend` and raises
`ValueError` on anything else, both official backends included, so there is no silent fallback. See
[Backend dispatch](#backend-dispatch) below.

- **LiNR** ([`modules/knn.py`](../../retrieve/src/retrieve/modules/knn.py),
  [`modules/bit_knn.py`](../../retrieve/src/retrieve/modules/bit_knn.py)) — four
  primitives (`PostfilterKNN` dense fp16-input, `PrefilterKNN` sparse pre-filter, `OneBitKNN` 1-bit Sign-OPORP,
  `SimHashKNN` 1-bit SimHash; the two bit-KNNs share the
  [`_PackedBitsKNN`](../../retrieve/src/retrieve/modules/bit_knn.py)
  base, which owns scoring/dispatch while subclasses own bit production).
  Filtering is **decoupled**: each forward takes a mask or a candidate-id
    buffer the caller computed via a
  [`FilterModule`](../../retrieve/src/retrieve/interfaces.py)
  ([`ExactAttributeFilter`](../../retrieve/src/retrieve/modules/filters.py)),
  [`compact_mask`](../../retrieve/src/retrieve/functional.py), or
  any upstream cascade composed via
  [`combine_masks` / `combine_indices`](../../retrieve/src/retrieve/functional.py).
  The paper's variants V1–V3 are shipped as compositions of these in
  [`modules/linr.py`](../../retrieve/src/retrieve/modules/linr.py) —
  `LiNRV1`–`LiNRV3`, each holding its filter as the `filter` submodule —
  with the same `prepare_queries(query_clause_attrs)` → `forward(query,
  prepared=None)` pair as `SilverTorch` (see [LiNR variants](#linr-variants)).
- **SilverTorch** ([`modules/silvertorch.py`](../../retrieve/src/retrieve/modules/silvertorch.py)) —
  IVF + INT8 ANN with an inline attribute filter selected by
  `filter_mode ∈ {"none", "bloom", "exact"}`: our k-means, quantization and probe selection,
  then phases 2+3 of Algorithm 1 on Meta's ops ([kernels](kernels.md#silvertorch-on-metas-ops)).
  Filtering is **inline**: the bloom index or the clause attrs live on the module; no standalone
  `ExactAttributeFilter` instance is wired in.

The filter module is [`modules/filters.py`](../../retrieve/src/retrieve/modules/filters.py)'s
`ExactAttributeFilter` (exact, supports reverse); our LiNR bloom filter is deleted
([kernels](kernels.md#deleted-backends)). The `combine_masks` / `combine_indices` composition
helpers and the torch-side exact predicate (`clause_subset_match`) are in
[`functional.py`](../../retrieve/src/retrieve/functional.py). It subclasses
[`FilterModule`](../../retrieve/src/retrieve/interfaces.py), whose
`register_index(item_clause_attrs, *, clause_is_reverse=None)` signature is keyword-only after
the first argument. The rest of the query-time glue — `compact_mask`, `post_filter_topk`, the
shared masked top-K epilogue (`masked_topk`, `counts_to_valid`) and the torch `popcount_int64` —
is [`functional.py`](../../retrieve/src/retrieve/functional.py); the index-build helpers —
`KMeans`, the IVF layout, the quantizers (`quantize_int8` / `quantize_int8_global`,
`quantize_oporp_1bit`, `quantize_simhash_1bit`) — are
[`indexing/`](../../retrieve/src/retrieve/indexing/).

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

- `evaluate_mask(query_clause_attrs, start=0, end=None) → [B, end - start]
  bool` — dense path over items `[start, end)` (a row slice of the item
  table; [filtering](filtering.md#item-range-masks)), fed to
  the mask-taking dense layers (`PostfilterKNN`,
  which masks scores before top-K) or compacted via `compact_mask` for
  the candidates-taking layers. `ExactAttributeFilter` routes to the
  fused `clause_mask` Triton kernel on the `"triton"` backend (no
    `[B, N, C, A_max]` intermediate), pure-torch broadcast on `"torch"`.
- `prepare_queries(query_clause_attrs)` → the filter's query encoding (the int64 attrs), done
  once per batch outside any timed call; every eval path below takes it.
- `evaluate_indices(q) → (positive_indices[B, N] int64,
  counts[B] int64)` — sparse path, used by the callers that feed
  `PrefilterKNN` / the bit-KNNs' candidates path. The returned index
    buffer is **full-width** `[B, N]`: only the first `counts[b]` entries
  of each row are meaningful (an unwritten tail on the kernel path, arbitrary argsort tails on
  `compact_mask`), and within-row order is ascending item order on both backends.
    `ExactAttributeFilter` routes to the fused `clause_compact` Triton kernel on the `"triton"`
  backend; the `"torch"` backend takes `compact_mask(evaluate_mask)`.
- `evaluate_subset(query_clause_attrs, candidate_ids) → [B, P] bool` —
      apply a filter only to the given candidate ids. `ExactAttributeFilter` overrides this with
  gather + `clause_subset_match` ([`functional.py`](../../retrieve/src/retrieve/functional.py));
  the default is `evaluate_mask(...).gather(1, candidate_ids)`.

Callers that already have a bool mask from some other source (a hand-rolled
predicate, an external mask passed through the API) use
[`compact_mask`](../../retrieve/src/retrieve/functional.py)
directly to convert it to the `(positive_indices, counts)` form V2 wants.

Two composition helpers ship in
[`functional.py`](../../retrieve/src/retrieve/functional.py):

- `combine_masks(*masks)` — element-wise AND of N optional `[B, N]` masks,
  None-tolerant; one torch op, no kernel.
- `combine_indices(filters, queries)` — sparse cascade. The first filter
  produces `(ids, counts)` via its native compact path; each subsequent
  filter is invoked via `evaluate_subset` on those ids and the survivors
  are re-compacted. No `[B, N]` from the cascade itself. Caller orders
  filters most-selective first.

## LiNR variants

The paper's V1–V3 are modules of their own
([`modules/linr.py`](../../retrieve/src/retrieve/modules/linr.py)), each a
composition of the primitives below plus an optional `FilterModule`
attached at construction as `filter=` (a submodule, so `buffers()` and a
state dict cover index and filter; its buffers carry the `filter.` prefix).
All three share `register_index(item_embs, item_clause_attrs=None,
clause_is_reverse=None)` — which registers the filter too when attributes
are given — `prepare_queries(query_clause_attrs)` (the filter's encoding)
and `forward(query, prepared=None) -> (ids [B, k], scores [B, k])`; `k` forwards to the primitive that owns the final top-k and
is settable after registration; `capturable = True` is a class attribute
(every LiNR backend captures):

| class | composition | `forward` |
|---|---|---|
| `LiNRV1(k, *, filter=None, backend)` | `PostfilterKNN` (`idx`, its `score`) + `filter.mask_scores` | dense dot (fp16 inputs, fp32 scores), the filter folded into one pass writing `score` or `-inf`, top-k (`masked_topk(masked=True)`) |
| `LiNRV2(k, *, filter, backend)` | `PrefilterKNN` (`idx`) over `filter.evaluate_indices` | `query_clause_attrs` required — the filter is the candidate source |
| `LiNRV3(k, *, candidate_pool=5000, seed=0, filter=None, backend)` | `OneBitKNN(k=candidate_pool)` (`stage1`) → `PrefilterKNN(k)` (`stage2`); the filter's candidates feed stage 1, stage 2 is bounded by the survivors' count | `set_query_params(candidate_pool=…)` re-validates against `N` |

`LiNRBuilder(variant, **kwargs)` (`variant ∈ {"v1", "v2", "v3"}`,
`kwargs` the class's) builds one: `.set_item_embeddings(x)` and
`.set_filter(filter, item_attrs=None, clause_is_reverse=None)` for a fresh
index, or `.set_state_dict(sd)` (plus `.set_filter(filter)` so the `filter.`
buffers have a home) for a prebuilt one, then `.set_backend(...)`,
`.set_device(...)`, `.build()`. Both builders share
[`interfaces.load_prebuilt`](../../retrieve/src/retrieve/interfaces.py): one
buffer per state-dict key (shape and device from the saved tensor), then
`load_state_dict`, so every load hook re-derives what a forward caches from
a buffer — `SilverTorch`'s two scalars and `OneBitKNN`'s `k_bits` sentinel
(from `oporp_signs`).

The primitives all return `(ids[B, K], scores[B, K])`; `-1` / `-inf` are the
"no item" sentinels for masked-out or short rows. Each module takes a
`backend=` arg in `__init__` (`LinrBackend`: `"official"` is rejected here —
see [Backend dispatch](#backend-dispatch)); `"triton"` is the default
(the eval harness and the original paper experiments target Triton). The
torch backend is eager — callers wanting Inductor fusion or cudagraph
capture wrap the module with `torch.compile` themselves. The library does
not bind compile internally. The torch-side "mask to `-inf` → topk →
map ids → `-1`-sentinel non-finite winners → pad to k" epilogue is the
shared [`masked_topk`](../../retrieve/src/retrieve/functional.py)
utility, not per-class copies.

- **`PostfilterKNN` — dense similarity, optional mask.** Full
  `query @ item_embs.T`, masked scores set to `-inf`, then top-K. Forward
  takes `(query, mask=None)`. There is no Triton kernel because cuBLAS +
  CUB already deliver the same memory traffic; the
  `backend=` flag is accepted for API symmetry but is a no-op on this
  class.
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
  [`_PackedBitsKNN`](../../retrieve/src/retrieve/modules/bit_knn.py):
  score = `64·W − 2·popcount(query_bits ^ item_bits)` over packed int64
  sign bits — purely bitwise, `k_bits/8` bytes per item, ~16× smaller
  than fp16. The base owns `register_index` (asserts `k <= N`), backend
  dispatch, and the two forward paths; subclasses own only the index
  quantizer and the query projection:
  - `OneBitKNN(k, seed=0, backend="triton", k_bits=0)` —
    [`quantize_oporp_1bit`](../../retrieve/src/retrieve/indexing/quantize.py)
    builds `item_bits[N, W]` int64, `oporp_signs[D]` int8,
    `oporp_perm[D]` int64. `k_bits=0` is a sentinel resolved to `D` at
    `register_index` (re-resolved from the pristine constructor arg on
    every registration).
  - `SimHashKNN(k, k_bits, seed=0, backend="triton")` — fixed Gaussian
    projection `simhash_R[k_bits, D]` then sign-pack
    ([`quantize_simhash_1bit`](../../retrieve/src/retrieve/indexing/quantize.py));
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

[`SilverTorch`](../../retrieve/src/retrieve/modules/silvertorch.py) implements the SilverTorch
paper's Algorithm 1 on Meta's ops: IVF clustering over INT8-quantized item codes (one global
per-tensor scale, paper §3.2) with an inline attribute predicate. Two axes are chosen at
construction: `filter_mode` picks the predicate, `backend` the ops.

The predicate, `filter_mode ∈ {"none", "bloom", "exact"}`:

- `"none"`: plain IVF + INT8 ANN, no attribute filter.
- `"bloom"`: Meta's bloom index (their murmur3 hash over `(clause, value)` features, bundles of
  2048 docs) queried through Meta's expression DSL; `k_hash` is the search `k` (≤ 10), `m_bits`
  the width per doc on the adapted code (optional and ignored on `official`, whose width is
  `OfficialConfig.b_multiplier`'s). The adapted code rechecks the bloom's survivors exactly by
  default (`OfficialConfig.recheck`), so it returns no false positive.
- `"exact"`: the AND-of-OR clause predicate over the `[N, C, A_max]` narrow attrs, reverse
  clauses supported: tested inside the adapted search op, or packed by our `clause_mask_packed`
  into the published scorer's bit mask.

The ops, `backend ∈ {"official-fork", "official"}` ([kernels](kernels.md#silvertorch-on-metas-ops)):

- `"official-fork"` (default): Meta's code adapted (`torch.ops.stfork.*`, the `official-fork/`
  workspace member, its changes in [CHANGES](../../official-fork/CHANGES.md)): one `search` op
  after phase 1, bloom plans prepared once on the device, no host sync in the forward, so it
  captures and compiles.
- `"official"`: Meta's code as published (`torch.ops.st.*`, pinned at `21aa35e`), the parity
  reference: eager-only, `torch.compile` of it raises. Its `OfficialConfig(bloom_path=...)` picks
  the partial-mask co-design or the full-`N` mask ablation; the adapted code has the partial path
  only and refuses `"full"`.

Both take `official=OfficialConfig(...)`: `score_path` `"fp16"` (default, the shipped int8
serving path) or `"int32"` (the bit-exact parity path), `bloom_path`, `b_multiplier`,
`n_stored_hashes`, `bloom_width`, `recheck`, … (the dataclass's docstring).

Constructed directly (`SilverTorch(k, n_lists, n_probe, filter_mode="none", m_bits=None,
k_hash=None, n_iter=10, seed=0, kmeans_init="random", backend="official-fork", official=None)`;
`kmeans_init="kmeans++"` opts into D² seeding, the default stays random because every recorded
number was taken on it, [decisions](../../research/decisions.md#library)) or through
`SilverTorchBuilder(**those)` → `.set_item_embeddings(x)` [+ `.set_item_attributes(attrs,
clause_is_reverse)`] or `.set_state_dict(sd)` → [`.set_backend(b, official=None)`,
`.set_device(d)`] → `.build()`, which is construct → `register_index` (or the prebuilt load, no
k-means) → `.to(device)`. After `register_index`, `build_timings` is `{"kmeans_s", "assemble_s",
"quantize_s", "filter_s"}` (device-synchronised wall seconds of the four phases; `{}` before
registration and on a prebuilt module) and `set_query_params(n_probe=…)` changes the probe width
with the two `register_index` validations re-run (`n_probe <= n_lists`, a probe width of at least
`k`: the sum of the `n_probe` largest clusters, [kernels](kernels.md#silvertorch-on-metas-ops));
`k` is a plain attribute. `capturable` is a property: `True` on `official-fork`, `False` on
`official`.

Forward takes `(query, prepared=None, candidate_ids=None)`, `prepared` from
`prepare_queries(query_clause_attrs)` (a `PreparedFilter`: Meta's parsed plans, uploaded once on
the adapted code, or the attrs; the query side of the filter, done outside the timed call); there
is no mask parameter. `prepared=None` is a documented fast path that skips the predicate; passing
`prepared` together with `candidate_ids` raises `ValueError` (the candidates path scores the
given candidates *without* the filter, so accepting both would silently drop the predicate). On
that path `-1` candidate ids are padding: a compaction's output (`evaluate_indices`,
`compact_mask`) must be masked to `-1` past its `counts` first, since its tail is unwritten or
argsort leftovers; pads are gathered as `clamp_min(0)` but masked to `-inf` and returned as `-1`
through `masked_topk`, so a pad never scores or surfaces as `inv_perm[-1]`.

`register_index` is validate → `_build_ivf` (k-means, registers `centroids`) → the layout
([`indexing.csr_layout`](../../retrieve/src/retrieve/indexing/ivf.py), one stable `argsort` of
the assignment) → `_quantize_items` → `_register_filter_buffers`, with a frozen
buffer-registration order (state-dict key order): `centroids`, `item_codes`
(**cluster-sorted**), `global_scale`, `cluster_offsets[n_lists+1]`, `cluster_sizes`,
`sort_perm[N]` (sorted position → original id), `inv_perm[N]`, then the filter buffers, also in
cluster-sorted order:

- `"bloom"`: Meta's index, `bloom_index[W]` int64 and `bundle_b_offsets[n_bundles+1]` from
  `bloom_index_build` (`bloom_index_build_fixed` at the configured `m_bits` width on the adapted
  code) over the sorted attrs; where the survivors are rechecked (the adapted code's default),
  the exact attrs beside it. Registered without attributes, the index is empty and a later
  attribute query raises.
- `"exact"`: `item_clause_attrs` and `clause_is_reverse[C]` bool. On `official` the attrs are
  the `[N, C, A_max]` int64 table our `clause_mask_packed` reads; on `official-fork` the adapted
  code's clause-major int32 slot layout (`fork_clause_attrs`, with `clause_slots[C + 1]`;
  [kernels](kernels.md#official-fork--our-fork-of-metas-ops)).
- `"none"`: no filter buffers; `prepare_queries` raises and `forward` requires `prepared=None`.

> **State dicts are portable between the two backends in `"none"`**: both register the same CSR.
> The exact attrs differ in layout and the bloom index in width, so a filtered module is rebuilt
> with `register_index` across the boundary. Loading is `nn.Module.load_state_dict` into a module
> whose `register_index` already ran (or `load_prebuilt` / the builder's `set_state_dict`). A
> `load_state_dict` post-hook re-derives the two Python scalars the forwards read instead of the
> buffers, `_global_scale_f` (from `global_scale`) and `_probe_width` (from `cluster_sizes` and
> `n_probe`), with two `.item()` syncs at load time, so a loaded index scores exactly like the
> saved one.

## Utility modules

- [`FullScanKNN`](../../retrieve/src/retrieve/modules/knn.py) (in
  `modules/knn.py` next to the LiNR primitives) —
  exhaustive `query @ item_embs.T` + top-K with optional post-mask or
  candidate_ids (`-1` = padding, never scored or returned — the same
  `masked_topk` epilogue as `SilverTorch`'s candidates path), used as a
  baseline / sanity check. The mask implements
  **post-filter** semantics (the LiNR baseline): top-K is selected over
  the full corpus first, then masked winners are tombstoned to `-1` —
  not backfilled — so recall vs a pre-filter oracle is < 1 by design
  (documented on the class docstring).
- [`post_filter_topk`](../../retrieve/src/retrieve/functional.py) —
  applies a post-filter to already-computed top-K results, returning
  `(ids, counts)`; `FullScanKNN.forward` discards the counts.
- [`masked_topk` / `counts_to_valid`](../../retrieve/src/retrieve/functional.py) —
  the shared torch-side masked top-K epilogue (pure tensor-flow, traces
  cleanly under the eval harness's `torch.compile(mode="reduce-overhead",
  dynamic=False, fullgraph=True)`).
- [`KMeans`](../../retrieve/src/retrieve/indexing/kmeans.py) — pure-torch
  Lloyd's k-means used by
  `SilverTorch` for IVF index building (not a `RetrievalModule`);
  `KMeans(n_lists, n_iter=10, seed=0, init="random" | "kmeans++")`,
  `fit(embs) -> (centroids, assignments)`, `assign(embs, centroids)`.
  `init="kmeans++"` is greedy D² sampling — one `mv` over the index per
  centroid, the uniform draws from the CPU generator located by
  `searchsorted` on the device-side cumsum, no host sync per step (9.5 s
  against 0.8 s random at N = 3 M × 128, `n_lists = 8192`).
  `fit` is **bit-for-bit reproducible
  run to run**: the centroid update reduces with a float64 one-hot GEMM
  accumulated panel by panel, not `index_add_`'s floating-point atomics,
  whose scheduling-dependent order makes two seed-0 builds differ by 1.8e-2
  in the centroids and puts every SilverTorch quality number out of reach of
  the golden gate's 1e-6. It costs ~1.3× the wall time of the atomic
  reduction at N=200k, D=128, n_lists=1024, bounded by one extra
  assignment-sized matmul per Lloyd iteration.
- Two abstract bases plus the two backend literals and their validator in
  [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py):
  `RetrievalModule` (a minimal lifecycle ABC — `k` attr + abstract
  `register_index`, called exactly once; forward signatures deliberately
  unconstrained), `FilterModule`, `LinrBackend = Literal["torch",
  "triton"]`, `SilverTorchBackend = OfficialBackend = Literal["official",
  "official-fork"]`, `check_backend(backend, literal)` (every constructor
  calls it — see [Backend dispatch](#backend-dispatch)),
  `ops_for(backend)` (the backend → op-namespace resolver), `DISPATCH` (the
  dispatch table as data) and `load_prebuilt` (the builders' state-dict
  path). All retrieval layers
  (`PostfilterKNN`, `PrefilterKNN`,
  `_PackedBitsKNN` and its two subclasses, `SilverTorch`, `FullScanKNN`,
  `LiNRV1`–`LiNRV3`) subclass `RetrievalModule`.
- [`modules/official.py`](../../retrieve/src/retrieve/modules/official.py) —
  Meta's own `silvertorch.modules` classes (`BloomIndexSearchModule`,
  `FilterQueryParserModule`, their builders), fetched on first attribute
  access after `retrieve.ops.official.ensure_loaded()`; without the extra
  the access raises `OfficialMissing` with the install hint. Nothing in the
  library calls them — `SilverTorch(backend="official")` uses the ops.

## Kernels

The Triton kernels live flat under
[`ops/triton/`](../../retrieve/src/retrieve/ops/triton/), one file per
kernel, plus
[`common.py`](../../retrieve/src/retrieve/ops/triton/common.py) —
shared `@triton.jit` building blocks called from the kernel bodies
([kernels.md](kernels.md#shared-kernel-helpers-opstritoncommonpy)) — and
[`_host.py`](../../retrieve/src/retrieve/ops/triton/_host.py), the plain-Python
launch scaffold the files share (`compact_finish` for the compaction op, `grid_batch_tiles` for
the 3-D grids of the filter kernels, the boundary checks and `wide`); each kernel keeps its own
loads and masking policy.
The package [`__init__.py`](../../retrieve/src/retrieve/ops/triton/__init__.py) is the one
place the kernel files are imported, so `import retrieve.ops.triton` is
the registration. Every kernel of ours is Triton (see
[kernels.md](kernels.md#deleted-backends)). Each op has a pure-torch
twin of the same name and signature in
[`ops/reference/`](../../retrieve/src/retrieve/ops/reference/) — the
`"torch"` backend and the oracle the parity suite scores the kernel
against.

Each Triton path is **one launch per `forward()`** followed by a
host-side `torch.topk` over the score buffer (CUB beats anything we can
write in pure Triton). `SilverTorch` runs no kernel of ours:
[`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py)
adapts Meta's `torch.ops.st.*` ops as published (eager-only: 19 launches and 3 host syncs per
unfiltered forward on the A100) and the adapted `torch.ops.stfork.*` (one `search` op, no sync).
Seven of the eight ops are registered with `@torch.library.triton_op` so inductor can see the
`@triton.jit` body; the stream-compaction kernel `clause_compact` is an opaque
`@torch.library.custom_op` instead, because its data-dependent store address makes inductor's
mutation analysis flag the index buffers and cudagraph trees skip the compiled forward. Full
per-kernel detail, and that mechanism, in [kernels.md](kernels.md).

| namespace | op (`retrieve::*` in `ops.triton`; same name in `ops.reference`) | consumer |
|---|---|---|
| `ops.triton` | [`fused_masked_knn_topk`](../../retrieve/src/retrieve/ops/triton/fused_masked_knn_topk.py) — gather + dot over `positive_indices` | `PrefilterKNN` (candidates path) |
| `ops.triton` | [`oporp_1bit_match_topk_full` / `_indirect`](../../retrieve/src/retrieve/ops/triton/oporp_1bit_match_topk.py) — XOR + popcount, both bit-KNN paths | `OneBitKNN` / `SimHashKNN` |
| `ops.triton` | [`gemv_scores`](../../retrieve/src/retrieve/ops/triton/gemv_scores.py) — one query's dense scores in cuBLAS's `gemv2N` order | `PostfilterKNN` at `B = 1`, where `register_index` found it bit-exact |
| `ops.triton` | [`clause_compact`](../../retrieve/src/retrieve/ops/triton/clause_compact.py) — fused clause eval + stream compaction | `ExactAttributeFilter.evaluate_indices` |
| `ops.triton` | [`clause_mask`](../../retrieve/src/retrieve/ops/triton/clause_mask.py) — fused clause eval emitting `[B, N]` bool | `ExactAttributeFilter.evaluate_mask` |
| `ops.triton` | [`clause_mask_scores`](../../retrieve/src/retrieve/ops/triton/clause_mask.py) — the same pass writing `scores` or `-inf` | `ExactAttributeFilter.mask_scores` (`LiNRV1`) |
| `ops.triton` | [`clause_mask_packed`](../../retrieve/src/retrieve/ops/triton/clause_mask.py) — the same pass packed into Meta's `filtering_bit_mask` | `SilverTorch(backend="official", filter_mode="exact")` |

| `ops.official` | [`official_probe_score`, `fork_search`, `bloom_partial_masks`, …](../../retrieve/src/retrieve/ops/official/adapter.py) — adapter over Meta's `torch.ops.st.*` (as published, eager-only) and `torch.ops.stfork.*` (adapted); no kernel of ours | `SilverTorch` |

### Backend dispatch

Every module resolves `backend` to an op namespace with
`interfaces.ops_for` — `"triton"` → `retrieve.ops.triton`, `"torch"` →
`retrieve.ops.reference` (identical op names and signatures, so a module's
forward is one code path calling `ops.<op>(...)`) — and `SilverTorch` takes
`"official"` / `"official-fork"` only, through `retrieve.ops.official`, its `forward`
branching on `backend` to `_forward_official` / `_forward_fork`.
The constructor rejects anything else.
The importable truth is
[`interfaces.DISPATCH`](../../retrieve/src/retrieve/interfaces.py):
`{class name: {backend: "triton" | "torch" | "cublas" | "official" | None}}`,
`"cublas"` where both backends score with cuBLAS (on triton, a single query takes the
`gemv_scores` GEMV where it equals cuBLAS bit for bit, so the two still match) and `None`
where the constructor raises
(`tests/correctness/test_boundary.py` checks that every `None` raises and
every key is a `retrieve.modules` class). The same table, with the op each
label stands for:

| module | `"triton"` | `"torch"` | `"official"` / `"official-fork"` |
|---|---|---|---|
| `SilverTorch` | raises `ValueError` | raises `ValueError` | Meta's ops: `official` as published (eager-only, the parity reference), `official-fork` adapted (the default) |
| `LiNRV2` / `LiNRV3` | the primitives' Triton ops below | eager (`ops.reference`) | raises `ValueError` |
| `LiNRV1` | cuBLAS; one query on `gemv_scores` where `register_index` found it bit-exact ([kernels](kernels.md), `gemv_scores`); a filter cell adds the filter's own path | cuBLAS | raises `ValueError` |
| `PrefilterKNN` | `fused_masked_knn_topk` | eager (`ops.reference`) | raises `ValueError` |
| `OneBitKNN` / `SimHashKNN` | `oporp_1bit_match_topk` | eager (`ops.reference`) | raises `ValueError` |
| `ExactAttributeFilter` | `clause_mask` / `clause_compact` | eager (`ops.reference`) | raises `ValueError` |

| `PostfilterKNN` | cuBLAS; one query on `gemv_scores` where bit-exact | cuBLAS | raises `ValueError` |

A cell labelled `backend="official"` for anything other than
`SilverTorch` therefore cannot exist. The eval harness mirrors that
rather than catching the error: its `PATHS` table maps every
`(algo, filter_kind, backend)` to the code path that actually runs
(`cublas`, `triton`, `torch`, `cublas+triton`, `official`, or `None`),
collapses backends that run the same code into one job, and builds the
standalone filter modules for `official` cells with `backend="triton"` —
see [evaluation.md](evaluation.md#algorithms-and-the-paths-table).
`PATHS` is derived from `DISPATCH`
([`bench/algos.py`](../../evaluation/bench/algos.py)), and
[`tests/bench/test_paths.py`](../../evaluation/tests/bench/test_paths.py)
pins the derivation.

## Testing

Tests live in [`retrieve/tests/`](../../retrieve/tests/) and split by purpose,
not by module: [`correctness/`](../../retrieve/tests/correctness/) for module-
level semantics against torch baselines, [`parity/`](../../retrieve/tests/parity/)
for Triton kernel vs pure-torch agreement, and
[`compile/`](../../retrieve/tests/compile/) for `torch.compile`
graph-break and `torch.export` kernel-reference gates, plus the CPU-only
[`test_public_api.py`](../../retrieve/tests/test_public_api.py) (the `__all__`
lists and the no-kernel-on-import property; `pytest.mark.cpu` exempts it
from the gate). The suite is otherwise GPU-only and skipped
without CUDA via the root [`conftest.py`](../../retrieve/tests/conftest.py)
gate. Performance characterization (latency, memory, recall sweeps) lives in
[`evaluation/`](../../evaluation/), not in `tests/`. Full reference in
[testing.md](testing.md).

## Module layout

The layout is by role ([Package layout](#package-layout)), with these
conventions: the version-numbered classes are compositions only — `LiNRV1`–`LiNRV3` own no kernel and no quantizer, the
primitives keep `register_index` as their only build path and the builders
call it; no caller crosses quantizer families (the bit-KNNs never see INT8,
`SilverTorch` never sees OPORP/SimHash); a kernel file is reached with
`from retrieve.ops.triton.<kernel> import ...` (its `_impl` and `Config`),
while the package attribute `retrieve.ops.triton.<kernel>` is the op of
the same name. Op schemas, buffer names and registration order are
stable, so a state dict written by an earlier release loads into the
current modules ([decisions](../../research/decisions.md#library)); state is added,
never renamed (the composites' `filter.` prefix).

## Which page owns what

One home per fact (AGENTS.md rule 4, [coding guidelines](../../research/contracts/coding-guidelines.md)):
when a code area's behaviour changes, this is the `docs/system` page to
update in the same commit.

| code area | page |
|---|---|
| `retrieve/src/retrieve/ops/`, kernel internals, the tuner | [kernels.md](kernels.md) |
| `retrieve/src/retrieve/modules/`, `interfaces.py`, package layout | this page |
| filter semantics (`modules/filters.py`, the clause predicate) | [filtering.md](filtering.md) |
| `retrieve/tests/` | [testing.md](testing.md) |
| `evaluation/bench/` (the harness CLI, `PATHS`, the cell loop) | [evaluation.md](evaluation.md) |
| `evaluation/eval_datasets/` (ETL, Hub transfer) | [datasets.md](datasets.md) |
| training (`evaluation/training/`) and checkpoints | [checkpoints.md](checkpoints.md) |
| where the harness keeps data, its environment variables | [storage.md](storage.md) |
| standing decisions and constraints (why, not what) | [decisions.md](../../research/decisions.md) |
| what gates pass and the measured results that stand | [validation.md](../../research/validation.md) |
