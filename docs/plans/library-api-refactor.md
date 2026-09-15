# `torchretrieve` 0.2 — the Meta-shaped API: `modules` / `ops` / `indexing`

> **Status:** planned 2026-09-15 on `development` at `76f8985` (nothing
> implemented, no code touched). Authored from three inventories taken the
> same day against that commit: this package file by file, the harness file
> by file, and Meta's [meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch)
> at `main` (the official-integration plan pins `21aa35e`; `main` has since
> grown two `*ModuleBuilder` classes and nothing else moved). Every "today"
> claim cites a path in this repo; every "Meta does X" claim cites a path in
> their tree. Supersedes the parts of the 2026-09-06
> [library review](architecture-review-2026-09-06-library.md) that were
> deferred to "after B4" (A5 `_host.py`, A9's remaining `__all__` rows, D3,
> D5) — they land inside this plan's work packages.
>
> Question this plan answers: *what does `retrieve` look like when it has the
> same two-level shape as Meta's package — high-level `nn.Module`s with
> builders, low-level registered ops, and nothing else — while still containing
> Meta's own modules and ops (imported and wired in, as B1 already does), our
> LiNR V1–V4, our Triton SilverTorch kernels, and a sensible home for the
> helpers Meta never shipped (k-means / k-means++, quantizers, bloom hashing,
> IVF layouts, top-k glue, the tuner)?*
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 Phase L. Companion
> plans: [evaluation-package-layout.md](evaluation-package-layout.md) (the
> harness side) and [library-harness-boundary.md](library-harness-boundary.md)
> (the contract between the two). This plan amends *paths* in
> [silvertorch-official-integration.md](silvertorch-official-integration.md)
> §5–§8 (table in §9 below) and the `algos.py` shape in
> [evaluation-harness-v2.md](evaluation-harness-v2.md) §3.1; it changes no
> decision in either. There is no Mac: every step runs on the A100 box, and
> "CPU" below means "needs no GPU time" (roadmap A0).

## 1. Where it starts

### 1.1 Ours today (`development` @ `76f8985`, 36 files, ≈ 6,000 lines)

```
retrieve/src/retrieve/
├── __init__.py            20 names: 11 classes, 2 ABCs, LinrBackend, SilverTorchBackend, 6 functions
├── interfaces.py          LinrBackend / SilverTorchBackend Literals, check_backend, RetrievalModule, FilterModule
├── tune.py                452-line autotune CLI, 7 subcommands
├── kernels/               10 registered ops: 8 @triton_op + 2 opaque @custom_op (clause_compact, bloom_compact — C4 fix)
│   ├── common.py          shared @triton.jit helpers
│   ├── filters/           bloom_compact, clause_mask, clause_compact
│   ├── linr/              fused_masked_knn_topk, oporp_1bit_match_topk_{full,indirect}
│   └── silvertorch/       codesigned_probe_score{,_bloom,_exact}, bloom_match, official.py (735 lines, B1's adapter)
└── layers/
    ├── silvertorch/main.py   SilverTorch (639 lines; table dispatch; official / triton / torch) + build_silvertorch
    ├── linr/                 PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, _PackedBitsKNN → OneBitKNN, SimHashKNN
    ├── filters/              BloomFilter, ExactAttributeFilter, combine_masks/indices, bloom_hash.py (salt buffer, build_transposed_sigs)
    └── utils/                KMeansTorch (deterministic, random init), quantize.py, topk.py, compact.py, retrieval.py (FullScanKNN)
```

What the 2026-09-06 session already settled, so this plan does not touch it:
the backend vocabulary is split and validated (`LinrBackend`,
`SilverTorchBackend`, `check_backend`; an unsupported backend raises —
review #5); `SilverTorch.forward` dispatches through a table built in
`__init__` (review A2); the `-1` trap on both candidates paths is closed
(review #3); a `load_state_dict` post-hook re-derives the cached scalars
(review #4); the bloom salt is a registered buffer (B5); the official adapter
is one module with `OfficialConfig` (`score_path`, `bloom_path`,
`cache_plans`, `n_stored_hashes`, …) and `ensure_loaded()` at construction
(B1); `argsort(stable=True)` in `_build_ivf` (B2); `clause_compact` /
`bloom_compact` are opaque custom ops so compiled LiNR V2/V3 capture (C4
fix); k-means is bit-reproducible (C4 fix); the CUDA and CuTe backends are
gone (B4). The library suite is 519 green on the box with the official extra.

What is still true and is this plan's business:

- **The pure-torch `"torch"` backend exists twice**: inline eager code in every
  layer (`SilverTorch._forward_torch_eager`, `_PackedBitsKNN._forward_torch_eager`,
  the `else:` branches of both filters) and a second copy of the SilverTorch
  reference in the test tree (`tests/parity/conftest.py::ref_cps_phase23`).
  Parity gates compare kernels against the test copy, not the shipped one.
- **The layout is by history, not by role.** `kernels/silvertorch/` holds a
  Triton kernel, a standalone bloom kernel *and* the official-op adapter;
  `layers/utils/` holds an index builder, quantizers, a top-k epilogue and a
  retriever (`FullScanKNN`); `layers/filters/bloom_hash.py` holds build-time
  hashing next to the filter module. Meta's tree answers "where is X" from
  X's role (§1.2); ours does not.
- **`KMeansTorch` is random-init Lloyd's** ([kmeans.py](../../retrieve/src/retrieve/indexing/kmeans.py):
  seeded `randperm`); the paper says k-means++ (SilverTorch §4.1). There is
  no k-means++ anywhere.
- **The LiNR paper variants V1–V4 live in the harness**
  ([evaluation/retrieval/algos.py](../../evaluation/retrieval/algos.py):
  `LinrV1`–`LinrV4`, 362 lines with `Silvertorch`), and so do two things the
  library should own: `set_query_params` (on the harness `Silvertorch` and
  `LinrV3` wrappers, not on the layers) and the `capturable` flag (stamped by
  the harness's `build`). The library cannot be said to "contain the LiNR
  versions" while that is so.
- **Small debts the reviews left for "after B4"**: `PostfilterKNNInt8` keeps a
  private `_quantize_int8_global`; `kernels/linr/__init__.py` exports a
  private `_impl`; `OfficialConfig` / `OfficialMissing` / `compact_mask` are
  not in `retrieve.__all__` (review A9); the shared launch/finish/grid
  scaffold (review A5); the pyproject `name` is `retrieve` while PyPI knows
  `torchretrieve` ([retrieve/pyproject.toml](../../retrieve/pyproject.toml)).
- **No `build_timings`** on `SilverTorch` (H §2.3 asked for per-phase build
  time; the harness records one wall number).

### 1.2 Meta's package (`silvertorch/`, verified at `main` 2026-09-15)

```
silvertorch/
├── __init__.py                 license header only — no imports, no __all__
├── modules/
│   ├── __init__.py             __all__ = [BloomIndexSearchModule, BloomIndexSearchModuleBuilder,
│   │                                      FilterQueryParserModule, FilterQueryParserModuleBuilder]
│   ├── bloom_index_search_module.py          nn.Module: buffers bloom_index, bloom_bundle_b_offsets; forward → torch.ops.st.bloom_index_search_batch
│   ├── bloom_index_search_module_builder.py  fluent: set_bloom_index / set_feature_data / set_b_multiplier / set_device → build()
│   ├── filter_query_parser_module.py         stateless nn.Module over parse_expression_query_batch
│   └── filter_query_parser_module_builder.py
└── ops/
    ├── __init__.py             license header only
    ├── _load_ops.py            `import silvertorch.ops._load_ops` registers torch.ops.st.* (one .so, silvertorch._C)
    └── csrc/                   11 registered ops under TORCH_LIBRARY(st): bloom_index_build, bloom_index_search_batch,
                                …_return_partial_response(_multiple), generate_column_info_for_clusters,
                                parse_expression_query_batch, fused_kmean_ann, fused_kmean_ann_with_partial_masks(_multiple),
                                is_topk, take_top_k_and_gather_from_main_and_fresh (the last two are dead source: not in setup.py, A2)
```

Conventions to copy: **two layers and nothing else** (`modules/` = things you
instantiate, `ops/` = registered kernels reached through `torch.ops.<ns>`);
ops become available through one side-effecting import; no `utils`, no
config objects beyond plain constructor arguments; a module holds its index
as registered buffers and its `forward` is a thin call into one op; a
`*Builder` per module separates "build the index from raw data" from "load a
prebuilt index". What they do **not** ship, and we must place ourselves:
k-means (README: "bring your own k-means, sort items by cluster"), any
quantizer, any exact / non-probabilistic filter, any ANN-level module
(nothing wraps `fused_kmean_ann`), any tuner, any Triton.

## 2. Decisions

- **D1 — Two public layers, mirrored from Meta: `retrieve.modules` and
  `retrieve.ops`.** Plus two supporting namespaces Meta has no equivalent for,
  kept deliberately small: `retrieve.indexing` (index-*build*-time math:
  clustering, quantization, hashing, layouts) and `retrieve.functional`
  (query-time torch glue that is not a kernel: masked top-k, compaction,
  mask/index composition, the subset predicates). `retrieve.layers`,
  `retrieve.kernels`, `layers/utils` are gone.
- **D2 — `retrieve.__init__` exports the modules, the two backend literals and
  the two ABCs; the op and helper namespaces are reached as submodules**
  (`retrieve.ops.triton.…`, `retrieve.indexing.…`, `retrieve.functional.…`).
  "Both high-level and low-level primitives exposed, and that's it."
- **D3 — Ops keep their names and schemas.** `retrieve::codesigned_probe_score`
  stays; nothing is renamed to `fused_kmean_ann`-style names. Renaming an op
  changes the `torch.ops` schema, invalidates every tune JSON and parity
  artifact, and buys cosmetics. Module names likewise stay (`SilverTorch`,
  `PostfilterKNN`, …): they are the names in the thesis, the user guide and
  both plans. `LinrBackend` / `SilverTorchBackend` stay as the review shaped them.
- **D4 — Three op namespaces, one per backend, with identical signatures:**
  `ops.triton` (the kernels), `ops.reference` (pure-torch eager: the
  `"torch"` backend and the parity oracle, one home instead of two),
  `ops.official` (B1's adapter over `torch.ops.st.*`, moved). A module
  resolves its backend to a namespace once, in `__init__`, through the table
  dispatch that already exists.
- **D5 — LiNR V1–V4 become library modules** (`modules/linr.py`: `LiNRV1`,
  `LiNRV2`, `LiNRV3`, `LiNRV4`), each owning its filter as a submodule and
  exposing the same forward as `SilverTorch`:
  `forward(query, query_clause_attrs=None) -> (ids, scores)`. Their bodies are
  the harness wrappers' bodies, moved (`LinrV3`'s cascade verbatim). The
  primitives they compose stay public. `set_query_params` moves with them;
  `capturable` becomes a class attribute. This is what makes the harness's
  `algos.py` a table ([library-harness-boundary.md](library-harness-boundary.md) §3).
- **D6 — Builders for the composites only.** `SilverTorchBuilder` and
  `LiNRBuilder` (one class parameterised by variant) follow Meta's fluent
  shape — `set_item_embeddings(...)`, `set_item_attributes(...)`,
  `set_backend(...)`, `set_device(...)`, or `set_state_dict(...)` for a
  prebuilt index — and end in `build()`. Primitives and filters keep
  `register_index` as their only build path. `build_silvertorch(...)` is
  retired; the 0.2 shim keeps it.
- **D7 — Meta's modules and ops are imported and wired, never copied** —
  already the case for the ops (B1). `retrieve.modules.official` re-exports
  the four `silvertorch.modules` classes lazily; `retrieve.ops.official` is
  the existing adapter plus `st = torch.ops.st` after `ensure_loaded()`.
  Both raise `OfficialMissing` naming `torchretrieve[official]` (unchanged).
- **D8 — Structure-preserving first, features second.** WP-1 moves code and
  changes no number: every op keeps its schema, every module keeps its buffer
  names and registration order (state dicts load unchanged), every parity
  gate stays bit-exact, `code_version` changes exactly once. The feature
  items — the LiNR composites, builders, k-means++ init, `build_timings`,
  the boundary clauses — are WP-2 with their own gates.
- **D9 — k-means++ is opt-in until the campaign says otherwise.**
  `KMeans(n_lists, n_iter, seed, init="random" | "kmeans++")`, default
  `"random"`: A1's golden (and its pending re-derivation) and C4's 1e-6 gate
  are taken on random init. The `deep` suite sweeps `init` as a build
  parameter (H §8.2 A); flipping the default is a 0.3 decision made from D1's
  numbers. The deterministic reduction of the C4 fix is kept for both inits.
- **D10 — One-release compatibility shim.** `retrieve.layers` and
  `retrieve.kernels` become ≈ 40-line modules re-exporting the new names
  under a `DeprecationWarning` (`KMeansTorch = KMeans`, `build_silvertorch`
  over the builder), removed in 0.3. The pyproject `name` becomes
  `torchretrieve`, version `0.2.0`; the import name stays `retrieve`. The
  golden worktree (`/workspace/wt/main-golden`, old harness against the new
  library) keeps working through the shim.
- **D11 — Sequenced before the C4 gate rerun and before D1.** `code_version`
  is the tree hash of `retrieve/src/retrieve` (H §8.2 B): a move changes it
  and invalidates every recorded cell, so the move must land before any
  campaign cell that is meant to survive. Doing WP-1 *and* WP-2 before the
  C4 rerun means C4 gates the final layout once instead of twice, and B3's
  head-to-head measures `ops.triton` against `ops.official` by their final
  names. The alternative (C4 first, then L, then a second C4) costs a second
  14-cell gate day and is the fallback if the user wants the harness loop
  closed first.
- **D12 — The test tree keeps its by-purpose split** (`correctness/`,
  `parity/`, `compile/`) and is re-pointed, not restructured; parity's shared
  reference (`ref_cps_phase23`) moves *into* the library as
  `ops.reference.codesigned_probe_score`, so the gate compares the shipped
  reference, not a test-only copy.

## 3. Target layout

```
retrieve/src/retrieve/
├── __init__.py               modules.* re-exported; LinrBackend, SilverTorchBackend; RetrievalModule, FilterModule
├── interfaces.py             unchanged names + `ops_for(backend)` (imports the namespace on first use)
├── functional.py             masked_topk, counts_to_valid, compact_mask, combine_masks, combine_indices,
│                             post_filter_topk, popcount_int64, clause_subset_match, bloom_subset_match
├── modules/
│   ├── __init__.py           __all__ (§4.1)
│   ├── silvertorch.py        SilverTorch, SilverTorchBuilder, OfficialConfig re-exported   (Algorithm 1; triton | torch | official)
│   ├── linr.py               LiNRV1, LiNRV2, LiNRV3, LiNRV4, LiNRBuilder                    (paper variants; filter as submodule)
│   ├── knn.py                PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, FullScanKNN
│   ├── bit_knn.py            OneBitKNN, SimHashKNN (+ _PackedBitsKNN base)
│   ├── filters.py            BloomFilter, ExactAttributeFilter
│   └── official.py           re-exports silvertorch.modules.* (lazy; OfficialMissing with the install hint)
├── ops/
│   ├── __init__.py           `triton`, `reference`, `official` as lazy submodules; available_backends(); nothing eager
│   ├── triton/               _load.py (imports every kernel file: registers retrieve::*), _host.py (review A5: shared
│   │                         launch / finish / grid), common.py, codesigned_probe_score.py, codesigned_probe_score_exact.py,
│   │                         bloom_match.py, bloom_compact.py, clause_mask.py, clause_compact.py,
│   │                         fused_masked_knn_topk.py, oporp_1bit_match_topk.py
│   ├── reference/            same file names, same signatures, pure torch (the "torch" backend + parity oracle)
│   ├── official/             __init__.py (is_available / ensure_loaded / OfficialMissing / OfficialConfig / `st`),
│   │                         adapter.py (the rest of today's official.py: attrs_to_features … official_probe_score)
│   └── tune.py               the autotune CLI (console script `tune-kernels`), KernelTuneSpec registry
└── indexing/
    ├── __init__.py
    ├── kmeans.py             KMeans (deterministic Lloyd's; init="random"|"kmeans++"), assign()
    ├── ivf.py                padded_layout(), csr_layout()  — the two cluster layouts (triton/torch vs official)
    ├── quantize.py           quantize_int8, quantize_int8_global, quantize_oporp_1bit, project_oporp_1bit_query,
    │                         quantize_simhash_1bit, project_simhash_1bit_query
    └── bloom_hash.py         generate_seeds, generate_clause_salt, build_signatures, build_query_signatures,
                              words_per_cluster, build_transposed_sigs (for G-a)
```

Move table (old → new). Every row is a `git mv` plus import rewrites; no
row changes behaviour.

| today | after WP-1 |
|---|---|
| `layers/silvertorch/main.py` | `modules/silvertorch.py` (`build_silvertorch` → shim; `SilverTorchBuilder` in WP-2) |
| `layers/linr/{postfilter_knn,postfilter_knn_int8,prefilter_knn}.py`, `layers/utils/retrieval.py::FullScanKNN` | `modules/knn.py` |
| `layers/linr/{_bit_knn,one_bit_knn,simhash_knn}.py` | `modules/bit_knn.py` |
| `layers/filters/{bloom,exact_attribute}.py` (the classes) | `modules/filters.py` |
| `layers/filters/__init__.py::{combine_masks,combine_indices}`, `layers/utils/{topk,compact}.py`, `retrieval.py::post_filter_topk`, `exact_attribute.py::clause_subset_match`, `bloom_hash.py::bloom_subset_match`, `quantize.py::popcount_int64` | `functional.py` |
| `layers/utils/kmeans.py` | `indexing/kmeans.py` (class renamed `KMeans`; `KMeansTorch` alias in the shim) |
| `layers/utils/quantize.py` (+ `postfilter_knn_int8._quantize_int8_global`, deduplicated) | `indexing/quantize.py` |
| `layers/filters/bloom_hash.py` (builders, salt, `build_transposed_sigs`) | `indexing/bloom_hash.py` |
| `SilverTorch._build_ivf`'s padded table and CSR view | `indexing/ivf.py` |
| `kernels/common.py`, `kernels/{filters,linr,silvertorch}/*.py` (Triton) | `ops/triton/*.py` (flat; the three subdirectories go) |
| `kernels/silvertorch/official.py` | `ops/official/__init__.py` + `ops/official/adapter.py` |
| eager branches in layers + `tests/parity/conftest.py::ref_cps_phase23` | `ops/reference/*.py` |
| `tune.py` | `ops/tune.py` |
| `interfaces.py` | unchanged path |

## 4. Public API specification

This section is the contract `retrieve/docs/modules.md`,
`docs/system/architecture.md` and the harness's `algos.py` are rewritten from.

### 4.1 `retrieve` and `retrieve.modules`

```python
from retrieve import (
    LinrBackend, SilverTorchBackend, RetrievalModule, FilterModule,   # interfaces
    SilverTorch, SilverTorchBuilder, OfficialConfig,                  # Algorithm 1
    LiNRV1, LiNRV2, LiNRV3, LiNRV4, LiNRBuilder,                      # LiNR paper variants
    PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, FullScanKNN,      # dense / sparse primitives
    OneBitKNN, SimHashKNN,                                            # 1-bit primitives
    BloomFilter, ExactAttributeFilter,                                # filters
)
import retrieve.modules.official   # BloomIndexSearchModule(+Builder), FilterQueryParserModule(+Builder)
```

`retrieve.modules.__all__` is exactly the 20 non-interface names above; the
four LiNR variants, the two builders and `OfficialConfig` (review A9) are the
additions; `build_silvertorch`, `KMeansTorch`, the six `quantize_*` /
`combine_*` / `post_filter_topk` functions leave the top level (they are the
builder, `retrieve.indexing` and `retrieve.functional`). Signatures:

| class | `__init__` | build | `forward` |
|---|---|---|---|
| `SilverTorch` | `(k, n_lists, n_probe, filter_mode="none", m_bits=None, k_hash=None, n_iter=10, seed=0, kmeans_init="random", backend="triton", official=None)` — today's signature plus `kmeans_init` | `register_index(item_embs, item_clause_attrs=None, clause_is_reverse=None)` unchanged; **new** `set_query_params(n_probe=…)` (moved from the harness wrapper, re-running the two `register_index` checks); **new** `build_timings: dict` after registration; `capturable: bool` class attr (`False` on official) | `(query, query_clause_attrs=None, candidate_ids=None) -> (ids, scores)` — unchanged |
| `LiNRV1` | `(k, *, filter: FilterModule \| None = None, backend="triton")` — dense fp16 + post-mask | `register_index(item_embs, item_clause_attrs=None, clause_is_reverse=None)` registers the index and, if a filter is attached, the filter | `(query, query_clause_attrs=None)` |
| `LiNRV2` | `(k, *, filter: FilterModule, backend="triton")` — filter → candidates → `PrefilterKNN` | same | same (`query_clause_attrs` required) |
| `LiNRV3` | `(k, *, candidate_pool=5000, seed=0, filter=None, backend="triton")` — `OneBitKNN(k=candidate_pool)` → `PrefilterKNN(k)`; the cascade body of the harness's `LinrV3` verbatim; `set_query_params(candidate_pool=…)` | same | same |
| `LiNRV4` | `(k, *, filter=None, backend="triton")` — int8 dense + post-mask | same | same |
| `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`, `OneBitKNN`, `SimHashKNN`, `FullScanKNN` | unchanged | unchanged | unchanged |
| `BloomFilter`, `ExactAttributeFilter` | unchanged | unchanged | unchanged |

Builders (D6), modelled on `BloomIndexSearchModuleBuilder`:

```python
ann = (SilverTorchBuilder(k=100, n_lists=1024, n_probe=24, filter_mode="bloom", m_bits=1024, k_hash=5)
       .set_item_embeddings(item_embs)
       .set_item_attributes(item_attrs, clause_is_reverse=None)
       .set_backend("official", official=OfficialConfig(cache_plans=False))
       .set_device(torch.device("cuda"))
       .build())                                  # == construct + .to(device) + register_index

v3 = (LiNRBuilder("v3", k=100, candidate_pool=8000)
      .set_item_embeddings(item_embs)
      .set_filter(ExactAttributeFilter(backend="triton"), item_attrs, clause_is_reverse)
      .build())

same = SilverTorchBuilder(...).set_state_dict(torch.load("st.pt")).build()   # prebuilt: no k-means; load_state_dict + the post-hook
```

`set_item_embeddings` and `set_state_dict` are mutually exclusive; `build()`
without either raises, as Meta's `build()` does for missing data.

### 4.2 `retrieve.ops`

```python
import retrieve.ops.triton          # side-effect: registers torch.ops.retrieve.* (needs triton)
retrieve.ops.triton.codesigned_probe_score(query, flat_probed_items, item_codes, global_scale, k)
retrieve.ops.reference.codesigned_probe_score(...)      # same signature, pure torch, torch.equal to the above
retrieve.ops.official.ensure_loaded(); retrieve.ops.official.st.fused_kmean_ann(...)   # Meta's op, verbatim
retrieve.ops.official.official_probe_score(...)         # B1's adapter, unchanged
retrieve.ops.available_backends()                       # ("triton", "torch") | (..., "official")
```

Op list per namespace (schemas unchanged from today, D3):

| op | `triton` | `reference` | `official` (O §1.1) |
|---|---|---|---|
| `codesigned_probe_score` / `_bloom` / `_exact` | `retrieve::…` (3 `@triton_op`) | yes (today's `ref_cps_phase23` + `_forward_torch_eager`) | `official_probe_score` over `st.fused_kmean_ann*`; exact = `clause_mask` + `pack_mask` + `filtering_bit_mask` |
| `bloom_match`, `bloom_compact` | `retrieve::…` (`bloom_compact` stays an opaque `@custom_op`) | yes | `bloom_full_mask` / `bloom_partial_masks` (S9 ablation) |
| `clause_mask`, `clause_compact` | `retrieve::…` (`clause_compact` opaque) | yes | — |
| `fused_masked_knn_topk` | `retrieve::…` | yes | — |
| `oporp_1bit_match_topk_full` / `_indirect` | `retrieve::…` | yes | — |
| Meta-only: `bloom_index_build`, `parse_expression_query_batch`, `generate_column_info_for_clusters`, the `_multiple` variants | — | — | reachable as `ops.official.st.<name>`; not wrapped |

`ops/triton/_load.py` is the only file that imports the kernel modules;
`retrieve.modules` reaches a namespace through `interfaces.ops_for(backend)`,
which imports it on first use. That keeps `import retrieve` free of kernel
imports (the compile-time cost of registering ten ops moves to first
forward, where it already is for `official`).

### 4.3 `retrieve.indexing` and `retrieve.functional`

| name | signature | consumers |
|---|---|---|
| `indexing.KMeans` | `(n_lists, n_iter=10, seed=0, init="random"\|"kmeans++")`; `.fit(embs) -> (centroids, assignments)`; `.assign(embs, centroids)`; the deterministic `_cluster_sums` kept | `SilverTorch.register_index`; the harness's IVF ablations through `kmeans_init` |
| `indexing.padded_layout` | `(assignments, n_lists) -> (padded_cluster_items, cluster_sizes)` | `SilverTorch` on `triton` / `torch` |
| `indexing.csr_layout` | `(assignments) -> (sort_perm, inv_perm, cluster_offsets, cluster_sizes)` (stable argsort, B2) | `SilverTorch` on `official` |
| `indexing.quantize_int8`, `quantize_int8_global`, `quantize_oporp_1bit`, `project_oporp_1bit_query`, `quantize_simhash_1bit`, `project_simhash_1bit_query` | unchanged | `SilverTorch`, `PostfilterKNNInt8`, `OneBitKNN`, `SimHashKNN` |
| `indexing.generate_seeds`, `generate_clause_salt`, `build_signatures`, `build_query_signatures`, `words_per_cluster`, `build_transposed_sigs` | unchanged | `BloomFilter`, `SilverTorch`; G-a |
| `functional.masked_topk`, `counts_to_valid`, `compact_mask`, `combine_masks`, `combine_indices`, `post_filter_topk`, `popcount_int64`, `clause_subset_match`, `bloom_subset_match` | unchanged | every module; `ops.reference`; `FilterModule` defaults |

"Encoders" — the SASRec / text query encoders — are **not** library code: the
library consumes embedding tensors and has no encoder (grep for
`encode|Encoder|sasrec` under `retrieve/src` is empty). They stay in the
harness ([evaluation-package-layout.md](evaluation-package-layout.md) §5).
The library's only "encoders" are the quantizers, hence `indexing`.

## 5. Backend dispatch after this plan

| module | `"triton"` | `"torch"` | `"official"` |
|---|---|---|---|
| `SilverTorch` | `ops.triton` | `ops.reference` | `ops.official` (eager only, O D7; `capturable = False`) |
| `LiNRV1`–`V4`, the KNN primitives, the filters | `ops.triton` | `ops.reference` | `ValueError` (`check_backend`, already) |
| `PostfilterKNN`, `PostfilterKNNInt8`, `FullScanKNN` | cuBLAS either way; the flag is accepted for symmetry and documented as a no-op | | `ValueError` |

The harness's `PATHS` table (H §3.1) is derived from
`retrieve.interfaces.DISPATCH` by a test, not maintained by hand
([library-harness-boundary.md](library-harness-boundary.md) §4).

## 6. What the library promises the harness (the boundary, in one place)

Implemented in WP-2, consumed by the harness's `algos.py` rewrite (roadmap
C5): `module.k` settable after registration on every algo-level module (the
harness tests this today per layer — `test_algos.py::test_layer_*_k_not_baked` —
and the composites forward it); `set_query_params` on `SilverTorch`
(`n_probe`) and `LiNRV3` (`candidate_pool`), moved from the harness wrappers;
`SilverTorch.build_timings` = `{"kmeans_s", "quantize_s", "assemble_s",
"filter_s"}`; `capturable` on every algo-level module; index size =
`Σ numel·itemsize` over `buffers()` including the filter submodule (already
what `bench.index_bytes` computes); no module calls `torch.compile`
internally and `official` refuses it (already). Full contract:
[library-harness-boundary.md](library-harness-boundary.md).

## 7. Compatibility and packaging

- `pyproject`: `name = "torchretrieve"`, `version = "0.2.0"`, scripts
  `tune-kernels = "retrieve.ops.tune:main"`, extras `official = ["silvertorch"]`
  (unchanged). The root workspace's `[tool.uv.sources]` and
  `dependency-metadata` for `silvertorch` are unchanged.
- `retrieve/layers/__init__.py`, `retrieve/kernels/__init__.py`: re-export
  the old names from the new locations with a `DeprecationWarning`
  (`KMeansTorch = KMeans`, `build_silvertorch` over the builder,
  `retrieve.kernels.silvertorch.official` → `retrieve.ops.official`).
  Removed in 0.3.
- State dicts: buffer names and registration order are unchanged
  (`centroids, item_codes, global_scale, padded_cluster_items | cluster_offsets…, cluster_sizes,
  <filter buffers>`), so a 0.1 checkpoint loads into a 0.2 module and the
  load hook re-derives the scalars as today. The composites add a
  `filter.` prefix for their filter's buffers — new state, not renamed state.
- Op schemas: unchanged (D3). `torch.ops.retrieve.*` names, `KernelTuneSpec`
  keys and the tune JSONs stay valid.
- `code_version` (H §8.2 B) changes once at WP-1 and once at WP-2; D11
  places both before any cell that must survive.

## 8. Docs and tests

- `retrieve/docs/`: `getting-started.md` rewritten around the builders and
  the three-namespace map; `modules.md` becomes the §4.1 table with one
  section per class (V1–V4 gain sections); a new `indexing-and-ops.md` lists
  §4.2 / §4.3 and absorbs the quantization half of
  `filtering-and-quantization.md` (which keeps its filtering half).
- `docs/system/architecture.md`: module map → §3 tree; "Backend dispatch" →
  §5; "Module layout" → the move table. `kernels.md`, `filtering.md`,
  `testing.md`: paths (content is per kernel and unchanged).
- Tests: `tests/correctness/test_linr.py` gains the four composites (each
  `torch.equal` to the hand-composed primitives on the same inputs — the
  wrappers are moved, not rewritten); `test_silvertorch.py` gains the builder
  round-trip (`set_state_dict` == `set_item_embeddings` on the same seed,
  using the existing `TestStateDict` deep-copy pattern) and
  `kmeans_init="kmeans++"` rows; `test_kmeans.py` gains k-means++ seeding
  (valid D² sample, deterministic per seed, inertia ≤ random init on a
  separable blob); `tests/parity/*` import `retrieve.ops.reference` instead
  of `conftest.ref_cps_phase23`; `test_tune_smoke.py` follows `ops.tune`;
  `test_official.py` follows `ops.official`. A new `tests/test_public_api.py`
  asserts `retrieve.__all__` and `retrieve.modules.__all__` equal §4.1's list
  and that `import retrieve` does not import the kernel modules
  (`sys.modules` check) — CPU-only, runs in the collect phase of every gate.

## 9. Interaction with the other plans

| plan says | read as, after WP-1 |
|---|---|
| **O** §5.1 `kernels/silvertorch/official.py`, `main.py: _forward_official` | `ops/official/{__init__,adapter}.py`; `modules/silvertorch.py`; CSR buffers from `indexing.csr_layout` |
| **O** §5.2 `tests/parity/test_official.py`, `require_official()` | unchanged paths; imports re-pointed |
| **O** §8 TF-1 (transposed bloom in Triton, roadmap G-a) and review A5 (`_host.py`) | `ops/triton/bloom_transposed.py` + a `HAS_MASK` path in `ops/triton/codesigned_probe_score.py`; `ops/triton/_host.py` is written in WP-1 as the move's one refactor of the launch scaffold (review A5 was deferred to G-a "to avoid two perf gates" — the perf gate it feared is B3, which now runs after this plan) |
| **O** §9 head-to-head scripts (`bench_common.py` callables) | `retrieve.ops.triton.*` vs `retrieve.ops.official.*` by name |
| **H** §3.1 `algos.py` (five wrappers + tables, 362 lines today) | ≈ 80 lines: `ALGOS` name → class table, `build(job, inputs)`, `PATHS` derived by test — roadmap C5 |
| **H** §7 / §8.2 A library nits (`set_query_params`, `build_timings`) | WP-2 (`set_query_params` moves *from* the harness; `build_timings` is new) |
| review §D "fold into G-a": A5, D3, `bloom_sigs_t` wording | A5 and D3 in WP-1; the wording in G-a |
| review A9 (`__all__` rows), A8 (`.long()`), T3 (helper fold), B.4 test-support move | WP-1 |
| [torch-export-refactor.md](torch-export-refactor.md), [live-update-api.md](live-update-api.md) (parked, G-e) | re-scope against this layout; the composites' single forward signature is what the export plan wanted; live update lands on `retrieve.modules` |

## 10. Work packages

All on the box. CPU gates: `ruff check retrieve && ruff format --check
retrieve`, `uv run --directory retrieve pytest tests/test_public_api.py -q`,
`python3 scripts/check_doc_links.py` at zero, `cd evaluation && uv run pytest
retrieval/tests/` green (the harness still imports the old names through the
shim in WP-1). GPU gates: the full library suite (≈ minutes, 519 today), plus
the bit-exactness checks named per WP. The GPU is shared: one job at a time.

- **WP-1 — the move, no behaviour change (CPU 1.5 d + GPU suite).** §3 move
  table, `ops/triton/_load.py`, `ops/triton/_host.py` (A5), `interfaces.ops_for`,
  `ops/reference/` (extracted from the layers' eager branches and
  `tests/parity/conftest.py`), `functional.py`, `indexing/` (incl. `ivf.py`
  extracted from `_build_ivf`), the shim (§7), the pyproject rename, review
  A8 / A9 / T3 / B.4 leftovers, test imports re-pointed, docs paths re-pointed
  (§8). **No behaviour change**: the diff shows moves, import rewrites, and
  the extraction of already-existing eager code. Gate (CPU): the four checks;
  `git diff --stat -M` reviewed against the move table. Gate (GPU): full
  suite green; every parity file bit-exact; `ops.reference.codesigned_probe_score`
  `torch.equal` to the pre-move `ref_cps_phase23` on `make_probe_family`
  inputs (one-off script kept under `library-api-refactor-artifacts/`); a
  pre-move `SilverTorch` state dict (all three backends × three filter modes)
  loads into the moved module and returns identical ids and scores; the shim
  imports with a warning; the old-harness golden worktree still runs one
  cell. Branch `dev/l1-library-layout` off `development`.
- **WP-2 — composites, builders, k-means++, build timings, boundary clauses
  (CPU 2 d + GPU suite).** `modules/linr.py` (V1–V4 moved from
  `evaluation/retrieval/algos.py`), `SilverTorchBuilder`, `LiNRBuilder`,
  `build_silvertorch` retired into the shim, `KMeans(init="kmeans++")`,
  `set_query_params` moved into the library, `build_timings`, `capturable`
  class attributes, `retrieve.modules.official`, `interfaces.DISPATCH`; docs
  §8; tests §8; `retrieve/tests/correctness/test_boundary.py` (**X** §5).
  Gate (CPU): the four checks; (GPU): `test_linr.py` composites `torch.equal`
  to the hand-composed primitives on every backend × filter kind; builder
  round-trip; k-means++ tests; `test_boundary.py`; full suite green.
  **Unblocks roadmap C5** (the harness table needs the composites) and,
  with C5, the C4 gate rerun on the final layout (D11).

WP-1 → WP-2 is 3.5 CPU days plus two library-suite runs. Nothing here is
citable (rule 2); the plan's only numbers are line counts.

## 11. Risks

- **The move collides with in-flight edits.** WP-1 touches every library
  file; nothing else may edit `retrieve/` between WP-1's branch point and its
  merge. The 2026-09-06 status block says no branch carries unmerged code,
  which is the condition.
- **Extracting `ops/reference` changes a number.** It must not: the extracted
  eager code is the same code. WP-1's `torch.equal` against the pre-move
  reference is the check; if it fails, the extraction is wrong, not the
  tolerance.
- **`code_version` invalidation.** Any cell recorded before WP-2 merges is
  re-run by `--resume`. Today no v2 cell has been recorded on `development`
  (the C4 probe results live under the artifacts directory), so the cost is
  zero if D11's order holds.
- **Builders duplicate `register_index`.** They call it; there is one build
  path underneath. If a builder grows logic `register_index` lacks, that is a
  bug.
- **k-means++ on 3 M × 128** is `n_lists` D²-sampling passes; at 1,024 lists
  that is ≈ 1.5 TB of fp32 reads on an A100, minutes. Implement the greedy
  k-means‖ variant (`oversampling = 2·log(n_lists)` candidates per round) if
  the plain version exceeds a minute at `n_lists = 8192`;
  `build_timings["kmeans_s"]` reports it either way, and D9 keeps it out of
  every gated number until D1.
- **Two gates, not one, if D11 is overruled.** If the C4 rerun goes first,
  C4 must be rerun after WP-2 (14 cells, one GPU day); the plan still works.

## 12. Validation record

*(appended by WP-1 and WP-2 when they run; model:
[archive/cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).)*
