# Chapter 4. Реализация (Implementation) — Reference Notes

> **Working document.** English-language reference notes for the Russian-prose pass of Chapter 4 of the HSE MSc thesis. **This chapter delivers goal 1 of the thesis.** It documents the implementation of the open-source `torchretrieve` package (PyPI distribution name) as a reusable PyTorch retrieval-layer library: drop-in `nn.Module` retrieval layers, Triton-optimized kernels under the hood, per-layer benchmarks shipped alongside the code — not a reproduction artifact for a single paper. Three properties carry the framework claim and must be made explicit in the prose, each with concrete evidence inside the chapter: (a) `torch.compile` interop (no graph breaks at module boundaries; `cudagraph_trees`-safe with `mode="reduce-overhead"`; evidence in §4.5); (b) an **encoder-agnostic public API** — the layers consume already-computed embedding tensors and are agnostic to how those embeddings were produced; the evaluation demonstrates this by feeding the same layers from a SASRec / gSASRec sequence encoder on Goodreads + Yambda and from a pretrained Nomic-Embed text encoder on arXiv (evidence in §4.11); (c) **extension points** for downstream users — the `FilterModule` ABC and `Backend` literal in [`retrieve/src/retrieve/interfaces.py`](../../retrieve/src/retrieve/interfaces.py), swappable kernels under `kernels/`, and the offline kernel-autotuning CLI in [`retrieve/src/retrieve/tune.py`](../../retrieve/src/retrieve/tune.py) (evidence in §4.4, §4.6, §4.7). The package bundles two reference retriever families: paper-faithful Triton implementations of the LinR algorithm family, and a composite IVF + INT8 + Bloom retriever assembled from classical primitives — present in the package as a **second bundled retriever** that demonstrates the framework's contracts extend naturally beyond the LinR line. Reproducibility of the LinR paper is acknowledged as a quality bar, not the chapter's headline contribution. Citation Policy from [00-thesis-plan.md](00-thesis-plan.md) is in effect: no Meta-affiliated work is cited. The in-repo symbol `silvertorch` (directory name, type identifier, kernel name) is retained as a legacy code-internal symbol; body prose always says "co-designed IVF + INT8 + Bloom retriever" or "co-designed retriever". Formal algorithm definitions are the responsibility of Chapter 2 (Methods); this chapter is implementation-only and cross-references Ch.2 where appropriate.

---

## §4.0 Фреймворк как библиотека (The framework as a library)

This section frames the rest of the chapter. It states *what* `torchretrieve` is as a product before §4.1 onward describes *how* it is implemented. The rest of the chapter is implementation detail; this section is the product surface.

### Public surface area

A user installs `torchretrieve` and writes `from retrieve import ...`. The single import surface is the package's top-level [`__init__.py`](../../retrieve/src/retrieve/__init__.py), which re-exports exactly 17 symbols (the `__all__` list audited in §4.2). The exports group into three categories:

- **Retrieval layers** (`nn.Module` subclasses): `FullScanKNN`, `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`, `OneBitKNN`, `SimHashKNN`, `SilverTorch`. These are the drop-in layers a user composes into their model.
- **Filter primitives** (`FilterModule` subclasses): `ExactAttributeFilter`, `BloomFilter`, plus the composition helpers `combine_masks` and `combine_indices`.
- **Quantization and assembly helpers**: `KMeansTorch`, `build_silvertorch`, `post_filter_topk`, `quantize_int8`, `quantize_oporp_1bit`, `quantize_simhash_1bit`, plus the two contract types `Backend` and `FilterModule`.

A user never has to read the kernel layer to use the library; the kernels under `retrieve/kernels/` are an implementation detail behind the layer modules.

### Input / output contract

The retrieval layers are encoder-agnostic by construction: they take `queries: Tensor` (shape `[B, D]`) and an index built from `items: Tensor` (shape `[N, D]`), and they return top-K indices alongside top-K scores. The layers do not produce embeddings, do not assume how embeddings were produced, and do not depend on any encoder architecture. The empirical demonstration of this — the same layers consumed by a SASRec/gSASRec sequence encoder on Goodreads + Yambda and by a pretrained Nomic-Embed text encoder on arXiv — is documented in §4.11.

Filter modules ([`retrieve/src/retrieve/interfaces.py`](../../retrieve/src/retrieve/interfaces.py)) take a query's clause attributes and return a boolean predicate over the item index in one of three native shapes (`evaluate_mask`, `evaluate_indices`, `evaluate_subset`) — see §4.7 for the full contract.

### Extension points exposed to users

The library exposes four extension points that downstream users (not just this thesis) can hook into:

- **`Backend` literal** — `Literal["torch", "triton"]` in [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py). Every KNN module accepts a `backend=` argument; the choice routes to either a pure-PyTorch reference path or a Triton-kernel fast path. The two paths are kept at bit-level / float-tolerance parity (§4.4). New backends (e.g., a CUTLASS path) can be added by extending the literal and the dispatch site.
- **`FilterModule` ABC** — abstract base class in [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py) with one mandatory method (`register_index`) and one mandatory predicate method (`evaluate_mask`); two further predicate shapes (`evaluate_indices`, `evaluate_subset`) ship with default implementations and may be overridden by faster fused versions. Users add new filter primitives (e.g., metadata-based, geographic, time-window) by subclassing this ABC. The KNN modules consume `FilterModule` instances through duck-typed composition, never through hard-coded filter logic — see §4.7.
- **Swappable kernels** — every Triton kernel is registered as a `@torch.library.triton_op` with a documented signature (§4.5). A user can substitute their own kernel implementation for any operator without touching the layer above it.
- **Offline kernel autotuning** — the `tune-kernels` console script in [`retrieve/src/retrieve/tune.py`](../../retrieve/src/retrieve/tune.py) lets users re-tune the shipped Triton kernels to their own GPU architecture and workload shapes (`uv run tune-kernels <subcommand>`); the printed `DEFAULT_CONFIG` line is pasted into the kernel file. The shipped per-kernel defaults are tuned for SM80 (A100); users on other architectures retune locally. Full mechanism in §4.6.

### Product shape

The package is shaped as a small, focused library of drop-in PyTorch retrieval layers with Triton-optimized kernels under the hood and benchmarks shipped alongside the code. No algorithmic novelty is claimed for any of the bundled retrievers — the LinR family is paper-faithful and the composite IVF + INT8 + Bloom retriever is assembled from classical primitives (k-means inverted file, per-row INT8 with `torch._int_mm`, BitFunnel-style Bloom signatures). What the framework provides is the integration surface: encoder-agnostic layers that `torch.compile` cleanly, expose stable extension points, and ship with per-layer micro-benchmarks (§4.10) and a full cross-dataset evaluation (Ch.6) so a downstream user can pick the layer that matches their recall / latency / memory budget without having to re-derive any numbers themselves.

---

## §4.1 Архитектурный обзор (Architecture overview)

### Workspace layout

- The repository root is a `uv` workspace with two members. Source: [pyproject.toml:14-15](../../pyproject.toml#L14-L15).
  ```toml
  [tool.uv.workspace]
  members = ["retrieve", "evaluation"]
  ```
- Member 1 — `retrieve/`: the published PyPI distribution. Source: [retrieve/pyproject.toml:1-14](../../retrieve/pyproject.toml#L1-L14).
  - Project name: `retrieve` (version `0.1.0`).
  - Distribution name on PyPI: `torchretrieve` (per README line 1 and the standalone README banner; the source-level Python package name is `retrieve`).
  - Build backend: `hatchling`; package root `src/retrieve` (line 21).
  - Declared dependencies: `click>=8.1`, `torch>=2.4,<3`, `triton>=3.0` (lines 8–10).
  - One console script entry point: `tune-kernels = "retrieve.tune:main"` (lines 13–14).
- Member 2 — `evaluation/`: the benchmark and training harness. Not published to PyPI. Imports from `retrieve` for the layers it benchmarks. (Cross-referenced from Ch.3, Ch.5, Ch.6 — not the subject of this chapter; mentioned only for completeness.)
- The workspace-level meta-package [pyproject.toml:1-19](../../pyproject.toml#L1-L19) is `retrieve-workspace`, an unpublished meta-package whose only purpose is to make both members share one `.venv` under `uv`.

### Three-tier separation inside the package

The package source tree under [retrieve/src/retrieve/](../../retrieve/src/retrieve/) has three clearly separated tiers, each with a distinct role:

```
src/retrieve/
├── __init__.py              # public re-exports (the surface)
├── interfaces.py            # Backend type alias + FilterModule ABC (the contracts)
├── tune.py                  # offline autotuner CLI (the build-time tool)
├── layers/                  # nn.Module classes (the user-facing API)
│   ├── __init__.py
│   ├── linr/                #   LinR family — V1, V2, V3, V1-INT8
│   │   ├── postfilter_knn.py
│   │   ├── postfilter_knn_int8.py
│   │   ├── prefilter_knn.py
│   │   └── one_bit_knn.py
│   ├── silvertorch/         #   Co-designed IVF + INT8 + Bloom retriever
│   │   └── main.py
│   ├── filters/             #   Standalone FilterModule implementations
│   │   ├── bloom.py
│   │   └── exact_attribute.py
│   └── utils/               #   Helpers: FullScanKNN, KMeans, quantize, compact
│       ├── retrieval.py
│       ├── kmeans.py
│       ├── quantize.py
│       └── compact.py
└── kernels/                 # Triton kernels + host wrappers + DEFAULT_CONFIGs
    ├── linr/
    │   ├── fused_masked_knn_topk.py
    │   └── oporp_1bit_match_topk.py
    ├── silvertorch/
    │   ├── codesigned_probe_score.py
    │   ├── codesigned_probe_score_exact.py
    │   └── bloom_match.py
    └── filters/
        ├── clause_mask.py
        ├── clause_compact.py
        └── bloom_compact.py
```

**Tier 1 — `interfaces.py` (contracts).** Holds the two cross-tier abstractions: `Backend = Literal["torch", "triton"]` (line 8) and `class FilterModule(nn.Module, abc.ABC)` (lines 11–57). These are the only types that propagate across both tiers below; everything else is owned by either `layers/` or `kernels/`. Source: [retrieve/src/retrieve/interfaces.py:1-58](../../retrieve/src/retrieve/interfaces.py#L1-L58).

**Tier 2 — `layers/` (nn.Module classes, the user-facing API).** Each retrieval algorithm is one `nn.Module` subclass. The module's contract is uniform across all classes:
- `__init__(self, k: int, …, backend: Backend = "triton") -> None` — the constructor accepts the top-K size and (where applicable) a backend selector.
- `register_index(self, item_embs: Tensor, …) -> None` — registers the immutable item index. Saves device buffers via `register_buffer`. Called once.
- `forward(self, query: Tensor, …) -> tuple[Tensor, Tensor]` — returns `(topk_ids[B, K], topk_scores[B, K])`. May accept optional `mask`, `candidate_ids`, or `query_clause_attrs` depending on the module.
The `layers/` tier never contains `@triton.jit` definitions; instead it imports kernel host wrappers from `kernels/` and dispatches on `self.backend`.

**Tier 3 — `kernels/` (Triton kernels + Python wrappers + offline tuning configs).** Each kernel file follows a uniform pattern:
- `@dataclass(frozen=True) class <Kernel>Config` — holds tunable hyperparameters (block size, `num_warps`, `num_stages`).
- `DEFAULT_CONFIG = <Kernel>Config(...)` — single offline-tuned constant chosen by [tune.py](../../retrieve/src/retrieve/tune.py).
- `@triton.jit def _<kernel>_kernel(...)` — the actual device kernel.
- `def _<kernel>_impl(..., *, config: <Config> | None = None)` — eager host wrapper (used by tune scripts and parity tests; accepts a config override).
- Public wrapper decorated by `@torch.library.triton_op` (uniform across all 9 public wrappers) — visible to `torch.compile`. Body duplicates the `_impl` launch logic but with a fixed schema and hard-coded `DEFAULT_CONFIG`. The duplication is structural: torch.export's kernel registry requires that `wrap_triton(_kernel)[grid](...)` is textually present in the decorated function's source.

### Per-tier design rationale

- The split between `layers/` and `kernels/` is a layering invariant, not a packaging convenience: a downstream user can import `BloomFilter` (a `layers/` class) without ever touching Triton, and a kernel developer can edit `kernels/silvertorch/codesigned_probe_score.py` without touching the user-facing `SilverTorch` class.
- The `interfaces.py` file is tiny by design (57 LOC). Its job is to define the two types that need to be visible from both tiers: `Backend` (used by every `layers/` constructor and every dispatch site) and `FilterModule` (the contract every filter implementation honours). Because both types are stable, the file has the lowest churn in the package.
- `tune.py` is the only `*.py` file at the package root besides `__init__.py` and `interfaces.py`. It is not part of the user-facing API in the sense that user code does not call into it from inside a retrieval pipeline; it is invoked from the command line via the `tune-kernels` console script. The module is large (712 LOC) because it sweeps multi-parameter grids for every tunable kernel and pretty-prints results — see §4.6.

### Writer's notes

- Mention `evaluation/` once for completeness (single sentence) but do not catalogue it — that is Ch.3 and Ch.5.
- The directory tree above is a candidate for a TikZ figure in the chapter (one of the mandatory visual deliverables per the contract). Suggested rendering: a three-column tree with `interfaces.py` as a thin connector between `layers/` (left) and `kernels/` (right) — see "Visual deliverables" at the end of this file.
- [TODO: clarify with author — is `tune.py` considered part of the "public API" for the purpose of §4.2? Operationally it is invoked as a CLI, not imported. The recommended position is to introduce it briefly in §4.1 as "build-time tool" and document it in §4.6.]

---

## §4.2 Публичный API (Public API)

### `__all__` listing — 17 symbols

Source: [retrieve/src/retrieve/__init__.py:20-38](../../retrieve/src/retrieve/__init__.py#L20-L38).

```python
__all__ = [
    "Backend",
    "BloomFilter",
    "ExactAttributeFilter",
    "FilterModule",
    "FullScanKNN",
    "KMeansTorch",
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
    "SilverTorch",
    "build_silvertorch",
    "combine_indices",
    "combine_masks",
    "post_filter_topk",
    "quantize_int8",
    "quantize_oporp_1bit",
]
```

The seventeen exports break down as: 6 KNN modules, 2 filter classes, 1 IVF factory helper, 3 composition / post-processing helpers, 2 quantization functions, 1 K-Means utility, 1 type alias, 1 abstract base class.

Note: `quantize_int8_global` is **not** re-exported through `retrieve/__init__.py` — it is used internally by `SilverTorch` (for the per-tensor item-side scale) and is accessible only via `from retrieve.layers.utils.quantize import quantize_int8_global`. Writer should flag this as a possible API-surface inconsistency: per-row `quantize_int8` is public; the per-tensor variant is package-internal.

### Public API map

| Symbol | Kind | File | One-line role |
|---|---|---|---|
| `Backend` | type alias | [interfaces.py:8](../../retrieve/src/retrieve/interfaces.py#L8) | `Literal["torch", "triton"]` — backend selector |
| `FilterModule` | abstract base | [interfaces.py:11-57](../../retrieve/src/retrieve/interfaces.py#L11-L57) | Contract for boolean predicates over an item index |
| `FullScanKNN` | nn.Module | [layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | Reference exhaustive `query @ item^T` + `torch.topk` |
| `PostfilterKNN` | nn.Module | [layers/linr/postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) | LinR V1: dense fp16 score + optional mask + topk |
| `PostfilterKNNInt8` | nn.Module | [layers/linr/postfilter_knn_int8.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) | LinR V1-INT8: `torch._int_mm` dense score + mask + topk |
| `PrefilterKNN` | nn.Module | [layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py) | LinR V2: sparse gather + dot + topk over a candidate set |
| `OneBitKNN` | nn.Module | [layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) | LinR V3: Sign-OPORP 1-bit + Hamming popcount + topk |
| `SilverTorch` | nn.Module | [layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py) | Co-designed IVF + INT8 + optional fused Bloom/exact filter |
| `build_silvertorch` | factory | [layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py) | Auto-configures `n_lists`, `n_probe`, `m_bits`, `k_hash` for a catalogue |
| `BloomFilter` | nn.Module / FilterModule | [layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | Conjunctive Bloom-signature subset test (paper-strict, no NOT) |
| `ExactAttributeFilter` | nn.Module / FilterModule | [layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | Exact AND-of-OR clause matching with reverse flag |
| `KMeansTorch` | nn.Module | [layers/utils/kmeans.py](../../retrieve/src/retrieve/layers/utils/kmeans.py) | Mini-batch K-Means for IVF centroid construction |
| `quantize_int8` | function | [layers/utils/quantize.py:25-33](../../retrieve/src/retrieve/layers/utils/quantize.py#L25-L33) | Per-row symmetric INT8 quantization (returns codes, scales) |
| `quantize_oporp_1bit` | function | [layers/utils/quantize.py:81-108](../../retrieve/src/retrieve/layers/utils/quantize.py#L81-L108) | Sign-OPORP 1-bit (returns bits, signs, perm) |
| `combine_masks` | function | [layers/filters/__init__.py:13-23](../../retrieve/src/retrieve/layers/filters/__init__.py#L13-L23) | Element-wise AND of N optional `[B, N]` masks |
| `combine_indices` | function | [layers/filters/__init__.py:26-70](../../retrieve/src/retrieve/layers/filters/__init__.py#L26-L70) | Sparse cascade over a sequence of `FilterModule`s |
| `post_filter_topk` | function | [layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | Apply a boolean post-mask to pre-computed top-K ids |

### Class signatures — KNN modules

`FullScanKNN`. Source: [layers/utils/retrieval.py:22-59](../../retrieve/src/retrieve/layers/utils/retrieval.py#L22-L59).
- `__init__(self, k: int) -> None` — no backend selector. Always runs `torch.matmul` + `torch.topk`.
- `forward(self, query: Tensor, mask: Tensor | None = None, candidate_ids: Tensor | None = None) -> tuple[Tensor, Tensor]` — accepts optional dense mask or sparse candidate-ids; returns `(topk_ids[B, K], topk_scores[B, K])`.

`PostfilterKNN`. Source: [layers/linr/postfilter_knn.py:9-60](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L9-L60).
- `__init__(self, k: int, backend: Backend = "triton") -> None` — backend flag accepted for API symmetry. Both paths run identical torch code.
- `register_index(self, item_embs: Tensor) -> None` — stores `item_embs` as a buffer (fp16-cast inside `forward`).
- `forward(self, query: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]` — pure-torch `query @ item_embs.T`, then `masked_fill(-inf)` on `mask`, then `torch.topk`. fp16 storage, fp32 accumulation. The previous Triton `fused_matmul_topk` kernel was removed because cuBLAS + CUB are already optimal — see [docs/system/kernels.md:151-162](../system/kernels.md#L151-L162).

`PostfilterKNNInt8`. Source: [layers/linr/postfilter_knn_int8.py:28-123](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L28-L123).
- `__init__(self, k: int, backend: Backend = "triton") -> None`.
- `register_index(self, item_embs: Tensor) -> None` — quantizes via `quantize_int8_global`, stores `[D, N_padded] int8` buffer and a single `global_scale` Python float.
- `forward(self, query: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]` — `torch._int_mm(query_int8, item_int8)` returns int32; dequantized by `query_scale * global_scale`; mask applied; `torch.topk`. For `B < 17` the query is zero-padded before the matmul (LtGemm requires a minimum M) and sliced after.

`PrefilterKNN`. Source: [layers/linr/prefilter_knn.py:10-120](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L10-L120).
- `__init__(self, k: int, backend: Backend = "triton") -> None`.
- `register_index(self, item_embs: Tensor) -> None` — stores `item_embs` as a buffer.
- `forward(self, query: Tensor, candidate_ids: Tensor | None = None, counts: Tensor | None = None) -> tuple[Tensor, Tensor]` — Triton path delegates to `fused_masked_knn_topk(query, item_embs, candidate_ids, counts, k)`; torch path does manual `bmm(query, item_embs[candidate_ids].transpose(-1, -2))` with masking. Without `candidate_ids` both paths fall back to dense.

`OneBitKNN`. Source: [layers/linr/one_bit_knn.py:34-167](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L34-L167).
- `__init__(self, k: int, seed: int = 0, backend: Backend = "triton") -> None` — `seed` is the deterministic OPORP seed.
- `register_index(self, item_embs: Tensor) -> None` — calls `quantize_oporp_1bit(item_embs, seed=self.seed)` and stores `item_bits[N, W] int64`, `signs[D] int8`, `perm[D] int64` as buffers.
- `forward(self, query: Tensor, candidate_ids: Tensor | None = None, counts: Tensor | None = None) -> tuple[Tensor, Tensor]` — Triton path calls `oporp_1bit_match_topk_full` (full corpus) or `oporp_1bit_match_topk_indirect` (sparse) depending on whether `candidate_ids` is None.

`SilverTorch`. Source: [layers/silvertorch/main.py:27-240](../../retrieve/src/retrieve/layers/silvertorch/main.py#L27-L240).
- `__init__(self, k: int, n_lists: int, n_probe: int, filter: Literal["none", "bloom", "exact"] = "none", m_bits: int | None = None, k_hash: int | None = None, n_iter: int = 10, seed: int = 0, backend: Backend = "triton") -> None` — the only KNN module with inline filter support. `n_lists` is the number of IVF centroids; `n_probe` is how many to scan per query; `m_bits` / `k_hash` are required when `filter == "bloom"`.
- `register_index(self, item_embs: Tensor, item_clause_attrs: Tensor | None = None, clause_is_reverse: Tensor | None = None) -> None` — runs K-Means via `KMeansTorch`, quantizes items via `quantize_int8_global`, builds Bloom signatures or stores clause attributes depending on `filter`.
- `forward(self, query: Tensor, query_clause_attrs: Tensor | None = None, candidate_ids: Tensor | None = None) -> tuple[Tensor, Tensor]` — Phase 1 (centroid top-`n_probe`) runs host-side as a small matmul; Phase 2+3 (gather candidate items from probed lists + filter + INT8 score) fused into one kernel on the Triton path, or staged with explicit `[B, P, D]` materialization on the torch path.

### Class signatures — Filter modules

`BloomFilter`. Source: [layers/filters/bloom.py:12-106](../../retrieve/src/retrieve/layers/filters/bloom.py#L12-L106).
- `__init__(self, m_bits: int, k_hash: int, backend: Backend = "triton") -> None` — `m_bits` must be a positive power of two and a multiple of 64; `k_hash` must be positive. Validation raises `ValueError` at construction.
- `register_index(self, item_clause_attrs: Tensor, item_embs: Tensor | None = None, clause_is_reverse: Tensor | None = None) -> None` — rejects any reverse clause (paper-strict, conjunctive only). Builds `[N, W] int64` signature tensor where `W = m_bits // 64`.
- Implements `FilterModule`: `evaluate_mask`, `evaluate_indices`, `evaluate_subset` — see §4.7 for the full bodies.

`ExactAttributeFilter`. Source: [layers/filters/exact_attribute.py:12-108](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L12-L108).
- `__init__(self, backend: Backend = "triton") -> None`.
- `register_index(self, item_clause_attrs: Tensor, clause_is_reverse: Tensor | None = None, item_embs: Tensor | None = None) -> None` — stores `[N, C, A_max] int64` attribute tensor and `[C] bool` reverse-flag tensor.
- Implements `FilterModule` — see §4.7 for the full bodies.

### Helper functions (one-line each)

- `quantize_int8(embs) -> (codes[N, D] int8, scales[N] fp32)` — per-row symmetric INT8 with per-row scales. [quantize.py:25-33](../../retrieve/src/retrieve/layers/utils/quantize.py#L25-L33).
- `quantize_oporp_1bit(embs, seed=0) -> (bits[N, W] int64, signs[D] int8, perm[D] int64)` — Sign-OPORP 1-bit projection, seed-deterministic. [quantize.py:81-108](../../retrieve/src/retrieve/layers/utils/quantize.py#L81-L108).
- `combine_masks(*masks) -> mask | None` — element-wise AND of dense masks. [filters/__init__.py:13-23](../../retrieve/src/retrieve/layers/filters/__init__.py#L13-L23).
- `combine_indices(filters, queries) -> (ids, counts)` — sparse cascade across filters. [filters/__init__.py:26-70](../../retrieve/src/retrieve/layers/filters/__init__.py#L26-L70).
- `post_filter_topk(topk_ids, post_mask) -> (ids_masked, counts)` — apply boolean post-mask to top-K ids.
- `build_silvertorch(item_embs, k, …) -> SilverTorch` — factory that picks default `n_lists`, `n_probe`, `m_bits`, `k_hash` from catalogue size.
- `KMeansTorch.fit(item_embs) -> (centroids[n_lists, D], assignments[N])` — k-means++ initialization, fixed-iteration EM.

### Minimal end-to-end example (verbatim from README)

Source: [retrieve/README.md:19-32](../../retrieve/README.md#L19-L32).

```python
import torch
from retrieve import SilverTorch

# 1M items, dim 128 — INT8 IVF + INT8 ANN
N, D, k = 1_000_000, 128, 10
item_embs = torch.randn(N, D, device="cuda")

ann = SilverTorch(k=k, n_lists=1024, n_probe=16).cuda()
ann.register_index(item_embs)

queries = torch.randn(4, D, device="cuda")
topk_ids, topk_scores = ann(queries)        # ([4, 10], [4, 10])
```

For attribute-filtered retrieval, the user constructs `SilverTorch(..., filter="bloom", m_bits=1024, k_hash=4)` and passes `query_clause_attrs=[B, C]` to `forward`. The README's "What's in the box" table at lines 36–48 mirrors the symbol map above and is a useful prose source for the writer.

### Writer's notes

- The README quick-example shows the `filter="none"` default. The thesis chapter should also include a filtered example (5–6 lines): construct `SilverTorch(..., filter="bloom", m_bits=1024, k_hash=4)`, then call `ann(queries, query_clause_attrs=q_attrs)`. Source for the syntax: [layers/silvertorch/main.py:222-240](../../retrieve/src/retrieve/layers/silvertorch/main.py#L222-L240).
- The contract (Ch.4 §4.2) calls for "Table: 6 KNN modules (FullScan, OneBitKNN, Postfilter, PostfilterInt8, Prefilter, IVF+INT8+Bloom — i.e., SilverTorch class) + 2 filters + helpers, with one-line descriptions." — the public-API table above is ready to lift; convert column headers to Russian.
- [TODO: clarify with author — does the chapter want the README example reproduced verbatim, or paraphrased into Russian with the code in English? Standard HSE practice is to keep the code in English and explain in Russian prose.]

---

## §4.3 Triton-ядра (Triton kernels)

The package contains 10 `@triton.jit` functions across 8 files: **8 main device kernels** (one or two per file) plus **2 inline device helpers** (`_popcount_int64` at [kernels/linr/oporp_1bit_match_topk.py:42](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L42) and `_or_combine` at [kernels/silvertorch/codesigned_probe_score.py:14](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L14)). The helpers are static-inlined into their parent kernels and do not have host-side entry points; they are not retrieval primitives. The eight main kernels are grouped by purpose into three subsubsections: LinR family (§4.3.1), co-designed retriever (§4.3.2), filter primitives (§4.3.3).

For each kernel the notes give: (i) file with line numbers for the Python launcher and the device kernel; (ii) Python entry signature; (iii) device-kernel signature with `tl.constexpr` parameters; (iv) launch-grid formula; (v) `DEFAULT_CONFIG` values; (vi) memory access pattern (one sentence); (vii) Triton primitives used; (viii) a verbatim header / design comment.

### §4.3.1 LinR kernels

#### `fused_masked_knn_topk` — LinR V2 sparse scoring

File: [retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py). Device kernel at lines 42–96. Python entry at lines 218–284.

Python entry signature:
```python
def fused_masked_knn_topk(
    query: Tensor,            # [B, D] fp16
    item_embs: Tensor,        # [N, D] fp16
    positive_indices: Tensor, # [B, P] int64 (candidate ids)
    counts: Tensor,           # [B] int64 (valid count per row)
    k: int,
) -> tuple[Tensor, Tensor]:   # (topk_ids[B, K] int64, topk_scores[B, K] fp32)
```

Device-kernel parameters (`@triton.jit`):
- `query_ptr`, `item_embs_ptr`, `pos_indices_ptr`, `counts_ptr`, `out_scores_ptr`
- `P: tl.constexpr` (bucketed width — see below), `D: tl.constexpr`
- strides for query, items, positive_indices, scores
- `BLOCK_N: tl.constexpr`

`DEFAULT_CONFIG` at line 39:
```python
DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n=32, num_warps=8)
# num_stages defaults to 3 in the dataclass
```
Tuned on A100; `block_n=32` wins in 4 of 5 catalogue buckets; `num_warps=8` everywhere. Tuning rationale at file lines 34–38.

Launch grid (line 250):
```python
grid = (triton.cdiv(p, cfg.block_n), b)
```

What it does: indirect gather + dot-product + score buffer. For each `(batch_b, tile_p)` program, the kernel loads the single query row `query[b, :D]` (in registers), gathers `BLOCK_N` candidate ids from `positive_indices[b, p:p+BLOCK_N]`, gathers the corresponding item rows `[BLOCK_N, D]`, computes elementwise products + row-sum to get `BLOCK_N` dot products, masks lanes whose `p >= counts[b]` to `-inf`, and stores `BLOCK_N` scores. The host then calls `torch.topk(all_scores, k, dim=1)`.

Memory access pattern: one cached `[D]` query load per program; `[BLOCK_N, D]` indirect gather of item rows; sequential `[BLOCK_N]` store of scores.

Triton primitives: `tl.load(..., mask=..., other=0)` (masked indirect load), `tl.sum(..., axis=1)` (row-wise reduction), `tl.where(in_count, dots, float("-inf"))`, `tl.store(..., mask=p_valid)`. No `tl.dot` — per-cell scoring is elementwise because the gathered rows differ per `(B, p)` cell so a true GEMM would re-load rows per query column (file lines 108–126).

Design comment (verbatim, lines ~108–126):
```
Per-cell scoring is elementwise (tl.sum) rather than tl.dot: the gathered
rows differ per (B, p) cell so a true GEMM would re-load rows per query
column. For the dense (high pass rate) path with no pre-filter, callers
should fall back to query @ item_embs.T + torch.topk.

The kernel runs over a bucketed width P_BUCKET = _bucket_p(P) so the JIT
cache compiles once per bucket × D (the role formerly played by
@triton.autotune's cache key).
```

#### `oporp_1bit_match_topk_full` and `oporp_1bit_match_topk_indirect` — LinR V3 Hamming scoring

File: [retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py). Single device kernel `_oporp_1bit_match_topk_kernel` at lines 56–125 with a `HAS_INDICES: tl.constexpr` flag distinguishing full-scan vs indirect paths. Two public entry points wrap it.

Python entry signatures:
```python
@triton_op("retrieve::oporp_1bit_match_topk_full", mutates_args=())
def oporp_1bit_match_topk_full(
    query_bits: Tensor,   # [B, W] int64
    item_bits: Tensor,    # [N, W] int64
    k: int,
) -> tuple[Tensor, Tensor]: ...

@triton_op("retrieve::oporp_1bit_match_topk_indirect", mutates_args=())
def oporp_1bit_match_topk_indirect(
    query_bits: Tensor,        # [B, W] int64
    item_bits: Tensor,         # [N, W] int64
    k: int,
    positive_indices: Tensor,  # [B, P] int64
    counts: Tensor,            # [B] int64
) -> tuple[Tensor, Tensor]: ...
```

Device-kernel parameters: `N: tl.constexpr` (bucketed or full corpus), `W: tl.constexpr` (word count `D // 64`), `D_TOTAL: tl.constexpr` (= `64 * W`), `HAS_INDICES: tl.constexpr`, `BLOCK_N: tl.constexpr`.

`DEFAULT_CONFIG` at line 39:
```python
DEFAULT_CONFIG = Oporp1BitMatchTopkConfig(block_n=512, num_warps=4)
```
Tuned on A100; `block_n=512` dominates at `N ≥ 65 536` (5 of 8 regimes), with small-N regimes within ~10% noise.

Launch grid: `(cdiv(n_kernel, BLOCK_N), B)` where `n_kernel = max(_bucket_n(P), _bucket_n(k))` for the indirect path and `n_kernel = N` for the full-scan path.

What it does: per-tile popcount-based Hamming distance. The kernel loads one query signature `qb[b, :W]` (in registers), loads `BLOCK_N` item signatures `[BLOCK_N, W]` (either at indices `0..BLOCK_N` for full scan, or via `positive_indices[b, ·]` for indirect), XORs them, applies a SWAR `_popcount_int64` (defined at lines 43–52, no `libdevice` dependency) across all W words, sums to get per-candidate Hamming distance, converts to similarity `score = D_TOTAL - 2 * hamming` (a standard 1-bit similarity surrogate), masks invalid lanes to `-inf`, stores scores. Host calls `torch.topk`.

Memory access pattern: one cached `[W]` query signature load per program; `[BLOCK_N, W]` item-signature reads; in-register bit operations; `[BLOCK_N]` score store.

Triton primitives: custom `_popcount_int64(x)` (bit-twiddle), `tl.full([BLOCK_N], 0, tl.int1)`, `tl.where(HAS_INDICES, ...)` (constexpr-gated branch), `tl.load(..., mask=in_count, other=0)`, `tl.sum(..., axis=1)`.

Design comment (verbatim, lines ~154–167):
```
The HAS_INDICES path runs over a bucketed width n_kernel =
max(_bucket_n(positive_indices.shape[1]), _bucket_n(k)) so the JIT cache
compiles once per bucket × W (the role formerly played by
@triton.autotune's cache key) and the buffer always has >= k lanes for
torch.topk(scores, k). The full-scan path uses N=item_bits.shape[0]
directly — the registered index size is fixed per process and the layer
asserts k <= n_items_total.
```

A second comment at lines ~270–280 of the same file:
```
Eager entry point for tune scripts and parity tests. The compiled path
goes through the two @triton_op wrappers (oporp_1bit_match_topk_full /
oporp_1bit_match_topk_indirect) which mirror this body inline so that
wrap_triton is textually in the decorated function's source — torch.export's
kernel registry requires that.
```

**Pseudocode for `oporp_1bit_match_topk` (both `_full` and `_indirect` paths).** Single kernel body with `HAS_INDICES: tl.constexpr` switching between full-scan and sparse modes; the constexpr branch is folded at compile time so the two paths share JIT compilation cost only within a (W, BLOCK_N) bucket.

```pseudocode
Algorithm: 1-bit Hamming-similarity top-K (full-scan and indirect paths)
Inputs:
    $\widetilde{q}_b \in \mathbb{Z}_{64}^{W}$ — packed query sign bits (one per b)
    $\widetilde{s}_n \in \mathbb{Z}_{64}^{W}$ — packed item sign bits (one per n)
    $K$                                       — top-K size
    $\mathrm{HAS\_INDICES}$                   — constexpr flag (compile-time branch)
    if HAS_INDICES:
        $I_b \in \mathbb{Z}^{P}$              — candidate ids per batch element b
        $c_b \in \mathbb{Z}$                  — valid count for row b
Output:
    $\mathrm{topk\_ids}, \mathrm{topk\_scores}$

Step 1 (host wrapper):
    $D_{\mathrm{total}} \leftarrow 64 \cdot W$
    If HAS_INDICES: $N_{\mathrm{kernel}} \leftarrow \max(\mathrm{bucket}(P), \mathrm{bucket}(K))$
    Else:           $N_{\mathrm{kernel}} \leftarrow N$    (size of registered item index)
    Allocate $\mathrm{all\_scores} \in \mathbb{R}^{B \times N_{\mathrm{kernel}}}$

Step 2 (kernel, one program per (b, $n_0$) tile):
    Load $\widetilde{q}_b \in \mathbb{Z}_{64}^{W}$ into registers.
    For $i \in \{0, \ldots, \mathrm{BLOCK\_N}-1\}$:
        $n \leftarrow n_0 + i$
        $n_{\mathrm{valid}_i} \leftarrow (n < N_{\mathrm{kernel}})$
        If HAS_INDICES:
            $\mathrm{in\_count}_i \leftarrow (n < c_b)$
            $\mathrm{id}_i \leftarrow \mathrm{load}(I_b[n],\ \mathrm{mask}=\mathrm{in\_count}_i,\ \mathrm{other}=0)$
            $r_i \leftarrow \mathrm{load}(\widetilde{s}_{\mathrm{id}_i},\ \mathrm{mask}=\mathrm{in\_count}_i,\ \mathrm{other}=0)$
            $\mathrm{valid}_i \leftarrow \mathrm{in\_count}_i$
        Else:
            $r_i \leftarrow \mathrm{load}(\widetilde{s}_n,\ \mathrm{mask}=n_{\mathrm{valid}_i},\ \mathrm{other}=0)$
            $\mathrm{valid}_i \leftarrow n_{\mathrm{valid}_i}$
    Compute $\mathrm{xor}_i \leftarrow \widetilde{q}_b \oplus r_i \in \mathbb{Z}_{64}^{W}$
    Compute $\mathrm{hamming}_i \leftarrow \sum_{w=0}^{W-1} \mathrm{popcount\_int64}(\mathrm{xor}_i[w])$  (SWAR popcount)
    Compute $\mathrm{score}_i \leftarrow D_{\mathrm{total}} - 2 \cdot \mathrm{hamming}_i$ if $\mathrm{valid}_i$ else $-\infty$
    Store $\mathrm{score}_i$ to $\mathrm{all\_scores}[b, n]$

Step 3 (host-side):
    $(\mathrm{topk\_scores}, \mathrm{topk\_ids}) \leftarrow \mathrm{torch.topk}(\mathrm{all\_scores}, K, \mathrm{dim}=1)$
```

The SWAR `popcount_int64` is itself a `@triton.jit` helper (one of the two device-side helpers counted in the kernel inventory). Its body is the canonical Hamming-weight bit-twiddle, with no `libdevice` dependency:
```pseudocode
function popcount_int64(x):
    M1, M2, M4 := 0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F
    H01        := 0x0101010101010101
    x := x - ((x >> 1) & M1)
    x := (x & M2) + ((x >> 2) & M2)
    x := (x + (x >> 4)) & M4
    return ((x * H01) >> 56)  // upper byte holds the population count
```
Source: [kernels/linr/oporp_1bit_match_topk.py:42-52](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L42-L52). The same body appears in the torch reference (`tests/parity/test_oporp_1bit_match_topk.py`), which is why backend parity for this kernel is *bit-exact* (§4.9).

### §4.3.2 Co-designed (IVF + INT8 + Bloom) kernels

#### `codesigned_probe_score` — IVF + INT8 + Bloom fused (also `_bloom` variant)

File: [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py). Device kernel `_codesigned_probe_score_kernel` at lines 36–140 with `HAS_QB: tl.constexpr` flag distinguishing the plain-INT8 path from the Bloom-fused path. Two public entry points (`codesigned_probe_score`, `codesigned_probe_score_bloom`) at lines 278–349.

Python entry signatures:
```python
@triton_op("retrieve::codesigned_probe_score", mutates_args=())
def codesigned_probe_score(
    query: Tensor,              # [B, D] fp32 — pre-quantized in wrapper
    flat_probed_items: Tensor,  # [B, P] int64 (-1 padding)
    item_codes: Tensor,         # [N, D] int8
    global_scale: float,        # scalar paired with item_codes
    k: int,
) -> tuple[Tensor, Tensor]: ...

@triton_op("retrieve::codesigned_probe_score_bloom", mutates_args=())
def codesigned_probe_score_bloom(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,         # [B, W] int64 (Bloom)
    bloom_sigs: Tensor,         # [N, W] int64
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]: ...
```

Device-kernel parameters: `P: tl.constexpr`, `D: tl.constexpr`, `W: tl.constexpr` (or 1 if no Bloom), `HAS_QB: tl.constexpr`, `BLOCK_P: tl.constexpr`.

`DEFAULT_CONFIG` at line 32:
```python
DEFAULT_CONFIG = CodesignedProbeScoreConfig(block_p=256, num_warps=4)
```
Tuned on A100; `block_p=256` wins plurality (3/6 regimes on the `tl.dot` int8 × int8 path); `num_warps=4` wins all 6.

Launch grid (line 238):
```python
grid = (triton.cdiv(int(p), cfg.block_p), int(b))
```

What it does (the load-bearing fused kernel of the co-designed retriever). For each `(batch_b, tile_p)` program: (1) load pre-quantized query INT8 codes and the per-row query scale (computed once in the wrapper, outside the loop); (2) optionally load the query Bloom signature `qb[b, :W]`; (3) load `BLOCK_P` candidate ids from `flat_probed_items[b, p:p+BLOCK_P]`; (4) if Bloom is enabled, load `[BLOCK_P, W]` item signatures, test `(qb & sig) == qb` per word, AND-reduce over W to get a `keep` mask; (5) load `[BLOCK_P, D]` item INT8 codes for kept candidates; (6) compute INT8 × INT8 → INT32 dot via `tl.dot(q_codes_2d, codes_T, out_dtype=tl.int32)` (which Triton lowers to the `dp4a` CUDA-core instruction at M=1); (7) dequantize: `score = dots * q_scale * global_scale`; (8) store scores, with `-inf` for failed candidates or out-of-bounds (`item_id == -1`) lanes. Host calls `torch.topk(all_scores, k, dim=1)`.

Memory access pattern: registers cache the query INT8 codes (`[D]`), the scalar query scale, and the query Bloom signature (`[W]`); SRAM holds the `[BLOCK_P, D]` item-code tile and (if Bloom) `[BLOCK_P, W]` signatures; HBM-bound bandwidth dominates with the bandwidth saving coming from INT8 representation (4× over fp32, 2× over fp16).

Triton primitives: `tl.load(..., mask=valid[:, None], other=0)` (masked 2-D gather), `_or_combine(a, b)` reduction over W words for Bloom, `tl.dot(..., out_dtype=tl.int32)`, `tl.sum(..., axis=0)` (squeeze M=1), `tl.where(keep, dots, float("-inf"))`.

Design comments (verbatim, lines ~73–168, abridged):
```
Query is pre-quantized in the wrapper — symmetric per-row int8 with a
fp32 scale, paying one amax + div per batch row outside the loop.
Quantizing inside the kernel would recompute the amax in every (P_tile,
bid) program, redundant by a factor of cdiv(P, BLOCK_P).

int8 × int8 → int32 matmul (paper §4.2). Shapes: q[1, D] @ codes^T[D,
BLOCK_P] → out[1, BLOCK_P] int32. With M=1 (single query row per
program) IMMA tensor cores can't fit, so Triton lowers this to the dp4a
/ int8 CUDA-core path — the exact instruction the paper claims. Win
over the old dequant-then-fp32 design: 4× less code bandwidth (int8
stays narrow through the reduction) and dp4a's 4-mul-add throughput per
CUDA core.

The bloom subset test is fused into the same kernel — items that fail
get -inf without a separate scratch pass. The bloom intermediate
([B, P, W] sigs / bool match) and the int8 item-code tile ([B, P, D])
never touch HBM — they live in registers/SRAM.
```

The "paper §4.2" reference in the file's comment is to LinR (Borisyuk et al. 2024) — used internally as a design anchor for the INT8 scheme. Citation in body prose is to LinR for the INT8-with-dp4a idea (Ch.1 §1.7 already covers this).

**Pseudocode** for the writer (mandatory visual deliverable per contract — algorithm box). Variables marked with `$...$` for LaTeX rendering.

```pseudocode
Algorithm: Fused IVF + INT8 + Bloom probe-and-score kernel
Inputs:
    $q_b \in \mathbb{R}^{D}$    — query embedding (one per batch element b)
    $\mathrm{flat}_b \in \mathbb{Z}^{P}$ — concatenated probed-list item ids (−1 padding)
    $c_n \in \{-128,\ldots,127\}^{D}$ — INT8 item codes (one row per item n)
    $\sigma \in \mathbb{R}$     — global INT8 dequant scale
    $\widetilde{q}_b \in \mathbb{Z}^{W}$ — query Bloom signature (optional)
    $\widetilde{s}_n \in \mathbb{Z}^{W}$ — item Bloom signatures (optional)
    $K$                          — top-K size
Output:
    $\mathrm{topk\_ids}, \mathrm{topk\_scores}$ — top-K item ids and scores

Step 1 (wrapper, host-side):
    For each batch b: compute INT8 query codes $\hat{q}_b = \mathrm{round}(127 \cdot q_b / \max|q_b|)$
                       and per-row scale $\alpha_b = \max|q_b| / 127$.

Step 2 (kernel, one program per (b, $p_0$) tile):
    Load $\hat{q}_b \in \mathbb{Z}_{8}^{D}$ and $\alpha_b$ into registers.
    If Bloom is enabled, load $\widetilde{q}_b \in \mathbb{Z}_{64}^{W}$ into registers.
    For $i \in \{0, \ldots, \mathrm{BLOCK\_P}-1\}$:
        $p \leftarrow p_0 + i$
        $\mathrm{id}_i \leftarrow \mathrm{flat}_b[p]$
        $\mathrm{valid}_i \leftarrow (\mathrm{id}_i \ne -1)$
        If Bloom: load $\widetilde{s}_{\mathrm{id}_i} \in \mathbb{Z}_{64}^{W}$ (masked by $\mathrm{valid}_i$)
                   $\mathrm{keep}_i \leftarrow \mathrm{valid}_i \land \bigwedge_{w=0}^{W-1} \big((\widetilde{q}_b[w] \,\&\, \widetilde{s}_{\mathrm{id}_i}[w]) = \widetilde{q}_b[w]\big)$
        Else: $\mathrm{keep}_i \leftarrow \mathrm{valid}_i$
    Load $C \in \mathbb{Z}_8^{\mathrm{BLOCK\_P} \times D}$ where row $i$ = $c_{\mathrm{id}_i}$ (masked by $\mathrm{keep}_i$)
    Compute $d \leftarrow \mathrm{tl.dot}(\hat{q}_b^\top, C^\top, \mathrm{out\_dtype}=\mathrm{int32}) \in \mathbb{Z}^{\mathrm{BLOCK\_P}}$  (lowered to dp4a)
    Compute $\mathrm{score}_i \leftarrow \alpha_b \cdot \sigma \cdot d_i$ if $\mathrm{keep}_i$ else $-\infty$
    Store $\mathrm{score}_i$ to $\mathrm{all\_scores}[b, p]$

Step 3 (host-side):
    $(\mathrm{topk\_scores}, \mathrm{topk\_ids}) \leftarrow \mathrm{torch.topk}(\mathrm{all\_scores}, K, \mathrm{dim}=1)$
```

#### `codesigned_probe_score_exact` — IVF + INT8 + exact AND-of-OR

File: [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py). Device kernel at lines 28–127; Python entry at lines 262–337.

Python entry signature:
```python
@triton_op("retrieve::codesigned_probe_score_exact", mutates_args=())
def codesigned_probe_score_exact(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    item_clause_attrs: Tensor,   # [N, C, A_max] int64
    clause_is_reverse: Tensor,   # [C] bool (stored int8)
    query_clause_attrs: Tensor,  # [B, C] int64
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]: ...
```

Device-kernel parameters: `P: tl.constexpr`, `D: tl.constexpr`, `C: tl.constexpr` (clause count), `A_MAX: tl.constexpr` (max attribute values per clause), `BLOCK_P: tl.constexpr`.

`DEFAULT_CONFIG` at line 24:
```python
DEFAULT_CONFIG = CodesignedProbeScoreExactConfig(block_p=256, num_warps=4)
```
Mirrors `codesigned_probe_score` tuning (same P-grid).

Launch grid (line 219): `grid = (triton.cdiv(p, cfg.block_p), b)`.

What it does: sibling of `codesigned_probe_score` but with exact-clause predicate instead of Bloom. For each candidate: AND-reduce across `C` clauses, where each clause is `OR_{a=0..A_MAX} (item_attrs[id, c, a] == query_attrs[b, c])`, XOR-inverted by `clause_is_reverse[c]`, OR'd with the inactive-clause sentinel `query_attrs[b, c] == -1`. Passing items get INT8 scoring; failing items get `-inf`.

Memory access pattern: per-program, loads query INT8 codes + scale, loads `[BLOCK_P]` candidate ids, for each of `C` clauses loads `[BLOCK_P, A_MAX]` per-item attribute values (`C × A_MAX` int64 = `C × A_MAX × 8` bytes per item), evaluates predicate in registers via static loop unrolling, then INT8-scores survivors as in `codesigned_probe_score`.

Triton primitives: `tl.static_range(C)` and `tl.static_range(A_MAX)` (unrolled loops), `tl.load(..., mask=valid, other=-1)`, element-wise `==` and `|`, `^` with broadcast reverse flag, AND-reduce across clauses, then the same `tl.dot` INT8 path as the Bloom kernel.

Design comment (verbatim, abridged):
```
Per-item gather is C × A_max int64 — at C=2, A_max=2 that's 32 B/item
vs the bloom variant's W × 8 = 128 B/item at M=1024. The clause loop is
static-unrolled by Triton; the AND/OR/XOR chain stays in registers.

Same int8×int8 → int32 dp4a path as the bloom-variant kernel; see
codesigned_probe_score.py for the bandwidth rationale. For each
(query, probed-item) cell: evaluate the clause predicate against
item_clause_attrs[item_id], and (if it passes) score
(item_codes[id] · q_codes[b])_i32 * q_scale[b] * global_scale.
```

The cross-over between Bloom and exact kernels is governed by clause cardinality: for small `C × A_MAX` the exact kernel reads less per-item attribute data than a wide Bloom (e.g. at `C=2, A_max=2` the exact gather is 32 B/item vs `W=16` (m_bits=1024) Bloom's 128 B/item). The choice between them is exposed to the user via the `filter` parameter of `SilverTorch.__init__`.

#### `bloom_match` — standalone Bloom subset test

File: [retrieve/src/retrieve/kernels/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py). Device kernel at lines 10–50; Python entry at lines 57–80.

Python entry signature:
```python
@triton_op("retrieve::bloom_match", mutates_args=())
def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """Compute (qb & sig) == qb across W int64 words, AND-reduced.

    qb:   [B, W] int64 — packed query bloom signatures
    sigs: [N, W] int64 — packed item bloom signatures
    Returns: [B, N] bool
    """
```

Device-kernel parameters: `N: tl.constexpr`, `W: tl.constexpr`, `BLOCK_N: tl.constexpr` (hard-coded to 128; not offline-tuned).

Launch grid (line 72): `grid = (b, triton.cdiv(n, 128))`.

What it does: standalone Bloom predicate emitting a dense `[B, N]` bool. Used by `BloomFilter.evaluate_mask` (the dense path). For each `(batch_b, tile_n)` program: load query signature `[W]`, load `[BLOCK_N, W]` item signatures, test `(qb & sig) == qb` per word, take `min` across W (AND-reduce), interpret nonzero as pass, store `[BLOCK_N]` bool.

Triton primitives: `tl.load(..., mask=valid[:, None], other=0)`, `tl.min(..., axis=1)`, `all_eq != 0`, `tl.store(..., mask=valid)`.

### §4.3.3 Filter kernels

#### `clause_mask` — dense exact-clause evaluation

File: [retrieve/src/retrieve/kernels/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/filters/clause_mask.py). Device kernel at lines 38–98; Python entry at lines 160–211.

Python entry signature:
```python
@triton_op("retrieve::clause_mask", mutates_args=())
def clause_mask(
    item_clause_attrs: Tensor,    # [N, C, A_max] int64
    clause_is_reverse: Tensor,    # [C] bool (stored int8)
    query_clause_attrs: Tensor,   # [B, C] int64
) -> Tensor:                       # [B, N] bool
    """Fused clause evaluation → [B, N] bool. No intermediate."""
```

Device-kernel parameters: `C: tl.constexpr`, `A_MAX: tl.constexpr`, `BLOCK_N: tl.constexpr`. Note the 3D launch grid for large N (handles the 65535 grid-y limit).

`DEFAULT_CONFIG` at line 35:
```python
DEFAULT_CONFIG = ClauseMaskConfig(block_n=512, num_warps=2)
```
Tuned on A100 against Goodreads (N=797K), arXiv-retrieval (N=3M), and arXiv-synth (N=15M), with `C∈{4,5}`, `A_MAX=4`, `B∈{1,16}`. `block_n=512` and `num_warps=2` win all batched (B=16) regimes at N ≥ 3M.

Launch grid (lines 184–187):
```python
tiles = triton.cdiv(n, cfg.block_n)
tiles_x = triton.cdiv(tiles, 65535)
tiles_y = triton.cdiv(tiles, tiles_x)
grid = (b, tiles_y, tiles_x)
```
The 3D grid is a workaround for the per-axis grid-size limit at very large N (e.g. arXiv-synth-15M).

What it does: fused clause evaluation emitting `[B, N]` bool directly without materializing the `[B, N, C, A_max]` bool intermediate that a pure-torch broadcast equality would produce.

Triton primitives: `tl.static_range(C)`, `tl.static_range(A_MAX)`, `tl.full([BLOCK_N], 1, tl.int1)`, `tl.load(..., mask=n_valid, other=-1)`, `clause_match | (ia == q_c)` (OR-reduce across A_MAX), `clause_match ^ rev_c` (XOR with reverse flag), `pass_mask & clause_match` (AND-reduce across clauses).

Design comment (verbatim, lines 1–5):
```
Fused clause evaluation emitting [B, N] bool directly. Avoids the
[B, N, C, A_max] intermediate that ExactAttributeFilter.evaluate_mask's
pure-torch broadcast materializes. Same inner loop as clause_compact
minus the cumsum + atomic_add epilogue.
```

#### `clause_compact` — fused clause evaluation + stream compaction

File: [retrieve/src/retrieve/kernels/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/filters/clause_compact.py). Device kernel at lines 51–124; Python entry at lines 193–257.

Python entry signature:
```python
@triton_op("retrieve::clause_compact", mutates_args=())
def clause_compact(
    item_clause_attrs: Tensor,    # [N, C, A_max] int64
    clause_is_reverse: Tensor,    # [C] bool
    query_clause_attrs: Tensor,   # [B, C] int64
) -> tuple[Tensor, Tensor]:        # (positive_indices[B, N] int64, counts[B] int64)
```

`DEFAULT_CONFIG` at line 48:
```python
DEFAULT_CONFIG = ClauseCompactConfig(block_n=512, num_warps=2)
```
Tuned on A100 against same regimes as `clause_mask`; `block_n=512, num_warps=2` wins 3/5. Low `num_warps` chosen to avoid the cliff at `num_warps=8` on B=1.

Launch grid: 3D as for `clause_mask`.

What it does: identical clause-evaluation inner loop as `clause_mask`, plus a stream-compaction epilogue: per tile, convert `pass_mask` to int, take `tl.cumsum` along the tile (gives 0-indexed write offsets within the tile), `tl.sum` to get tile total, `tl.atomic_add` to `counts[b]` to get the row's base offset, then `tl.store` the passing item ids to `positive_indices[b, base + intra]`. Output ordering within a row is unspecified across tiles (atomic ordering is non-deterministic) — callers must sort if order matters; `fused_masked_knn_topk` only consumes the *set*.

Triton primitives: same as `clause_mask` plus `tl.where(pass_mask, 1, 0).to(tl.int32)`, `tl.cumsum(..., axis=0) - 1`, `tl.sum(pass_int)`, `tl.atomic_add(counts_ptr + bid, tile_sum.to(tl.int64))`, `tl.store(..., mask=pass_mask)`.

Design comment (verbatim, lines 1–14, abridged):
```
Avoids materializing the dense [B, N] bool that
ExactAttributeFilter.evaluate_mask would otherwise produce. One kernel
launch: per program (b, tile): evaluate clauses for BLOCK_N items
against query_clause_attrs[b], tl.cumsum within tile to get intra-tile
write offsets, tl.atomic_add into counts[b] to get the row's base
offset, tl.store passing item ids at positive_indices[b, base + intra].

Output ordering within a row is unspecified (atomics across tiles).
V2's fused_masked_knn_topk only consumes the *set*, not the order —
callers that care must sort.
```

#### `bloom_compact` — fused Bloom subset test + stream compaction

File: [retrieve/src/retrieve/kernels/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py). Device kernel at lines 48–104; Python entry at lines 166–225.

Python entry signature:
```python
@triton_op("retrieve::bloom_compact", mutates_args=())
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction.

    qb:   [B, W] int64 — packed query bloom signatures
    sigs: [N, W] int64 — packed item bloom signatures
    Returns (positive_indices[B, N] int64, counts[B] int64).
    """
```

`DEFAULT_CONFIG` at line 45:
```python
DEFAULT_CONFIG = BloomCompactConfig(block_n=256, num_warps=8)
```
Tuned on A100 against Goodreads / arXiv / arXiv-synth with W=16 (m_bits=1024). `block_n=256, num_warps=8` wins all batched regimes; wider W=16 causes register spill at `block_n ≥ 512, num_warps ≤ 4` (3–15× slowdown).

Launch grid: 3D as for `clause_compact`.

What it does: sibling of `clause_compact` for Bloom predicates. Per program loads query signature `[W]`, loads `[BLOCK_N, W]` item signatures, computes `(qb & sig) == qb` per word, AND-reduces over W via `tl.min`, converts to pass mask, cumsum + atomic_add + scatter store. Output ordering same convention as `clause_compact`.

Design comment (verbatim, abridged): mirrors `clause_compact` — avoids materializing the dense `[B, N]` bool, atomics produce unordered output per row.

### Writer's notes (§4.3 cross-section)

- **Bucketing strategy** is a recurring design pattern across LinR and co-designed kernels: rather than recompiling on every shape change, the host wrappers round `P` (or `N`) up to a small set of pre-chosen buckets (powers of two and selected catalogue-size landmarks). This replaces the role of `@triton.autotune`'s cache key and is documented at [docs/system/kernels.md:70-124](../system/kernels.md#L70-L124) — see §4.6.
- **3D launch grid** in filter kernels: the writer should note that Triton's per-axis grid limit of 65 535 forces a 3D grid `(B, tiles_y, tiles_x)` for very large N. This is a Triton implementation detail; the prose can mention it as one sentence under §4.3.3.
- **No `@triton.autotune`**: every kernel ships a single `DEFAULT_CONFIG`. Tuning happens offline via `tune.py` — defer the rationale to §4.6.
- **Pseudocode rendering**: two algorithm boxes are now included — `codesigned_probe_score` (§4.3.2, the contract-mandatory one) and `oporp_1bit_match_topk` (§4.3.1, added during the GPU-free completion pass). Suggested LaTeX environment: `algorithm` / `algorithmic` (algorithm2e package). Writer may keep both or fold the OPORP one into a single algorithm box if the chapter's algorithm-box budget is tight.

---

## §4.4 Бэкенд-абстракция (Backend abstraction)

### The `Backend` type

Source: [retrieve/src/retrieve/interfaces.py:8](../../retrieve/src/retrieve/interfaces.py#L8).

```python
Backend = Literal["torch", "triton"]
```

A `typing.Literal` alias rather than an `enum.Enum`. The rationale (not in code; inferred from the design): a string `Literal` integrates with `mypy` for static checking, can appear in `nn.Module` constructor signatures without import cycles, and serializes naturally to YAML evaluation configs (see [evaluation/config/](../../evaluation/config/)).

### Parity invariant

For every module that accepts a `backend` parameter, both code paths must compute the same retrieval result — the same top-K ids per query (within score-tie ambiguity) and the same scores within numerical tolerance (`atol=1e-3, rtol=1e-3` for fp32 dot products; bit-exact for the 1-bit Hamming case).

The invariant is enforced by the test suite — see §4.9. Specifically, the [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py) test parametrizes over `backend ∈ {"torch", "triton"}` and asserts identical valid-id sets per row.

### Dispatch pattern

The backend switch lives inside `forward` (or a delegate method called from `forward`). Three representative snippets follow.

`PrefilterKNN.forward`. Source: [retrieve/src/retrieve/layers/linr/prefilter_knn.py:47-58](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L47-L58).
```python
def forward(
    self,
    query: Tensor,
    candidate_ids: Tensor | None = None,
    counts: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    query = query.to(torch.float16)
    if candidate_ids is None:
        return self._forward_full(query)
    if self.backend == "triton":
        return self._forward_prefilter_triton(query, candidate_ids, counts)
    return self._forward_prefilter(query, candidate_ids, counts)
```

`OneBitKNN.forward`. Source: [retrieve/src/retrieve/layers/linr/one_bit_knn.py:93-101](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L93-L101).
```python
def forward(
    self,
    query: Tensor,
    candidate_ids: Tensor | None = None,
    counts: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if self.backend == "triton":
        return self._forward_triton(query, candidate_ids, counts)
    return self._forward_torch_eager(query, candidate_ids, counts)
```

`SilverTorch.forward`. Source: [retrieve/src/retrieve/layers/silvertorch/main.py:222-240](../../retrieve/src/retrieve/layers/silvertorch/main.py#L222-L240).
```python
def forward(
    self,
    query: Tensor,
    query_clause_attrs: Tensor | None = None,
    candidate_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if candidate_ids is not None:
        return self._forward_candidates(query, candidate_ids)
    if self.filter == "none" and query_clause_attrs is not None:
        raise ValueError(
            "query_clause_attrs requires filter='bloom' or filter='exact'"
        )
    if self.backend == "triton":
        return self._forward_triton(query, query_clause_attrs)
    return self._forward_torch_eager(query, query_clause_attrs)
```

### Edge case: PostfilterKNN and PostfilterKNNInt8

For the two dense post-filter modules, both `backend="torch"` and `backend="triton"` run identical code. Reason: the previous Triton `fused_matmul_topk` kernel was removed because cuBLAS (for the matmul) plus CUB (for the top-K) already saturate the GPU on dense `[B, D] @ [D, N]` workloads. See [docs/system/kernels.md:151-162](../system/kernels.md#L151-L162) for the design note. The `backend` argument is retained on these classes for API symmetry — every module in the LinR family accepts the same `__init__` shape — but it is documentation-only. Writer should note this honestly: the parity invariant is trivially satisfied here because there is only one path.

### Writer's notes

- The dispatch pattern is the same shape in every module: an outer `if self.backend == "triton"` check selecting one of two private delegate methods (`_forward_triton` vs `_forward_torch_eager` or `_forward_prefilter`). This uniformity is worth a sentence in the prose.
- Backwards link to §4.5: the `backend="torch"` paths are not just for parity testing — they are also the `torch.compile + Inductor` codepath for users who want whole-program fusion through PyTorch's compiler rather than through the Triton wrappers. See README line 49. The largest documented compile win (`OneBitKNN._score_full`, ~43× at B=64 / N=50k) is on the torch backend, *not* the Triton backend — see §4.5.

---

## §4.5 Поддержка `torch.compile` (torch.compile integration)

### Kernel-wrapper decoration

Every device-resident kernel that is reachable from user code goes through a single wrapper pattern: `@torch.library.triton_op("retrieve::<name>", mutates_args=())` decorates a host wrapper whose body contains a literal `wrap_triton(_<name>_kernel)[grid](...)` call. The decorator captures the launch as a higher-order operator (HOP) so `torch.compile` does not graph-break at the boundary; the literal `wrap_triton(...)` call keeps the kernel reference textually present in the function source, which torch.export's kernel registry requires.

All nine public kernel wrappers (verified via `grep -rn 'torch.library' src/retrieve/`):

| Wrapper | File:line | Op name |
|---|---|---|
| `clause_mask` | [kernels/filters/clause_mask.py:160](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L160) | `retrieve::clause_mask` |
| `clause_compact` | [kernels/filters/clause_compact.py:193](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L193) | `retrieve::clause_compact` |
| `bloom_compact` | [kernels/filters/bloom_compact.py:166](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L166) | `retrieve::bloom_compact` |
| `fused_masked_knn_topk` | [kernels/linr/fused_masked_knn_topk.py:218](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py#L218) | `retrieve::fused_masked_knn_topk` |
| `oporp_1bit_match_topk_full` | [kernels/linr/oporp_1bit_match_topk.py:262](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L262) | `retrieve::oporp_1bit_match_topk_full` |
| `oporp_1bit_match_topk_indirect` | [kernels/linr/oporp_1bit_match_topk.py:323](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L323) | `retrieve::oporp_1bit_match_topk_indirect` |
| `bloom_match` | [kernels/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py) | `retrieve::bloom_match` |
| `codesigned_probe_score` (+ `_bloom` sibling) | [kernels/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py) | `retrieve::codesigned_probe_score`, `retrieve::codesigned_probe_score_bloom` |
| `codesigned_probe_score_exact` | [kernels/silvertorch/codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) | `retrieve::codesigned_probe_score_exact` |

The package does **not** use `@torch.library.custom_op` anywhere — earlier internal documentation drafts may suggest otherwise, but `grep -rn '@torch.library.custom_op' src/retrieve/` returns no matches. The unified choice of `triton_op` reflects a deliberate uniformity: the compact (atomic-add) kernels and the pure-scoring kernels are decorated identically, because the atomic-mutation hazard is sidestepped by allocating fresh output buffers at every host-wrapper call (see §4.6 "atomic-add hazard sidestep").

### Wrapper / impl duplication pattern

Each kernel ships two host-side entry points:
- `_<name>_impl(..., *, config: <Config> | None = None)` — eager, accepts a config override. Used by `tune.py` and by parity tests. Lives early in the file.
- `<name>(...)` — public `@triton_op`-decorated wrapper. Hard-codes `DEFAULT_CONFIG`. Its body duplicates the launch logic of `_impl` so that `wrap_triton(_kernel)[grid](...)` is textually present in the decorated function — torch.export's kernel registry walks the function source to register the kernel reference, and a call through an indirect helper would not be visible.

The duplication is structural, not a cleanup target. Documented at [docs/system/kernels.md:94-104](../system/kernels.md#L94-L104). Excerpt from [kernels/linr/oporp_1bit_match_topk.py:162-167](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L162-L167):

```
Eager entry point for tune scripts and parity tests. The compiled
path goes through the two ``@triton_op`` wrappers
(``oporp_1bit_match_topk_full`` / ``oporp_1bit_match_topk_indirect``)
which mirror this body inline so that ``wrap_triton`` is textually
in the decorated function's source — torch.export's kernel registry
requires that.
```

### No graph breaks — enforcement

The package guarantees zero graph breaks at module boundaries when wrapped in `torch.compile(dynamic=True, mode="reduce-overhead")`. The guarantee is enforced by a regression test:

Source: [retrieve/tests/compile/test_silvertorch_compile.py:66-82](../../retrieve/tests/compile/test_silvertorch_compile.py#L66-L82).

```python
@pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
def test_no_graph_breaks_on_forward(filter_mode):
    """``torch._dynamo.explain`` reports zero graph breaks on the forward.

    A non-zero count means the kernel host wrappers are still graph-breaking
    (e.g. ``@torch._dynamo.disable`` slipped back in) or the layer reintroduced
    a host sync (``.item()`` on global_scale, Optional Tensor branching).
    """
    eager = _build(filter_mode)
    b, c = 4, 2
    query = make_query(b, eager.item_codes.shape[1])
    q_attrs = make_query_attrs(b, c=c) if filter_mode != "none" else None

    explanation = torch._dynamo.explain(eager.forward)(query, q_attrs)
    assert explanation.graph_break_count == 0, (
        f"filter={filter_mode}: expected 0 graph breaks, got "
        f"{explanation.graph_break_count}\n{explanation}"
    )
```

### `cudagraph_trees` + `mode="reduce-overhead"` compatibility

The package is designed for capture into a single `cudagraph_trees` graph under `torch.compile(dynamic=True, mode="reduce-overhead")`. Two design choices are load-bearing:

**1. No host syncs in the forward pass.** Any `.item()` call would force a device-to-host transfer, breaking cudagraph_trees capture. The most critical case is the `global_scale` field on `SilverTorch`: it is stored as a Python `float` (computed once at `register_index` time) rather than a 0-dim tensor, specifically to avoid a `.item()` call inside `forward`. Source: [retrieve/src/retrieve/layers/silvertorch/main.py:190-192](../../retrieve/src/retrieve/layers/silvertorch/main.py#L190-L192).

**2. Loop-free tensor flow in pre-kernel preprocessing.** The Bloom query-signature build at [retrieve/src/retrieve/layers/filters/bloom.py:211-225](../../retrieve/src/retrieve/layers/filters/bloom.py#L211-L225) is implemented as a sequence of pure tensor operations (multiply, index_select, OR, shift) with no Python control flow over batch elements. Under `torch.compile`, the entire build collapses into one Inductor-fused kernel; without this discipline it would be ~15 kernels (~0.4 ms eager, flat in B) and would graph-break on the Python loop.

The OPORP query projection at [retrieve/src/retrieve/layers/utils/quantize.py:111-128](../../retrieve/src/retrieve/layers/utils/quantize.py#L111-L128) is similarly loop-free for the same reason.

References to cudagraph_trees and `mode="reduce-overhead"` are scattered across the kernel-side wrappers as design comments:
- [kernels/filters/clause_mask.py:169](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L169) — "...can stitch into surrounding cudagraphs..."
- [kernels/filters/clause_compact.py:209-210](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L209-L210) — "...`torch.compile(dynamic=True, mode='reduce-overhead')` can stitch into a single cudagraph_trees graph..."
- [kernels/filters/bloom_compact.py:182-183](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L182-L183) — same design note.
- [kernels/silvertorch/bloom_match.py:63-64](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py#L63-L64) — same.

### `dynamic=True` SymInt handling

For users who run with `torch.compile(dynamic=True)` and want one graph to cover multiple `(B, N, W)` shapes, the package keeps derived sizes symbolic. Example: in [retrieve/src/retrieve/layers/linr/one_bit_knn.py:25-26](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L25-L26), the `d_total` value (used as the multiplicative constant in the similarity formula `score = d_total - 2 * hamming`) is derived from `item_bits.shape[1]` inside the forward body, not cached as a Python int at `register_index` time. This keeps `d_total` symbolic so a single compiled graph covers all dimensions.

### Compile-mode speedups — qualitative measurements

No per-module compile-vs-eager benchmark JSON exists in [evaluation/results/](../../evaluation/results/). The only quantitative measurements are figures in the design documentation at [docs/system/kernels.md:574-595](../system/kernels.md#L574-L595), referring to code regions wrapped by `torch.compile(dynamic=True, mode="reduce-overhead")` at the algo-harness layer.

| Region | Source | Measured speedup (compile / eager) | Origin of the win |
|---|---|---|---|
| `project_oporp_1bit_query` (per-query OPORP) | [layers/utils/quantize.py:111-128](../../retrieve/src/retrieve/layers/utils/quantize.py#L111-L128) | ~2.5× at B=8 and B=64 | Launch-tax elision via cudagraph_trees capture (~5 small kernels collapse into one captured graph). |
| `OneBitKNN._score_full` (full-scan 1-bit reference) | [layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) | **~3× at B=8, ~43× at B=64 with N=50 000** — largest compile win documented in the repo | Inductor fusion: the eager path materializes `[B, N, W]` XOR once and re-streams it through six SWAR popcount ops; Inductor fuses XOR + popcount + reduce into a single elementwise Triton kernel. |
| `PostfilterKNN`, `FullScanKNN` (dense matmul + topk) | [layers/linr/postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py), [layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | Compile wrappers tried then reverted | cuBLAS + CUB already win the heavy op; cudagraph capture + mandatory output clone (to escape the `reduce-overhead` buffer pool) cost more than they save. |

Verbatim source of these numbers — [docs/system/kernels.md](../system/kernels.md):
```
- project_oporp_1bit_query (quantize.py). Per-query OPORP projection:
  multiply → index_select → sign-pack (~5 small kernels in eager).
  Under the outer cudagraph_trees the launch tax collapses — measured
  ~2.5× speedup at B=8 and B=64.
- OneBitKNN._score_full (one_bit_knn.py). xor → popcount → reduce over
  the full corpus. The win here is fusion, not launch elision: eager
  materializes the [B, N, W] xor once and re-streams it through six SWAR
  popcount ops; Inductor fuses those into a single elementwise triton
  kernel. Measured ~3× at B=8, ~43× at B=64, N=50k — by far the largest
  compile win in the repo.
The matmul-bearing references (PostfilterKNN, FullScanKNN) were tried
with their own dedicated compile wrappers and reverted — cuBLAS + CUB
already win the heavy op, and the cudagraph capture + mandatory output
clone (to escape the reduce-overhead buffer pool) cost more than they save.
```

For the Triton-backed retrieval modules (`PrefilterKNN`, `OneBitKNN.triton`, `SilverTorch`), the speedup of the **Triton backend itself** versus the pure-torch fallback is reported in Ch.6 (latency tables); `torch.compile`-on-top-of-Triton is enabled by the `@triton_op` HOP capture (see "Kernel-wrapper decoration" above) but the marginal compile speedup is small because the Triton kernel is already the dominant cost.

**Recommendation for the writer**: lift the table above into the chapter directly. The benchmark-sweep alternative ("add compile-vs-eager to Ch.6 harness") is out of scope for the current chapter and would require GPU runs. The compile-path correctness is independently verified by the regression test at [retrieve/tests/compile/test_silvertorch_compile.py:44-82](../../retrieve/tests/compile/test_silvertorch_compile.py#L44-L82) — every filter mode passes both numerical-equivalence and zero-graph-break checks.

### Writer's notes

- All 9 public kernel wrappers use `@torch.library.triton_op` uniformly — no `custom_op` is used anywhere in the package. The earlier-noted "split" between `custom_op` (compact/atomic) and `triton_op` (scoring) was incorrect; the atomic-mutation hazard for compact kernels is sidestepped at the host wrapper level (fresh output buffers per call) rather than via decorator choice. Worth one sentence in the prose to dispel a plausible-sounding but wrong intuition that a reader might bring from other Triton-on-PyTorch codebases.
- The graph-break regression test is a strong reproducibility argument — cite it as a citation `[CODE: tests/compile/test_silvertorch_compile.py:66-82]` in the prose.
- The OPORP projection's loop-free design is non-obvious; explain in 2–3 sentences that the Python loop over batch elements was deliberately replaced with tensor flow.
- [TODO: clarify with author — does the chapter narrative want to discuss `torch.export` integration too, or is that out of scope? The `triton_op` decorator is partly motivated by torch.export — see [docs/system/kernels.md:94-104](../system/kernels.md#L94-L104) — and [docs/plans/torch-export-refactor.md](../plans/torch-export-refactor.md) tracks future work in this direction. Per the contract, §4.5 is "torch.compile поддержка", not "torch.export"; recommend brief mention only.]

---

## §4.6 Оффлайн autotuning (Offline autotuning)

### Why no `@triton.autotune`?

The package deliberately does not use Triton's built-in `@triton.autotune` decorator. The rationale is documented at [docs/system/kernels.md:70-124](../system/kernels.md#L70-L124). Three reasons (paraphrased from the source doc):

1. **Cudagraph leakage on re-tune.** `@triton.autotune` re-runs the candidate sweep whenever the cache key changes (a new shape combination). This re-tuning leaks into the `torch.compile(dynamic=True, mode="reduce-overhead")` capture: a cudagraph_trees graph captured against the kernel during one autotune trial becomes invalid when the next trial runs with a different config. The result is silent recompilation pressure and, in the worst case, corrupted graph state.

2. **Atomic-add corruption in compact kernels.** Compact kernels (`clause_compact`, `bloom_compact`) write outputs via `tl.atomic_add`. The Triton autotune harness re-runs the candidate configs in sequence on the same output buffer; the atomics accumulate across trials, producing wrong "winning config" measurements and (worse) silently incorrect outputs that pass the autotune's own equality check.

3. **No empirical grounding.** Older versions of the kernels carried hard-coded `_BLOCK_N = 256`-style constants that were guessed once and never re-measured. The offline tuner replaces these with measured defaults against real evaluation-time shapes (see "Default regimes" below).

The offline approach replaces autotune with a single curated `DEFAULT_CONFIG` per kernel, chosen by a one-time sweep against representative shapes. The constant is shipped in the source file; runtime invocation never re-tunes. Override is possible by passing `config=<Config>(...)` to the private `_impl` function, used in parity tests and in the tuner itself.

### CLI entry point

Source: [retrieve/pyproject.toml:13-14](../../retrieve/pyproject.toml#L13-L14).

```toml
[project.scripts]
tune-kernels = "retrieve.tune:main"
```

The user invokes the tuner from the shell as:
```
uv run tune-kernels <subcommand> [--regime N,B,...] [--device cuda:0] [--json-out PATH]
```

Subcommands (one per tunable kernel, per [retrieve/src/retrieve/tune.py](../../retrieve/src/retrieve/tune.py)):
- `fused-masked-knn-topk [--d 128] [--b 16]`
- `oporp-1bit-match-topk [--w 2] [--b 16]`
- `codesigned-probe-score [--d 128] [--b 16] [--w 4]`
- `clause-mask [--regime N,B,C,A_MAX]…`
- `clause-compact [--regime N,B,C,A_MAX]…`
- `bloom-compact [--regime N,B,W]…`

Each subcommand sweeps a hard-coded grid of `(block, num_warps)` combinations against a built-in set of shape regimes (or user-supplied regimes via `--regime`), measures median latency via `triton.testing.do_bench` (500 reps, 100 warmup), and emits the winning config as a paste-able Python line (`DEFAULT_CONFIG = ...(...)`).

### `DEFAULT_CONFIG` per kernel

The six tunable kernels each define a `@dataclass(frozen=True)` config and a `DEFAULT_CONFIG` constant:

| Kernel | Config dataclass | DEFAULT_CONFIG values | File:line |
|---|---|---|---|
| `fused_masked_knn_topk` | `FusedMaskedKnnTopkConfig(block_n, num_warps, num_stages=3)` | `block_n=32, num_warps=8` | [kernels/linr/fused_masked_knn_topk.py:39](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py#L39) |
| `oporp_1bit_match_topk` | `Oporp1BitMatchTopkConfig(block_n, num_warps, num_stages=3)` | `block_n=512, num_warps=4` | [kernels/linr/oporp_1bit_match_topk.py:39](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L39) |
| `clause_mask` | `ClauseMaskConfig(block_n, num_warps, num_stages=3)` | `block_n=512, num_warps=2` | [kernels/filters/clause_mask.py:35](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L35) |
| `clause_compact` | `ClauseCompactConfig(block_n, num_warps, num_stages=3)` | `block_n=512, num_warps=2` | [kernels/filters/clause_compact.py:48](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L48) |
| `bloom_compact` | `BloomCompactConfig(block_n, num_warps, num_stages=3)` | `block_n=256, num_warps=8` | [kernels/filters/bloom_compact.py:45](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L45) |
| `codesigned_probe_score` (+ exact) | `CodesignedProbeScoreConfig(block_p, num_warps, num_stages=3)` | `block_p=256, num_warps=4` | [kernels/silvertorch/codesigned_probe_score.py:32](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L32), [.../codesigned_probe_score_exact.py:24](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py#L24) |

All six configs were tuned on A100 (sm_80). The shipped values are not necessarily optimal on sm_86 (L40, RTX 4090) or sm_90 (H100); the design assumes that users wanting cross-architecture optimum re-run `tune-kernels` on their target hardware and override `DEFAULT_CONFIG`.

### Tuning flow (illustrative excerpt)

Source: [retrieve/src/retrieve/tune.py:137-188](../../retrieve/src/retrieve/tune.py#L137-L188) (function `_tune_fmkt`). The pattern is similar across all tuners:

1. For each P-bucket (catalogue size class):
2. Generate random test tensors: `query[B, D]`, `embeddings[N, D]`, `positive_indices[B, P]`, `counts[B]`.
3. For each `(block_n, num_warps)` in the candidate grid:
   - Warm the JIT cache by running 3 iterations of `_fused_masked_knn_topk_impl(..., config=cfg)`.
   - Measure with `triton.testing.do_bench(500 reps, 100 warmup)` → median ms.
   - Record `(config, median_ms)`.
4. For each bucket, pick the config with lowest median.
5. Aggregate to a single `DEFAULT_CONFIG` via plurality vote across all buckets (ties broken by lower `num_warps`).
6. Print paste-able line: `DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n=…, num_warps=…)`.

### Candidate grids and default regimes

Source: [retrieve/src/retrieve/tune.py:67-105](../../retrieve/src/retrieve/tune.py#L67-L105).

```python
_FMKT_GRID = [(bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8)]            # 8 combos
_OPORP_GRID = [(bn, nw) for bn in (64, 128, 256, 512) for nw in (4, 8)]          # 8 combos
_CPS_GRID = [(bp, nw) for bp in (32, 64, 128, 256) for nw in (4, 8)]             # 8 combos
_CLAUSE_MASK_GRID    = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]  # 12 combos
_CLAUSE_COMPACT_GRID = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]  # 12 combos
_BLOOM_COMPACT_GRID  = [(bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8)]  # 12 combos

_DEFAULT_CLAUSE_REGIMES = (
    (797_085,    1, 4, 4),   # goodreads, single-query
    (797_085,   16, 4, 4),   # goodreads, batched
    (2_988_997,  1, 5, 4),   # arxiv-retrieval, single-query
    (2_988_997, 16, 5, 4),   # arxiv-retrieval, batched
    (15_000_001,16, 5, 4),   # arxiv-synth-15M, batched (DRAM-streaming extreme)
)

_DEFAULT_BLOOM_REGIMES = (
    (797_085,    1, 16),     # N, B, W (W=16 from m_bits=1024 default)
    (797_085,   16, 16),
    (2_988_997,  1, 16),
    (2_988_997, 16, 16),
    (15_000_001,16, 16),
)

_DEFAULT_CPS_P_GRID = (1024, 8192, 65536)  # P spans typical IVF sizings
```

The default regimes are derived from the real evaluation shapes used in Ch.6: Goodreads (N=797 085), arXiv (N=2 988 997), arXiv-synth (N=15M for DRAM-streaming stress test). Batch sizes 1 and 16 mirror the `evaluation/retrieval/config` defaults.

### Atomic-add hazard sidestep

For the compact kernels (`clause_compact`, `bloom_compact`), the offline tuner sidesteps the atomic-add corruption hazard naturally: each call to the host wrapper allocates *fresh* `out_indices` (filled with -1) and `counts` (zeros) buffers, so there is no cross-rep accumulation during `do_bench` repetitions. Each measurement is independent. By contrast, `@triton.autotune` would re-use the same output buffer across trials, accumulating spurious atomic adds. Documented at [docs/system/kernels.md:116-123](../system/kernels.md#L116-L123).

### Writer's notes

- The autotuning section is a strong contribution paragraph for the thesis: the package documents *why* `@triton.autotune` is unsuitable for retrieval kernels in three concrete ways, and presents an alternative pattern (offline tuner + curated default + override hook) that is reproducible and CI-friendly.
- A100 hardware caveat: the shipped defaults are sm_80-tuned. For the thesis writer, this is worth stating once but not relitigating — the package's portability is a Ch.7 limitation, not a Ch.4 implementation flaw.
- [TODO: clarify with author — does the chapter need a screenshot or transcript of an actual `tune-kernels` run? If so, the writer should run `uv run tune-kernels clause-mask` against a small N regime and capture the output.]

---

## §4.7 Композиция фильтров (Filter composition)

### `FilterModule` contract (verbatim)

Source: [retrieve/src/retrieve/interfaces.py:11-57](../../retrieve/src/retrieve/interfaces.py#L11-L57).

```python
class FilterModule(nn.Module, abc.ABC):
    """Boolean predicate over an item index.

    Concrete filters expose three native paths over a registered item set,
    keyed by the consumer's preferred shape:

    - ``evaluate_mask(q) -> [B, N] bool`` — dense; consumed by V1 / V3 mask path,
      and by ``combine_masks`` for AND-of-masks composition.
    - ``evaluate_indices(q) -> ([B, P] int64, [B] int64)`` — compact; consumed by
      V2 / V3 candidate path. Default falls back to ``compact_mask(evaluate_mask)``;
      override when a fused compact kernel exists.
    - ``evaluate_subset(q, candidate_ids) -> [B, P] bool`` — apply this filter
      only to the given candidate ids. Default gathers columns of
      ``evaluate_mask``; override when a per-row check is much cheaper than full
      evaluation (e.g. P ≪ N).

    ``forward`` is an alias to ``evaluate_mask``.
    """

    @abc.abstractmethod
    def register_index(
        self,
        item_clause_attrs: Tensor,
        item_embs: Tensor | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor: ...

    def evaluate_indices(
        self,
        query_clause_attrs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from retrieve.layers.utils.compact import compact_mask
        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        mask = self.evaluate_mask(query_clause_attrs)
        return mask.gather(1, candidate_ids)

    def forward(self, query_clause_attrs: Tensor) -> Tensor:
        return self.evaluate_mask(query_clause_attrs)
```

The contract is a thin ABC. Two methods are abstract (`register_index`, `evaluate_mask`); the other two methods (`evaluate_indices`, `evaluate_subset`) have default implementations that subclasses override only when a fused kernel exists. `forward` is a convenience alias for `evaluate_mask` (lets a `FilterModule` be called like a `Callable[[Tensor], Tensor]`).

The three evaluation paths align with the three consumer shapes in the LinR family:
- `evaluate_mask` → dense `[B, N]` bool → consumed by `PostfilterKNN` (mask path) and `OneBitKNN._score_full` (mask path).
- `evaluate_indices` → sparse `([B, P], [B])` int64 → consumed by `PrefilterKNN` and `OneBitKNN._score_indirect` (sparse path).
- `evaluate_subset(q, candidate_ids)` → `[B, P]` bool → used by `combine_indices` for the cascade pattern.

### `ExactAttributeFilter` — exact AND-of-OR with reverse

Source: [retrieve/src/retrieve/layers/filters/exact_attribute.py:12-108](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L12-L108).

```python
class ExactAttributeFilter(FilterModule):
    """Standalone exact clause-attribute filter. Decoupled from any retrieval
    module — callers compose: f = ExactAttributeFilter(); f.register_index(item_attrs);
    mask = f.evaluate_mask(qa); ids, cs = f.evaluate_indices(qa).

    backend="triton" (default) routes the dense / compact paths through
    the fused clause_mask / clause_compact Triton kernels. With backend="torch",
    the same semantics run via a broadcast equality + reduction — but materializes
    [B, N, C, A_max] bool intermediate, so expect HBM spikes at large N.
    """

    item_clause_attrs: Tensor  # [N, C, A_max] int64
    clause_is_reverse: Tensor  # [C] bool

    def __init__(self, backend: Backend = "triton") -> None:
        super().__init__()
        self.backend = backend
```

`evaluate_mask` and `evaluate_indices` dispatch on `self.backend`:

```python
def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
    if self.backend == "triton":
        return clause_mask(
            self.item_clause_attrs,
            self.clause_is_reverse,
            query_clause_attrs,
        )
    q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)
    ic = self.item_clause_attrs.unsqueeze(0)
    match = q == ic
    clause_pass = match.any(dim=-1)
    rev = self.clause_is_reverse.unsqueeze(0).unsqueeze(0)
    clause_pass = torch.where(rev, ~clause_pass, clause_pass)
    inactive = (query_clause_attrs == -1).unsqueeze(1)
    clause_pass = clause_pass | inactive
    return clause_pass.all(dim=-1)

def evaluate_indices(self, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
    if self.backend == "triton":
        return clause_compact(
            self.item_clause_attrs,
            self.clause_is_reverse,
            query_clause_attrs,
        )
    return compact_mask(self.evaluate_mask(query_clause_attrs))
```

Schema:
- Items: `[N, C, A_max]` int64. `C` is the number of clauses, `A_max` the maximum number of attribute values per clause; padding value is `-1`.
- Query: `[B, C]` int64. Exactly one attribute value per clause (or `-1` for "inactive clause", which always passes).
- Reverse: `[C]` bool. If set, the clause is NOT-inverted (XOR with predicate result).
- Semantics: OR within a clause over its `A_max` attribute slots; AND across all clauses (with optional per-clause NOT).

### `BloomFilter` — conjunctive Bloom-signature subset test (paper-strict)

Source: [retrieve/src/retrieve/layers/filters/bloom.py:12-106](../../retrieve/src/retrieve/layers/filters/bloom.py#L12-L106).

```python
class BloomFilter(FilterModule):
    """Per-item Bloom-signature attribute filter (paper-strict, conjunctive).

    Items: [N, C, A_max] int64 with -1 padding. Each item's signature is
    the OR of k_hash hash positions per non-pad attribute, packed into
    W = m_bits // 64 int64 words. Query: [B, C] int64 (single attribute
    per clause; -1 is inactive). Subset test (qb & sigs) == qb per word,
    AND-reduced.

    backend="triton" (default) routes the dense / compact paths through
    the fused bloom_match / bloom_compact Triton kernels. With backend="torch"
    the same semantics run via a pure-torch broadcast bitwise test, which
    materializes a [B, N, W] int64 intermediate.

    No reverse / NOT — that path stays in ExactAttributeFilter.
    """

    bloom_sigs: Tensor  # [N, W] int64
    hash_seeds: Tensor  # [k_hash, 2] int64

    def __init__(self, m_bits: int, k_hash: int, backend: Backend = "triton") -> None:
        super().__init__()
        if m_bits <= 0 or (m_bits & (m_bits - 1)) != 0:
            raise ValueError(f"m_bits must be a positive power of 2, got {m_bits}")
        if m_bits % 64 != 0:
            raise ValueError(f"m_bits must be a multiple of 64, got {m_bits}")
        if k_hash <= 0:
            raise ValueError(f"k_hash must be positive, got {k_hash}")
        self.m_bits = m_bits
        self.k_hash = k_hash
        self.word_count = m_bits // 64
        self.backend = backend
```

The `evaluate_subset` override (paper-strict, conjunctive):

```python
def evaluate_subset(
    self,
    query_clause_attrs: Tensor,
    candidate_ids: Tensor,
) -> Tensor:
    qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
    sigs = self.bloom_sigs[candidate_ids]            # [B, P, W]
    match = (qb.unsqueeze(1) & sigs) == qb.unsqueeze(1)  # [B, P, W]
    return match.all(dim=-1)
```

### Bloom keying invariant (critical)

The Bloom signature builder hashes `(clause_idx, value)` *tuples*, not raw attribute values. Source: design notes at [docs/system/filtering.md](../system/filtering.md). Per-clause salt mixed into post-hash bits via `_mix64(clause_id, ...)`. Rationale: a query for a single clause on a value that also appears in another clause's vocabulary would otherwise leak ~25–30 % false positives via cross-clause collision. With per-clause keying, the FPR returns to the theoretical `(1 - e^{-KN/M})^K` of the bare Bloom filter.

This is an implementation detail that is critical to correctness and worth one sentence in the prose. The keying scheme is also why `BloomFilter` is paper-strict-conjunctive: NOT-inverting a per-clause signature breaks the subset-test semantics, so `clause_is_reverse=True` raises `ValueError` in `register_index`.

### Composition helpers

#### `combine_masks` — dense AND of masks

Source: [retrieve/src/retrieve/layers/filters/__init__.py:13-23](../../retrieve/src/retrieve/layers/filters/__init__.py#L13-L23).

```python
def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional [B, N] masks.

    None inputs are ignored. Returns None if every input is None.
    """
    out: Tensor | None = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out
```

Simple element-wise AND with `None`-tolerance. No selectivity-aware routing; the caller has already committed to the dense path (one `[B, N]` materialization per filter).

#### `combine_indices` — sparse cascade

Source: [retrieve/src/retrieve/layers/filters/__init__.py:26-70](../../retrieve/src/retrieve/layers/filters/__init__.py#L26-L70).

```python
def combine_indices(
    filters: Sequence[FilterModule],
    query_clause_attrs: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Sparse cascade across multiple filters.

    The first filter produces (ids, counts) via its native compact path;
    each subsequent filter is applied to those ids via evaluate_subset and
    the survivors are re-compacted. No [B, N] is materialized by the cascade
    itself — only the first filter's evaluate_indices may, depending on its
    implementation.

    Caller orders filters most-selective first.
    """
    if len(filters) == 0:
        raise ValueError("combine_indices requires at least one filter")
    if len(filters) != len(query_clause_attrs):
        raise ValueError(
            f"filters / queries length mismatch: {len(filters)} vs {len(query_clause_attrs)}",
        )

    f0, q0 = filters[0], query_clause_attrs[0]
    ids, counts = f0.evaluate_indices(q0)

    for f, q in zip(filters[1:], query_clause_attrs[1:]):
        b, p = ids.shape
        if p == 0:
            return ids, counts
        valid = torch.arange(p, device=ids.device).unsqueeze(0) < counts.unsqueeze(1)
        # ids past counts[b] are scratch from clause_compact's torch.empty() —
        # gather them safely as 0; sub_mask & valid zeroes those positions out.
        safe_ids = torch.where(valid, ids, ids.new_zeros(()))
        sub_mask = f.evaluate_subset(q, safe_ids)  # [B, P] bool
        sub_mask = sub_mask & valid
        new_counts = sub_mask.sum(dim=1)
        new_p = int(new_counts.max().item())
        if new_p == 0:
            ids = torch.empty(b, 0, dtype=torch.long, device=ids.device)
            counts = new_counts
            continue
        sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
        ids = ids.gather(1, sorted_idx)
        counts = new_counts

    return ids, counts
```

The cascade is the only place where the API exposes a host-sync (`int(new_counts.max().item())`). This is intentional: subsequent filters need a concrete `P` to size their per-row work, and the alternative — keeping `P` symbolic and re-allocating worst-case buffers per filter — would defeat the point of the cascade.

#### `compact_mask` — torch fallback

Source: [retrieve/src/retrieve/layers/utils/compact.py:6-17](../../retrieve/src/retrieve/layers/utils/compact.py#L6-L17).

```python
def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a [B, N] bool mask to (positive_indices[B, N], counts[B]).

    Returns the full [B, N] argsort; rows shorter than the row max are
    right-padded with arbitrary item ids — callers must use counts to
    bound valid reads. Matches the triton bloom_compact / clause_compact
    contract verbatim so the torch-backend fallback and the triton path are
    interchangeable. No .item() host sync.
    """
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts
```

Returns the full-width `[B, N]` argsort (not a compact `[B, P]`), so downstream consumers must use `counts` to bound their reads. This matches the contract of the Triton compact kernels exactly — both produce `[B, N]` int64 with valid items at positions `[0, counts[b])` and padding past that. The torch fallback explicitly avoids `.item()` to stay symbolic under `torch.compile(dynamic=True)`.

### Cost model — dense vs sparse composition

**The package does not implement an automatic cost model for choosing between `combine_masks` (dense) and `combine_indices` (sparse).** The choice is exposed to the caller via the API: if the caller picks the mask path on every filter, the composition is dense; if the caller picks the indices path on the first filter and `evaluate_subset` on the rest, the composition is sparse.

The trade-off:
- **Dense** (`combine_masks`): one `[B, N]` bool per filter; final AND is one elementwise op. Best when pass rates are high (most items survive each filter) and N is small enough that the `[B, N]` materialization is cheap.
- **Sparse** (`combine_indices`): the first filter materializes `[B, P_0]` with `P_0 ≤ N`; each subsequent filter only checks `P_{i-1}` candidates. Best when pass rates are low (filters are selective) and N is large.

The break-even point depends on the catalogue size, the filter pass-rate distribution, and the cost ratio of `evaluate_mask` vs `evaluate_subset` for each filter. Choosing automatically would require either runtime profiling (a setup the package deliberately avoids — see the no-`@triton.autotune` argument in §4.6) or a static cost model with hand-tuned parameters. The current design pushes the decision to the caller; the Ch.6 evaluation harness picks per scenario via its YAML configs.

### Worked example: two-filter sparse cascade

To illustrate the sparse cascade, consider an arXiv-style retrieval scenario with two filters:
- $f_0$ — `BloomFilter(m_bits=1024, k_hash=4)` registered against `[N, C, A_max]` clause attributes — most-selective, pass rate ~5 %.
- $f_1$ — `ExactAttributeFilter()` registered against the same `[N, C, A_max]` tensor — applied in subset mode to verify Bloom positives and reject false positives.

For a single query $q$ (batch $B=1$, hence omit the $b$ subscript), with $N = 10^6$:

```pseudocode
# Step 0: caller orders most-selective first
filters  = [f0_bloom, f1_exact]
queries  = [q_bloom_attrs, q_exact_attrs]   # both [1, C]

# Step 1: first filter's native sparse path
ids0, counts0 = f0_bloom.evaluate_indices(q_bloom_attrs)
    # On the Triton path this calls bloom_compact(qb, sigs):
    #   - kernel walks [N, W] item signatures in tiles of BLOCK_N=256
    #   - per tile: tile_pass_mask = (qb & sigs) == qb  (AND across W words)
    #   - tile_cumsum gives intra-tile write offsets
    #   - tl.atomic_add(counts0[0], tile_sum) gives base offset
    #   - tl.store(ids0[0, base+intra]) writes the passing item ids
    # Result: ids0.shape = [1, 10^6]; counts0 = [50_000]  (~5% of N)
    # Valid ids live at ids0[0, 0:50_000]; ids0[0, 50_000:] is uninitialized padding.

# Step 2: cascade body — second filter applied via evaluate_subset
b, p = ids0.shape           # b=1, p=10^6
valid = arange(p)[None] < counts0[:, None]    # [1, 10^6], first 50k positions True
safe_ids = where(valid, ids0, 0)              # invalid positions read item 0 safely
sub_mask = f1_exact.evaluate_subset(q_exact_attrs, safe_ids)  # [1, 10^6]
    # evaluate_subset for ExactAttributeFilter is the default override (since the
    # Triton kernel doesn't have a subset variant): gather(evaluate_mask(q), safe_ids).
    # Walks the full [1, N] mask, then indirects by safe_ids — still O(N).
sub_mask = sub_mask & valid    # zero out positions past counts0
new_counts = sub_mask.sum(dim=1)     # say [49_500] — Bloom had ~1% false positives
new_p = int(new_counts.max().item()) # HOST SYNC — see note below

# Step 3: re-compact: shift surviving ids to the prefix
sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
ids1 = ids0.gather(1, sorted_idx)    # [1, 49_500] of surviving global ids
counts1 = new_counts                  # [49_500]

# Step 4: feed (ids1, counts1) to a sparse-path KNN module
topk_ids, topk_scores = prefilter_knn(query, candidate_ids=ids1, counts=counts1)
```

Two non-obvious points illustrated by this example:
- **The cascade contains exactly one host sync** (`int(new_counts.max().item())`). All other operations are pure tensor flow. The sync is necessary because the second filter's output buffer must be sized concretely; the alternative — keeping `new_p` symbolic and re-allocating worst-case `[B, N]` per cascade step — would defeat the point. Source: [layers/filters/__init__.py:55](../../retrieve/src/retrieve/layers/filters/__init__.py#L55).
- **The cascade's cost asymmetry**: in this example the Bloom filter does $O(N)$ work in one fused Triton kernel (~one HBM streaming pass), while the exact filter's `evaluate_subset` falls back to `gather(evaluate_mask)` which is *also* $O(N)$ because `ExactAttributeFilter` does not currently ship a fused `evaluate_subset` kernel. The architectural promise of the cascade — "subsequent filters touch only the survivors" — is realized only when the downstream filters override `evaluate_subset` with a kernel that scales as $O(P)$, not $O(N)$. This is a known design gap; the docstring at [interfaces.py:22-25](../../retrieve/src/retrieve/interfaces.py#L22-L25) notes "override when a per-row check is much cheaper than full evaluation (e.g. P ≪ N)". Writer should mention this honestly.

### Writer's notes

- The `FilterModule` ABC is a clean abstraction: two abstract methods, two with sensible defaults, one alias. Worth a short paragraph emphasizing that the ABC is the only cross-tier API shared between `layers/` and `kernels/` aside from `Backend`.
- The Bloom keying invariant is non-obvious and load-bearing for correctness. Recommend a separate paragraph in the prose ("Bloom signature construction") with the FPR formula and the cross-clause collision argument.
- The "no automatic cost model" point should be phrased honestly: the package exposes the choice; the harness picks per scenario. Avoid claiming a runtime heuristic the code does not have.
- Worked cascade example now included (§4.7, "Worked example: two-filter sparse cascade") — resolves the prior TODO. The example also exposes a known design gap: `evaluate_subset` for `ExactAttributeFilter` falls back to the default `gather(evaluate_mask)`, which is still $O(N)$ rather than $O(P)$. Writer should mention this as a current limitation and reference [interfaces.py:22-25](../../retrieve/src/retrieve/interfaces.py#L22-L25) which explicitly invites kernel-level subset implementations.

---

## §4.8 Квантизационные примитивы (Quantization primitives)

Three primitives in [retrieve/src/retrieve/layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py). Formal mathematical definitions live in Chapter 2 (§2.5 per the contract); this section is implementation-only and shows the bodies plus design comments.

### `quantize_int8` — per-row symmetric INT8

Source: [retrieve/src/retrieve/layers/utils/quantize.py:25-33](../../retrieve/src/retrieve/layers/utils/quantize.py#L25-L33).

```python
def quantize_int8(embs: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-item INT8 quantization.

    embs ≈ codes.float() * scales.unsqueeze(1).
    """
    abs_max = embs.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scales
```

Per-row symmetric quantization. For each row, compute the absolute maximum, derive a per-row scale `amax / 127`, quantize via division and rounding, clamp to the int8 range. Reconstruction is `codes.float() * scales[:, None]`. Used at query time in the co-designed retriever (each query row has its own scale; the global scale applies only to the item-side index).

The `.clamp(min=1e-8)` guards against all-zero rows (a row of zeros has `amax=0` which would divide-by-zero); the floor `1e-8` is small enough that the resulting all-zero `codes` row is rounded to exactly zero with no effect.

### `quantize_int8_global` — per-tensor symmetric INT8 (not in `__all__`)

Source: [retrieve/src/retrieve/layers/utils/quantize.py:36-55](../../retrieve/src/retrieve/layers/utils/quantize.py#L36-L55).

```python
def quantize_int8_global(embs: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor INT8 quantization — one scalar scale for the index.

    The SilverTorch paper's int8 ANN scheme (§4.2): "compute global min/max
    values across all embeddings, scale them to [-128, 127], and assign
    integer representations accordingly." A single scalar lets the kernel
    apply one scalar multiply per item in the dequant epilogue, vs a
    per-item gather under per-row scales — at the cost of coarser
    reconstruction (an outlier row stretches the global scale, narrowing
    the int8 lattice for every other row).

    Returns (codes[N, D] int8, scale: float) with reconstruction
    embs ≈ codes.float() * scale. For higher quality at the cost of an
    extra [N] fp32 buffer + a per-item gather in the kernel, use
    quantize_int8 and store its per-row scales instead.
    """
    abs_max = embs.abs().amax().clamp(min=1e-8)
    scale = float((abs_max / 127.0).item())
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scale
```

Per-tensor global symmetric quantization. Returns a Python `float` scale (deliberately, to avoid host syncs in the forward pass — see §4.5). Used by `SilverTorch.register_index` for the item index. The trade-off vs `quantize_int8` (per-row) is documented in the docstring: global scale simplifies the kernel (one scalar multiply per dot product, no per-item scale gather) but loses precision when the embedding distribution has outlier rows.

Note: not re-exported through `retrieve/__init__.py` (see §4.2). Users who want per-tensor quantization access it via `from retrieve.layers.utils.quantize import quantize_int8_global`.

### `quantize_oporp_1bit` — Sign-OPORP 1-bit

Source: [retrieve/src/retrieve/layers/utils/quantize.py:81-108](../../retrieve/src/retrieve/layers/utils/quantize.py#L81-L108).

```python
def quantize_oporp_1bit(
    embs: Tensor,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sign-OPORP 1-bit quantization.

    OPORP = One Permutation + One Random projection (Li et al., 2019). For each
    item x, the projection is (signs * x)[perm]; the result is then
    sign-quantized and packed into 64-bit words. Cheap: O(D) state, not O(D²).

    Returns:
        bits: [N, W] int64 packed sign bits, W = D // 64.
        signs: [D] int8 in {-1, +1} — Rademacher sign vector.
        perm: [D] int64 — permutation applied after the sign flip.

    Apply the same (signs, perm) to a query via project_oporp_1bit_query
    to land in the same bit space. Similarity is then
    D - 2 * popcount(query_bits ^ item_bits).
    """
    if embs.dim() != 2:
        raise ValueError(f"expected 2-D [N, D] embeddings, got shape {tuple(embs.shape)}")
    n, d = embs.shape
    if d % 64 != 0:
        raise ValueError(f"D must be a multiple of 64 for 1-bit packing, got {d}")
    signs, perm = _build_oporp(d, seed, embs.device)
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    bits = _pack_signs_to_int64(proj)
    return bits, signs, perm
```

Sign-OPORP from [CITE: Li & Li 2023 — OPORP: One Permutation + One Random Projection — arXiv:2302.03505]. The construction has `O(D)` state (one Rademacher vector of length D and one permutation of length D) rather than the `O(D²)` of a full random Gaussian projection. The cost is paid once at index-build time.

**Cite reconciliation note (resolved):** The in-code docstring says "Li et al., 2019" but the correct arXiv reference is the 2023 preprint, as confirmed by the LinR paper notes at [articles/linr.md:368-369](../../articles/linr.md#L368-L369): "*OPORP: One permutation+ one random projection. arXiv preprint arXiv:2302.03505 (2023).*" The 2019 figure in the docstring appears to refer to an earlier conference precursor of the same construction by Li and co-authors; for the thesis bibliography, use the 2023 arXiv preprint that LinR itself cites. Writer: cite as [CITE: Li & Li 2023, arXiv:2302.03505] and either correct the in-code docstring in a follow-up PR or note the discrepancy as a minor source-code typo.

### Sign-OPORP toy example (8-D illustration)

A worked example for intuition. Take $D = 8$, $W = D/64 = 0$ — wait, the code requires $D \% 64 = 0$, so the minimum example is $D = 64$. Below we walk through a $D = 8$ *conceptual* example by ignoring the int64 packing step (the packing is an orthogonal bit-packing detail; the projection arithmetic is what carries the semantic content).

Take embedding $x = (0.3, -0.7, 0.1, 0.9, -0.2, 0.5, -0.4, 0.6) \in \mathbb{R}^{8}$ and seed-derived $(\mathrm{signs}, \mathrm{perm})$ where:
- $\mathrm{signs} = (+1, -1, -1, +1, +1, -1, +1, -1)$ (Rademacher vector from `torch.randint(0, 2) * 2 - 1`),
- $\mathrm{perm} = (4, 0, 6, 2, 7, 1, 5, 3)$ (permutation from `torch.randperm`).

Step 1 — element-wise sign flip $x \odot \mathrm{signs}$:
$$x' = (+0.3, +0.7, -0.1, +0.9, -0.2, -0.5, -0.4, -0.6).$$

Step 2 — permutation $x''_i = x'_{\mathrm{perm}_i}$ (i.e. `index_select(perm)`):
$$x'' = (x'_4, x'_0, x'_6, x'_2, x'_7, x'_1, x'_5, x'_3) = (-0.2, +0.3, -0.4, -0.1, -0.6, +0.7, -0.5, +0.9).$$

Step 3 — sign quantization $b_i = [x''_i > 0]$:
$$b = (0, 1, 0, 0, 0, 1, 0, 1).$$

Step 4 — pack to a 64-bit word (in the real code; here, conceptually, $b$ is the 8-bit packed signature). Bit $i$ of word 0 is $b_i$, so the packed value is $0 \cdot 2^0 + 1 \cdot 2^1 + 0 \cdot 2^2 + 0 \cdot 2^3 + 0 \cdot 2^4 + 1 \cdot 2^5 + 0 \cdot 2^6 + 1 \cdot 2^7 = 162$ (binary `10100010`).

For a second vector $y$, the *same* $(\mathrm{signs}, \mathrm{perm})$ are applied (this is the seed-determinism property — it is why `OneBitKNN.__init__` accepts a `seed` parameter and propagates it through `register_index` to every subsequent `forward` call). The Hamming-XOR similarity is then $\mathrm{score} = D - 2 \cdot \mathrm{popcount}(b_x \oplus b_y)$: two perfectly-matching signatures have $\mathrm{popcount} = 0$ and $\mathrm{score} = D$; two perfectly anti-correlated signatures have $\mathrm{popcount} = D$ and $\mathrm{score} = -D$. The transformation is the surrogate for the cosine similarity of the original real-valued embeddings; the variance of the surrogate decreases as $1/D$ (the OPORP guarantee).

The real-code minimum is $D = 64$ (one int64 word). At thesis-relevant dimensions ($D \in \{64, 128, 256\}$) the bit budget is $W \in \{1, 2, 4\}$ words per item; storage cost is $8W$ bytes per item (vs $2D$ bytes for fp16 — a 16× compression for $D = 64$).

### Helper: `_build_oporp` — seed-driven Rademacher + permutation

Source: [retrieve/src/retrieve/layers/utils/quantize.py:58-63](../../retrieve/src/retrieve/layers/utils/quantize.py#L58-L63).

```python
def _build_oporp(d: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    signs = torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.int8) * 2 - 1
    perm = torch.randperm(d, generator=g, device=device)
    return signs, perm
```

The seed-determinism is load-bearing for live-index correctness (queries built at one timestamp must use the same `(signs, perm)` as items built at any other timestamp; the `seed` parameter on `OneBitKNN.__init__` propagates here). The signs are in {-1, +1} (randint(0,1) scaled to ±1) — bit-flipping under Rademacher gives the OPORP randomness; the subsequent random permutation gives the projection.

### Helper: `_pack_signs_to_int64`

Source: [retrieve/src/retrieve/layers/utils/quantize.py:66-78](../../retrieve/src/retrieve/layers/utils/quantize.py#L66-L78).

```python
def _pack_signs_to_int64(values: Tensor) -> Tensor:
    """Pack sign-quantized [..., D] floats into [..., W] int64 bit words.

    Bit b of word w is set iff values[..., 64*w + b] > 0.
    """
    *prefix, d = values.shape
    if d % 64 != 0:
        raise ValueError(f"projected dim must be a multiple of 64, got {d}")
    w = d // 64
    bits = (values > 0).to(torch.int64)
    bits = bits.reshape(*prefix, w, 64)
    shifts = torch.arange(64, device=values.device, dtype=torch.int64)
    return (bits << shifts).sum(dim=-1)
```

Pure tensor-flow bit packing: extract sign, reshape to W groups of 64, shift each bit by its position, sum to form 64-bit words. No Python loop over W or over D — stays symbolic under `torch.compile(dynamic=True)`. This is the partner of the SWAR `_popcount_int64` defined inside `oporp_1bit_match_topk` (§4.3.1).

### Companion: `project_oporp_1bit_query`

Source: [retrieve/src/retrieve/layers/utils/quantize.py:111-128](../../retrieve/src/retrieve/layers/utils/quantize.py#L111-L128).

```python
def project_oporp_1bit_query(
    query: Tensor,
    signs: Tensor,
    perm: Tensor,
) -> Tensor:
    """Apply the same Sign-OPORP projection to a query batch.

    Returns [B, W] int64 packed sign bits.

    Pure tensor flow — multiply, index_select, > 0, reshape, shifted
    sum. Called from inside OneBitKNN.forward; the eval-side algo
    wrapper compiles its forward with mode="reduce-overhead", so
    this work is captured into the outer cudagraph.
    """
    if query.dim() != 2:
        raise ValueError(f"expected 2-D [B, D] query, got shape {tuple(query.shape)}")
    proj = (query * signs.to(query.dtype)).index_select(1, perm)
    return _pack_signs_to_int64(proj)
```

The query projection uses the same `(signs, perm)` registered at index time, guaranteeing that queries and items land in the same bit space. The function is pure tensor-flow (no Python loop) so that `torch.compile(dynamic=True, mode="reduce-overhead")` can fuse it into the outer cudagraph_trees graph — see §4.5.

### Cross-references

- Formal definitions of INT8 per-tensor / Sign-OPORP and their error properties belong in Ch.2 §2.5 (per the chapter contract). This section is implementation-only.
- The `dp4a` hardware basis for INT8 dot products on Ampere+ is documented in Ch.1 §1.7 — cross-reference there rather than re-citing here.

### Writer's notes

- The two INT8 variants (per-row, per-tensor) are presented honestly as a trade-off. The package picks per-tensor for the index (one global scale) and per-row for queries (one scale per query). The trade-off paragraph in the `quantize_int8_global` docstring is ready to lift.
- OPORP cite mismatch is now reconciled in the notes: the bibliography entry is [CITE: Li & Li 2023, arXiv:2302.03505], confirmed against `articles/linr.md:368-369`. The in-code "Li et al., 2019" docstring is a minor source-code typo; writer should cite the 2023 arXiv preprint and optionally open a follow-up PR to correct the docstring.
- Sign-OPORP toy example now included (§4.8, "Sign-OPORP toy example") — resolves the prior TODO. The 8-D walk-through is conceptual (the real code requires `D % 64 == 0`); writer may keep the 8-D walk-through for intuition or replace with a 64-D variant if the supervisor prefers a runnable example.

---

## §4.9 Тестирование (Testing)

### Test layout

Source: [retrieve/tests/](../../retrieve/tests/).

```
retrieve/tests/
├── conftest.py                          # CUDA-only collection hook
├── correctness/                         # module-level semantics vs torch baselines
│   ├── test_bloom_filter.py
│   ├── test_combine_filters.py
│   ├── test_compact.py
│   ├── test_filters.py
│   ├── test_linr.py
│   ├── test_quantize.py
│   ├── test_retrieval_utils.py
│   └── test_silvertorch.py
├── parity/                              # Triton kernel vs torch reference
│   ├── conftest.py                      # assert_topk_matches helper
│   ├── test_bloom_compact.py
│   ├── test_bloom_match.py
│   ├── test_clause_compact.py
│   ├── test_clause_mask.py
│   ├── test_codesigned_probe_score.py
│   ├── test_codesigned_probe_score_exact.py
│   ├── test_fused_masked_knn_topk.py
│   └── test_oporp_1bit_match_topk.py
└── compile/                             # torch.compile graph-break regression
    └── test_silvertorch_compile.py
```

Totals: 17 test files / 162 test functions. Counts verified via `grep -c '^def test_\|^    def test_'` per file (parametrization expands further at collection time; raw `def test_*` count is what is reported here).

| Category | Files | Test funcs | Purpose |
|---|---|---|---|
| Correctness | 8 | 120 | Module-level semantics vs torch oracle (cross-backend) |
| Parity | 8 | 40 | Per-kernel Triton vs torch reference equivalence |
| Compile | 1 | 2 | torch.compile graph-break regression |
| **Total** | **17** | **162** | — |

Per-file test-function counts (sorted by category, then by size):

| File | Category | `def test_*` count |
|---|---|---:|
| `correctness/test_silvertorch.py` | correctness | 35 |
| `correctness/test_linr.py` | correctness | 25 |
| `correctness/test_bloom_filter.py` | correctness | 15 |
| `correctness/test_quantize.py` | correctness | 14 |
| `correctness/test_combine_filters.py` | correctness | 10 |
| `correctness/test_filters.py` | correctness | 8 |
| `correctness/test_retrieval_utils.py` | correctness | 7 |
| `correctness/test_compact.py` | correctness | 6 |
| `parity/test_clause_mask.py` | parity | 8 |
| `parity/test_bloom_compact.py` | parity | 6 |
| `parity/test_clause_compact.py` | parity | 6 |
| `parity/test_fused_masked_knn_topk.py` | parity | 6 |
| `parity/test_oporp_1bit_match_topk.py` | parity | 6 |
| `parity/test_codesigned_probe_score_exact.py` | parity | 4 |
| `parity/test_codesigned_probe_score.py` | parity | 3 |
| `parity/test_bloom_match.py` | parity | 1 |
| `compile/test_silvertorch_compile.py` | compile | 2 |

### Testing strategy

Per [docs/system/testing.md](../system/testing.md):

- **Correctness-only.** Every test asserts a semantic invariant, numerical equivalence, or edge-case contract. Latency and recall sweeps live in `evaluation/`; they never gate CI.
- **GPU-only.** The root [conftest.py](../../retrieve/tests/conftest.py) installs a collection hook that skips every test when CUDA is unavailable. There are no CPU-side stub tests.
- **Vanilla pytest.** No `hypothesis`, no `unittest`. No mocking — tests use real CUDA tensors and real kernels. Triton kernel paths run on every CUDA-equipped test.
- **Determinism.** Every random tensor is built with an explicit `torch.Generator(device="cuda").manual_seed(...)`. No `torch.manual_seed` at global scope.

### Parity invariant — `assert_topk_matches`

The cross-kernel parity tests use a shared helper that tolerates score-tie permutations between backends. Source: [retrieve/tests/parity/conftest.py](../../retrieve/tests/parity/conftest.py).

```python
def assert_topk_matches(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Assert two top-K implementations agree on finite-id sets and sorted scores.

    Tie-breaking on the id permutation can differ between backends, so we compare
    *sets* of finite-score ids per row and the *sorted* descending scores
    (with -inf replaced by 0 so ``allclose`` still works on padded rows).
    """
    b, k = out_ids.shape
    for bi in range(b):
        out_pairs = [
            (out_ids[bi, j].item(), out_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(out_scores[bi, j])
        ]
        ref_pairs = [
            (ref_ids[bi, j].item(), ref_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(ref_scores[bi, j])
        ]
        out_set = {p[0] for p in out_pairs}
        ref_set = {p[0] for p in ref_pairs}
        if out_set == ref_set:
            continue
        # Tensor-core matmul (`tl.dot`) and torch `@` differ in accumulator
        # order — score-tied items can swap at the K-th boundary. Allow that
        # provided each side's unique ids lie within `atol` of its own min.
        out_min = min(s for _, s in out_pairs) if out_pairs else float("-inf")
        ref_min = min(s for _, s in ref_pairs) if ref_pairs else float("-inf")
        for i in ref_set - out_set:
            s = next(sc for idx, sc in ref_pairs if idx == i)
            assert s <= ref_min + atol + rtol * abs(ref_min), (
                f"row {bi}: ref-only id {i} score={s:.6f} not at boundary {ref_min:.6f}"
            )
        for i in out_set - ref_set:
            s = next(sc for idx, sc in out_pairs if idx == i)
            assert s <= out_min + atol + rtol * abs(out_min), (
                f"row {bi}: out-only id {i} score={s:.6f} not at boundary {out_min:.6f}"
            )

    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    out_finite = torch.where(torch.isfinite(out_sorted), out_sorted, torch.zeros_like(out_sorted))
    ref_finite = torch.where(torch.isfinite(ref_sorted), ref_sorted, torch.zeros_like(ref_sorted))
    assert torch.allclose(out_finite, ref_finite, atol=atol, rtol=rtol)
```

Tolerance: `atol=1e-3, rtol=1e-3` for fp32 dot-product paths. For the 1-bit Hamming path, parity is *bit-exact* — the torch reference uses the same SWAR `_popcount_int64` as the Triton kernel, so scores must agree exactly (no tolerance needed). Documented at [docs/system/kernels.md](../system/kernels.md) under the OPORP section.

### Cross-backend correctness test pattern

Source: [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py).

```python
BACKENDS = ["torch", "triton"]

@pytest.mark.parametrize("backend", BACKENDS)
class TestPostfilterKNN:
    def test_no_mask_returns_topk(self, data, backend):
        m = PostfilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_external_mask(self, data, backend, pass_rate):
        m = PostfilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()
```

Every LinR test class is parametrized over `backend ∈ {"torch", "triton"}`. The same test body verifies (a) shape correctness, (b) descending-score ordering, (c) mask compliance (every returned id satisfies the input mask) for both backends.

### Bit-exact parity test pattern — OPORP

Source: [retrieve/tests/parity/test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py).

```python
def _ref_full(query_bits: torch.Tensor, item_bits: torch.Tensor, k: int):
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    scores = (d_total - 2 * hamming).to(torch.float32)
    topk_scores, topk_ids = torch.topk(scores, k, dim=1)
    return topk_ids.to(torch.long), topk_scores

@pytest.mark.parametrize("n,d,k", [(1024, 128, 8), (8192, 128, 32), (4096, 256, 16)])
@pytest.mark.parametrize("b", [1, 16])
def test_oporp_1bit_full_matches_torch(n, d, k, b):
    embs = make_index(n, d)
    query = make_query(b, d)
    item_bits, signs, perm = quantize_oporp_1bit(embs, seed=0)
    query_bits = project_oporp_1bit_query(query, signs, perm)

    out_ids, out_scores = oporp_1bit_match_topk_full(query_bits, item_bits, k)
    ref_ids, ref_scores = _ref_full(query_bits, item_bits, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)
```

The torch reference (`_ref_full`) uses the same SWAR `popcount_int64` as the Triton kernel, so the parity assertion is bit-exact (the `assert_topk_matches` helper tolerates score-tie permutations, but the underlying score values match exactly).

### Compile parity test pattern

Source: [retrieve/tests/compile/test_silvertorch_compile.py:44-82](../../retrieve/tests/compile/test_silvertorch_compile.py#L44-L82).

```python
@pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
def test_compiled_forward_matches_eager(filter_mode):
    eager = _build(filter_mode)
    compiled = torch.compile(eager.forward, dynamic=True, mode="reduce-overhead")
    # ... build inputs ...
    eager_ids, eager_scores = eager.forward(query, q_attrs)
    compiled_ids, compiled_scores = compiled(query, q_attrs)
    assert_topk_matches(compiled_ids, compiled_scores, eager_ids, eager_scores)

@pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
def test_no_graph_breaks_on_forward(filter_mode):
    eager = _build(filter_mode)
    # ... build inputs ...
    explanation = torch._dynamo.explain(eager.forward)(query, q_attrs)
    assert explanation.graph_break_count == 0
```

Two tests per filter mode: (a) numerical equivalence between eager and `torch.compile(dynamic=True, mode="reduce-overhead")`; (b) zero graph breaks at `torch._dynamo.explain` time.

### Writer's notes

- The three-tier test layout (correctness / parity / compile) is a clean prose hook — explain it once at the top of §4.9, then describe each tier with one paragraph.
- The `assert_topk_matches` helper deserves a verbatim excerpt in the prose: it formalizes what "backend parity" means in the presence of fp32 accumulator-order non-determinism (which is genuine, not a bug).
- Exact test counts (162 total across 17 files; per-file table) are now in §4.9; the volumetric table in §4.10 carries the same numbers. Counts produced via `grep -c '^def test_\|^    def test_'` per file at the time of writing — writer should re-run before submission and pin to a commit SHA.
- [TODO: at thesis-submission time, re-run `cd retrieve && grep -rc '^def test_' tests/ | awk -F: '{s+=$2} END {print s}'` and pin the resulting count + commit SHA in the prose. Current count assumes the working tree at the time these notes were generated.]

---

## §4.10 Объёмные характеристики (Code-volume characteristics)

### Volumetric table

| Module | LOC | Classes | Functions (module-level) | Test files | Test functions |
|---|---:|---:|---:|---:|---:|
| `layers/linr/` | 492 | 4 | 2 | 1 (`test_linr.py`) | 25 |
| `layers/silvertorch/` | 416 | 1 | 1 | 1 (`test_silvertorch.py`) | 35 |
| `layers/filters/` | 437 | 2 | 6 | 2 (`test_bloom_filter.py`, `test_filters.py`) | 23 |
| `layers/utils/` | 270 | 2 | 9 | 4 (`test_compact.py`, `test_quantize.py`, `test_retrieval_utils.py`, `test_combine_filters.py`) | 37 |
| `kernels/linr/` | 684 | 2 | 10 | 2 (parity: `test_fused_masked_knn_topk.py`, `test_oporp_1bit_match_topk.py`) | 12 |
| `kernels/silvertorch/` | 857 | 2 | 10 | 2 (parity: `test_codesigned_probe_score.py`, `test_codesigned_probe_score_exact.py`) | 7 |
| `kernels/filters/` | 692 | 3 | 9 | 3 (parity: `test_clause_mask.py`, `test_clause_compact.py`, `test_bloom_compact.py`) | 20 |
| `interfaces.py` | 57 | 1 | 0 | — | — |
| `tune.py` | 712 | 0 | — | — | — |
| `__init__.py` | 38 | 0 | — | — | — |
| **Total** | **4 696** | **17** | **47** | **17 (incl. `test_silvertorch_compile.py` → +2)** | **161** (+2 compile = **162**) |

Notes on the test column: tests are organized by purpose (correctness / parity / compile), not strictly by source module, so the mapping from a test file to one source directory is approximate — `test_linr.py` exercises both `layers/linr/` and the kernels they invoke. The "test functions" column above lists the raw `def test_*` count in the most-relevant test file; parametrization (`@pytest.mark.parametrize` over `backend ∈ {"torch", "triton"}`, batch sizes, pass rates, etc.) expands each into many test cases at collection time.

### Aggregate counts

- **Source LOC**: 4 696 (production); 3 303 (tests).
- **Triton kernels**: 10 `@triton.jit` functions across 8 files, decomposing as **8 main device kernels + 2 inline device helpers** (`_popcount_int64` in [kernels/linr/oporp_1bit_match_topk.py:42](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L42) and `_or_combine` in [kernels/silvertorch/codesigned_probe_score.py:14](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L14)). The two helpers are scalar / reduction utilities invoked from the main kernels via Triton's static-inlining; they share the same JIT compilation but are not standalone retrieval primitives.
- **Public API symbols** (`__all__`): 17 (see §4.2).
- **Filter primitives** (`FilterModule` subclasses): 2.
- **KNN modules** (LinR family + co-designed retriever): 6.
- **Decorator inventory**: 9 `@torch.library.triton_op` wrappers across 8 files (`codesigned_probe_score.py` carries two — the plain and Bloom-variant entry points share one `@triton.jit` body). Zero `@torch.library.custom_op` usages — see §4.5.

### Method

Counts were produced via:
- LOC: `find retrieve/src/retrieve -type f -name '*.py' | xargs wc -l` then aggregated by directory.
- Classes: `grep -c '^class ' <file>` aggregated by directory.
- Module-level functions: `grep -c '^def ' <file>` aggregated by directory (private helpers prefixed with `_` are included).
- Test functions: `grep -rc '^def test_' retrieve/tests/`.
- Triton kernels: `grep -rc '@triton.jit' retrieve/src/retrieve/`.

### Writer's notes

- The HSE-style volumetric table is a common chapter-closing artifact. The version above mirrors what the contract asks for; the writer may extend with cyclomatic complexity or dependency count if the supervisor expects it.
- The asymmetry between `kernels/silvertorch/` (857 LOC) and `layers/silvertorch/` (416 LOC) reflects design balance: the co-designed retriever does most of its work inside two large fused kernels (~340 LOC for `codesigned_probe_score`, ~340 LOC for `codesigned_probe_score_exact`), with a relatively thin `layers/` orchestration layer above them.
- `tune.py` is the largest single file (712 LOC). It is not part of the runtime API; the writer should explain this in the prose so the volumetric breakdown isn't misleading.
- Exact test-function counts now in the table (replacing the prior "20+/15+" approximations). The numbers reflect raw `def test_*` lines; pytest parametrization expands further at collection time. To get the *collected* test-case count (which is typically 3–10× larger), run `cd retrieve && pytest --collect-only -q | tail -2` against a CUDA-equipped machine.
- [TODO: clarify with author — does the supervisor want the *collected* test-case count instead of the raw `def test_*` count? If yes, this requires a GPU run (the pytest `conftest.py` skips collection without CUDA); to be done at submission time on a GPU machine.]

### Planned schema extensions (cross-reference to Ch.5)

The `retrieve/` library exposes only the algorithm-level forward kernels and modules tabulated above; **the benchmark-measurement schema** (latency / memory / recall fields emitted into the result JSONs) is the responsibility of the evaluation harness, documented in [06-eval-protocol.md](06-eval-protocol.md) §5.7. The library is intentionally unaware of any measurement convention.

The Chapter 6 plot catalog ([docs/thesis/results-data/recipes/plot_catalog.md](../../docs/thesis/results-data/recipes/plot_catalog.md)) requires additional schema fields that do not yet exist (`throughput_qps`, `mean_ms`, `p99_ms`, `build_time_s`, `topk_ids_jaccard_vs_torch`, per-Triton-kernel timing from Nsight Compute, and a per-query latency vector). All seven extensions belong inside `evaluation/retrieval/bench_tools.py` — none requires changes to `retrieve/src/retrieve/`. The full table with rationale per field is in [06-eval-protocol.md](06-eval-protocol.md) §5.7 "Planned schema extensions"; this Ch.4 reference is here only to make the boundary explicit: **library code is stable; missing data is a harness gap, not an algorithm gap**.

The one library-side prerequisite is that kernels and wrappers must return top-k INDEX TENSORS (not just embeddings) so the harness can compute Jaccard agreement between Triton and torch backends — this is already true for every wrapper documented in §4.2 (every `forward()` returns `(scores, indices)` or equivalent).

### Per-layer micro-benchmarks (Триточные микробенчмарки)

Per-layer micro-benchmark data — Triton-vs-torch backend speedup and compile-vs-eager mode speedup, both at the level of a single layer's `forward()` — lives under `evaluation/results/` alongside the chapter's main evaluation outputs. The Triton↔torch parity-and-speedup pivot is materialized at [docs/thesis/results-data/parity_and_speedup.csv](results-data/parity_and_speedup.csv) (2 823 rows; one row per cell × backend), and the per-impl JSON cells under `evaluation/results/{arxiv,goodreads,yambda}/d*-quality.json` and `evaluation/results/{arxiv,goodreads}/d*-filter.json` contain the raw `median_ms` measurements that the speedup pivot is computed from. The compile-vs-eager qualitative table is in §4.5 (sourced from [docs/system/kernels.md:574-595](../system/kernels.md#L574-L595)); a fuller per-module compile-vs-eager benchmark JSON is listed as a planned schema extension above. **No numbers are quoted in this section** — the engineering-validation reporting (backend parity scatter F1, speedup distribution F2, parity table G4, speedup matrix G5) is the responsibility of Ch.6 §6.7, which the writer should forward-reference from here.

---

## §4.11 Совместимость с произвольными энкодерами (Encoder-agnostic interop)

This section provides the concrete evidence for property (b) of the framework claim from the preamble: the bundled retrieval layers consume `(queries, items)` embedding tensors regardless of how those tensors were produced. The evaluation in Ch.6 demonstrates this by feeding the same layers from two structurally different encoder families and showing that the layer code, the kernels, and the public API are identical across both demos.

### Contract restated

Every KNN layer in §4.2 takes two inputs:

- `items: Tensor` of shape `[N, D]` (the catalog, used at index-build / `register_index` time);
- `queries: Tensor` of shape `[B, D]` (the per-batch query embeddings).

Neither input carries metadata about the producing encoder; both are treated as opaque dense float tensors. The layers do not import any model code, do not require any encoder-side hooks, and do not assume the encoder is trained jointly with the index. This is the contract that makes the framework encoder-agnostic.

### Demo (a) — SASRec / gSASRec sequence encoder (Goodreads + Yambda)

For Goodreads and Yambda, query embeddings are produced by a SASRec / gSASRec sequence model trained on user interaction histories. The model is defined in [`evaluation/training/model.py`](../../evaluation/training/model.py), with the training loop in [`evaluation/training/train_sasrec.py`](../../evaluation/training/train_sasrec.py) and the gBCE loss in [`evaluation/training/losses.py`](../../evaluation/training/losses.py); the trained checkpoints (`gsasrec-d{64,128,256}-drop0.5-id/best_model.pt`) are loaded by the evaluation harness to produce per-user query embeddings. These embeddings are then fed unchanged into each of the per-algorithm wrappers in [`evaluation/retrieval/algos/`](../../evaluation/retrieval/algos/) — `linr_v1.py` (`PostfilterKNN`), `linr_v2.py` (`PrefilterKNN`), `linr_v3.py` (`OneBitKNN` + `PrefilterKNN`), `linr_v4.py` (`PostfilterKNNInt8`), `silvertorch.py` (`SilverTorch`), and `torch_knn.py` (the reference `FullScanKNN`). Item embeddings are read from the same SASRec checkpoint's item-embedding table.

Cross-link: SASRec/gSASRec architecture, the gBCE loss, the training loop, and the per-dataset checkpoint convention are documented in [docs/thesis/04-datasets-notes.md](04-datasets-notes.md) §7 (SASRec / gSASRec query-model training); the Goodreads and Yambda dataset pipelines that exercise these checkpoints are §2 and §4 of the same file. Citation anchors are `[CITE: Kang & McAuley 2018 — UCSD — kept]` and `[CITE: Petrov & Macdonald 2023 RecSys — University of Glasgow — kept]`.

### Demo (b) — Nomic-Embed text encoder (arXiv)

For arXiv, the same layers consume embeddings produced by `nomic-embed-text-v1.5`, a pretrained text encoder used **out-of-the-box, not retrained** for this thesis. The embedding pipeline is documented in [`evaluation/datasets/arxiv.py`](../../evaluation/datasets/arxiv.py): per-item title + abstract → Nomic-prefixed input → `nomic-embed-text-v1.5` Matryoshka projection at `d ∈ {64, 128, 256}` → stored as the item-embedding tensor. Query embeddings come from the same encoder. The exact same per-algorithm wrappers under [`evaluation/retrieval/algos/`](../../evaluation/retrieval/algos/) — byte-identical Python files — consume these Nomic embeddings without any code change.

Cross-link: full Nomic-Embed pipeline (model, prefixing convention, Matryoshka dimensions, audit of authorship) is in [docs/thesis/04-datasets-notes.md](04-datasets-notes.md) §3.2 (arXiv pipeline); citation anchor is `[CITE: Nussbaum et al. 2024 — Nomic AI — kept]`.

### What this demonstrates

The same `KNN` / `LinR V1 V2 V3 V4` / `IVF+INT8+Bloom` layers — and the same Triton kernels underneath them — are used unchanged across two structurally different encoder families: a trained-from-scratch sequence model whose embeddings live inside its own checkpoint, and a frozen pretrained text encoder downloaded from a public registry. That is the framework's encoder-agnostic claim concretely realized, and it is the empirical demonstration that goal 1 of the thesis cites. Concretely: the algorithm wrappers in [`evaluation/retrieval/algos/`](../../evaluation/retrieval/algos/) are dataset-agnostic — they take `(queries, items)` and a backend choice; the per-dataset adapters in [`evaluation/datasets/`](../../evaluation/datasets/) handle the encoder-specific embedding production; the framework boundary lives exactly between these two layers.

---

## Visual deliverables (per Ch.4 contract)

### 1. Architecture diagram (description, not generated)

**What it shows**: the three-tier package layout from §4.1.
- **Top tier** (contracts): a thin horizontal box labelled `interfaces.py` containing two pills: `Backend` and `FilterModule`.
- **Middle tier** (layers/): a column of six boxes (one per KNN module + the two filter classes + helpers). Each box labelled with the class name. Arrows from each box up to `interfaces.py` (to indicate "uses `Backend`" or "implements `FilterModule`").
- **Bottom tier** (kernels/): a column of three boxes corresponding to `kernels/linr/`, `kernels/silvertorch/`, `kernels/filters/`. Each box lists the kernels it contains.
- **Cross-tier arrows**: arrows from each `layers/` box down to the `kernels/` boxes it invokes (e.g., `OneBitKNN` → `kernels/linr/oporp_1bit_match_topk`).

Render in TikZ or import an SVG. Data source: §4.1 directory tree and §4.2 public-API table.

### 2. Sequence diagram for the co-designed IVF forward pass

**What it shows**: the three-phase data flow of `SilverTorch.forward` in Triton-backend mode with `filter="bloom"`.
- **Phase 1 (host-side, ~1 line)**: `query` → matmul against `centroids` → `torch.topk(n_probe)` → `probed_list_ids[B, n_probe]`.
- **Phase 2 (kernel input prep, ~3 lines)**: gather assignments from the IVF inverted lists for each probed centroid → flatten to `flat_probed_items[B, P]` (P = max probed-list-size × n_probe, padded with -1).
- **Phase 3 (fused kernel, the load-bearing step)**: invoke `codesigned_probe_score_bloom(query, flat_probed_items, item_codes, query_bits, bloom_sigs, global_scale, k)` → returns `(topk_ids, topk_scores)`.

Each box of the sequence diagram corresponds to a code region in [layers/silvertorch/main.py:222-240](../../retrieve/src/retrieve/layers/silvertorch/main.py#L222-L240) and the kernel it ultimately invokes. Render with `tikz-uml` or `sequenced` package.

### 3. Algorithm box: fused `codesigned_probe_score` kernel

Already rendered in §4.3.2 as a ```pseudocode block with $...$ math variables. Suggested LaTeX environment: `algorithm2e`. The pseudocode covers all three phases (wrapper preprocessing, kernel body, host top-K finalization).

### 4. Volumetric table

Rendered in §4.10. Markdown source ready to be lifted into LaTeX (`booktabs` package, `\toprule` / `\midrule` / `\bottomrule`).

### 5. Compile-speedup table

Now populated in §4.5 as a small qualitative table with three rows (`project_oporp_1bit_query` ~2.5×; `OneBitKNN._score_full` 3×–43×; matmul refs — compile reverted). Sourced from [docs/system/kernels.md:574-595](../system/kernels.md#L574-L595). Per-module compile-vs-eager *benchmark* JSON would require GPU runs and is left as future work for the Ch.6 evaluation harness.

---

## Citation candidates summary

Eight citation candidates needed in the chapter (all kept; no Meta-affiliated work). Each is already established in Ch.1 (literature review) — Ch.4 references them by anchor citation, not re-introduction.

| # | Citation | Affiliation | Decision | Anchor section in Ch.1 |
|---|---|---|---|---|
| 1 | Tillet, Kung, Cox — Triton (MAPL 2019) | Harvard / OpenAI | **KEEP** | §1.8 |
| 2 | Paszke et al. — PyTorch (NeurIPS 2019) | mixed (PyTorch Foundation) | **KEEP** (per Ch.1 working position) | §1.8 |
| 3 | NVIDIA — `dp4a` instruction (CUDA C++ Programming Guide) | NVIDIA | **KEEP** | §1.7 |
| 4 | Borisyuk et al. — LinR (CIKM 2024) | LinkedIn | **KEEP** | §1.5 |
| 5 | Li & Li — OPORP (arXiv:2302.03505) | non-Meta | **KEEP** | §1.7 |
| 6 | Goodwin et al. — BitFunnel (SIGIR 2017) | Microsoft | **KEEP** | §1.6 |
| 7 | Jégou, Douze, Schmid — PQ (TPAMI 2011) | INRIA (pre-Meta) | **KEEP** | §1.3, §1.7 |
| 8 | `@torch.library.triton_op` + `wrap_triton` API | PyTorch / no literature | **NO CITE** — file paths only | — |

**Kept**: 7 literature citations + 1 file-path-only reference. **Dropped**: 0. The chapter does not introduce any new citation candidates that would require an affiliation audit.

---

## Sources consulted

Files read during preparation of these notes (production code, tests, documentation, configuration):

**Repository root and configuration:**
- [pyproject.toml](../../pyproject.toml) — workspace definition.
- [retrieve/pyproject.toml](../../retrieve/pyproject.toml) — package definition, entry point, dependencies.
- [retrieve/README.md](../../retrieve/README.md) — public-facing description, quick example, module table.

**Production code (`retrieve/src/retrieve/`):**
- [__init__.py](../../retrieve/src/retrieve/__init__.py) — public re-exports.
- [interfaces.py](../../retrieve/src/retrieve/interfaces.py) — `Backend`, `FilterModule`.
- [tune.py](../../retrieve/src/retrieve/tune.py) — offline autotune CLI.
- `layers/__init__.py`, `layers/linr/__init__.py`, `layers/silvertorch/__init__.py`, `layers/filters/__init__.py`, `layers/utils/__init__.py`.
- [layers/linr/postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py)
- [layers/linr/postfilter_knn_int8.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py)
- [layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)
- [layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)
- [layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py)
- [layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py)
- [layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py)
- [layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py)
- [layers/utils/kmeans.py](../../retrieve/src/retrieve/layers/utils/kmeans.py)
- [layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py)
- [layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py)
- [kernels/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py)
- [kernels/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py)
- [kernels/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)
- [kernels/silvertorch/codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py)
- [kernels/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py)
- [kernels/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/filters/clause_mask.py)
- [kernels/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/filters/clause_compact.py)
- [kernels/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py)

**Tests (`retrieve/tests/`):**
- [tests/conftest.py](../../retrieve/tests/conftest.py)
- [tests/parity/conftest.py](../../retrieve/tests/parity/conftest.py)
- [tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- [tests/correctness/test_silvertorch.py](../../retrieve/tests/correctness/test_silvertorch.py)
- [tests/correctness/test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py)
- [tests/correctness/test_filters.py](../../retrieve/tests/correctness/test_filters.py)
- [tests/correctness/test_combine_filters.py](../../retrieve/tests/correctness/test_combine_filters.py)
- [tests/correctness/test_compact.py](../../retrieve/tests/correctness/test_compact.py)
- [tests/correctness/test_quantize.py](../../retrieve/tests/correctness/test_quantize.py)
- [tests/correctness/test_retrieval_utils.py](../../retrieve/tests/correctness/test_retrieval_utils.py)
- [tests/parity/test_fused_masked_knn_topk.py](../../retrieve/tests/parity/test_fused_masked_knn_topk.py)
- [tests/parity/test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)
- [tests/parity/test_bloom_match.py](../../retrieve/tests/parity/test_bloom_match.py)
- [tests/parity/test_bloom_compact.py](../../retrieve/tests/parity/test_bloom_compact.py)
- [tests/parity/test_clause_mask.py](../../retrieve/tests/parity/test_clause_mask.py)
- [tests/parity/test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py)
- [tests/parity/test_codesigned_probe_score.py](../../retrieve/tests/parity/test_codesigned_probe_score.py)
- [tests/parity/test_codesigned_probe_score_exact.py](../../retrieve/tests/parity/test_codesigned_probe_score_exact.py)
- [tests/compile/test_silvertorch_compile.py](../../retrieve/tests/compile/test_silvertorch_compile.py)

**System documentation (`docs/system/`):**
- [architecture.md](../system/architecture.md)
- [kernels.md](../system/kernels.md)
- [filtering.md](../system/filtering.md)
- [testing.md](../system/testing.md)

**Thesis-wide context:**
- [docs/thesis/00-thesis-plan.md](00-thesis-plan.md) — chapter contract, citation policy.
- [docs/thesis/01-literature-review.md](01-literature-review.md) — anchor citations for Ch.1 cross-references.

**Evaluation results (checked for §4.5 compile-speedup data; absent):**
- `evaluation/results/{arxiv,goodreads,yambda}/**/*.json` — quality and latency results; no compile-mode breakdown.
- `evaluation/config/` — backend/mode declarations in YAML.
