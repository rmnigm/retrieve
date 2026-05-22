# Concern-separation refactor: `retrieve` module → torch.export-ready (linr)

> **Document audience**: Each phase below is a self-contained brief intended to be executed by a separate SWE agent without access to this conversation's history. Phases are ordered and depend on prior phases for shared conventions; do not run later phases out of order, but each phase's PR is reviewed and merged independently.

> **Scope note (2026-05-19):** silvertorch is no longer in scope. The original plan had **5 phases** covering 7 kernels and 6 layer classes; this version covers the 5 in-scope kernels and 5 layer classes (drop `SilverTorch`). **Phase 3 (`codesigned_probe_score` + `SilverTorch`)** is replaced with a stub; the `build_export.py` scaffold authoring has moved to Phase 4 (`fused_masked_knn_topk` + `PrefilterKNN`). Phase numbers are preserved so cross-doc references stay stable. See [../plans-silvertorch-backup/torch-export-refactor.md](../plans-silvertorch-backup/torch-export-refactor.md) for the original silvertorch content. Phase 2's `bloom_match` content is also stubbed (silvertorch-only kernel); `bloom_compact` + the `_build_query_signatures` export bypass remain in Phase 2 because they serve linr.

> **Stage 1 status (2026-05-22):** the autotune-separation piece of every phase below has **shipped** as part of the main-thread Stage 1 ([00-roadmap.md](00-roadmap.md)). Every in-scope kernel now has a `<Name>Config` dataclass + `DEFAULT_CONFIG` (single curated default per arch, no per-arch REGISTRY — that turned out to be ceremony) tuned offline via `uv run tune-kernels --kernel <name>`. The compact filter kernels also gained a 3D launch grid so they scale to the 15M-row catalog. The Phase 1 / 2 / 4 / 5 sub-steps below that describe REGISTRY+lookup wiring, `_BLOCK_N` constants, or `@triton.autotune` stripping are **superseded** by the convention documented in [../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation); follow that. What remains in scope here is everything *else* the phases call out: `.item()` removal, full-width `(ids, counts)` API, Optional collapse / `mode` flag wiring, `_build_query_signatures_eager` / `_project_oporp_1bit_query_eager` export bypasses, and `build_export.py` scaffolding.

## Context

The [`retrieve`](../../retrieve/src/retrieve/) package ships **5 in-scope Triton kernels** plus **5 layer classes** (`ExactAttributeFilter`, `BloomFilter`, `OneBitKNN`, `PrefilterKNN`, `SimilarityMasking`) and the pure-torch `FullScanKNN`. The kernels and layers still carry a set of patterns that block clean `torch.export.export()` traces and the future `torch.library.triton_op` + `register_fake` + `aoti_compile_and_package` wrap-up.

The previous version of this plan assumed a class layout (`ClauseIndex`, `LiNR_V*`, sibling `_Triton` classes) that no longer exists; the package has been refactored. **Backend selection is already encapsulated**: each layer's `__init__` takes a `backend: Literal["torch", "triton"]` flag (see [interfaces.py:8](../../retrieve/src/retrieve/interfaces.py#L8)), and there is one class per module — no `is_cuda` branches inside `forward` paths. Counts of pending blockers, against the current tree (in-scope only):

- **2 of 3 in-scope host-callable Triton wrappers** still use `@triton.autotune` with `grid=lambda meta: …` ([fused_masked_knn_topk.py:36](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L36), [oporp_1bit_match_topk.py:36](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L36)). Inductor's compile-time autotuner cannot serialize portably.
- **1 host wrapper** branches on `Optional[Tensor]` arguments and rebinds to dummy `1×1` tensors keyed by a `HAS_INDICES` constexpr flag ([oporp_1bit_match_topk.py:107-108](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L107-L108)).
- **Two compact host wrappers** materialize `[B, N]` indices via `torch.full(..., -1)`, then slice on `.item()`: [clause_compact.py:158](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L158), [bloom_compact.py:132](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L132).
- **One layer-forward `.item()` guard** at [one_bit_knn.py:156](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156) (`int(counts.max().item()) == 0` early return on the masked Triton path).
- **Three in-scope forward signatures branch on `Optional[Tensor]`**:
  - `PrefilterKNN.forward(query, candidate_ids=None, counts=None)` ([prefilter_knn.py:40-45](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L40-L45)) — two coupled Optionals.
  - `OneBitKNN.forward(query, mask=None, candidate_ids=None)` ([one_bit_knn.py:79-84](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L79-L84)) — two Optionals forming three modes.
  - `SimilarityMasking.forward(query, mask=None)` ([similarity_masking.py:37-41](../../retrieve/src/retrieve/layers/linr/similarity_masking.py#L37-L41)) — one Optional.
  - `FullScanKNN.forward(query, mask=None, candidate_ids=None)` ([retrieval.py:36-41](../../retrieve/src/retrieve/layers/utils/retrieval.py#L36-L41)) — two Optionals.
- **Variable-K topk via `min(self.k, p)`** at [prefilter_knn.py:80](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L80), [retrieval.py:57](../../retrieve/src/retrieve/layers/utils/retrieval.py#L57), and in the kernel host wrappers ([fused_masked_knn_topk.py:164](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L164), [oporp_1bit_match_topk.py:193](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L193)). Produces a data-dependent output shape under tracing.
- **`p == 0` early returns** in two in-scope kernel host wrappers ([fused_masked_knn_topk.py:118-122](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L118-L122), [oporp_1bit_match_topk.py:150-154](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154)).
- **Two compiled helpers wrap the bloom / OPORP query-side projections in `torch.compile(dynamic=True, mode="reduce-overhead")` (cudagraph_trees)**: [bloom.py:261-263](../../retrieve/src/retrieve/layers/filters/bloom.py#L261-L263) (`_build_query_signatures_compiled`) and [quantize.py:109-111](../../retrieve/src/retrieve/layers/utils/quantize.py#L109-L111) (`_project_oporp_1bit_query_compiled`). Both are invoked inside `forward` on the CUDA path ([bloom.py:280-282](../../retrieve/src/retrieve/layers/filters/bloom.py#L280-L282), [quantize.py:128-130](../../retrieve/src/retrieve/layers/utils/quantize.py#L128-L130)) and reached from `OneBitKNN._project_query` ([one_bit_knn.py:77](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L77)) and `BloomFilter` evaluate paths. **This is new since the prior plan version**: `torch.export.export()` cannot trace through an inner `torch.compile` wrapper (it raises on the cudagraph-decorated callable). Eager bodies (`_build_query_signatures_eager`, `_project_oporp_1bit_query_eager`) already exist and produce bit-identical outputs — the export path has to call those directly.

**Goal of this refactor**: separate concerns so that adding `torch.library.triton_op` + `register_fake` + `aoti_compile_and_package` later becomes mechanical. **Non-goal**: actually wiring AOTI yet. The refactor itself stays on torch ≥ 2.4 (current pin in [retrieve/pyproject.toml](../../retrieve/pyproject.toml): `torch>=2.4,<3`, `triton>=3.0`) and does not require the libtorch / AOTI bump.

### What changed since the previous version of this plan

The previous plan was written against a tree where:

- Filters were named `ClauseIndex`, `BloomFilter` (no `_Torch` / `_Triton` split yet) and routed via `is_cuda` branches inside `evaluate_*`.
- LiNR retrieval modules were `LiNR_V1` / `LiNR_V2` / `LiNR_V3` plus parallel `_Triton` sibling classes.
- `evaluation/retrieval/algo_registry.py` and `evaluation/scripts/bench_filter_kernels.py` existed.
- No `torch.compile(reduce-overhead)` wrappers were embedded in `forward`.

Today's tree (May 2026):

- Filters are `ExactAttributeFilter` ([layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py)) and `BloomFilter` ([layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)). Both already take `backend: Backend = "triton"` in `__init__`; **no `is_cuda` branch lives in `evaluate_mask` / `evaluate_indices`**. The `_Torch` / `_Triton` class split the prior plan prescribed has **already happened, but as a constructor flag rather than two classes**. Either is structurally fine for export (one `.pt2` per backend is the same end-state); the simpler form is the one in the tree. Phase steps that prescribed creating sibling classes are gone.
- LiNR retrieval modules are `SimilarityMasking` (≈ V1, dense matmul + optional mask), `PrefilterKNN` (≈ V2, sparse rescore over candidate_ids), `OneBitKNN` (≈ V3, 1-bit Sign-OPORP Hamming). Each is **one class** with a `backend` flag, not two sibling classes.
- Algorithm wiring for the eval harness lives in [evaluation/retrieval/algos/](../../evaluation/retrieval/algos/) (one file per algo); the old `algo_registry.py` is gone, replaced by `build_algorithm` in [algos/__init__.py:54](../../evaluation/retrieval/algos/__init__.py#L54). The `linr_v1` / `linr_v2` / `linr_v3` wrappers each apply `torch.compile(dynamic=True, mode="default")` to the index module **only on `backend="torch"`** ([algos/linr_v1.py:52](../../evaluation/retrieval/algos/linr_v1.py#L52), [algos/linr_v2.py:41](../../evaluation/retrieval/algos/linr_v2.py#L41), [algos/linr_v3.py:60](../../evaluation/retrieval/algos/linr_v3.py#L60)). Export targets the **`backend="triton"` path**, which the algo wrappers leave un-compiled, so the algo-level wrapper doesn't itself block export — but the **inline `torch.compile(reduce-overhead)` helpers inside `bloom.py` / `quantize.py` do**, and they fire under `backend="triton"` too.
- No `evaluation/scripts/` directory; **no `bench_filter_kernels.py`**. The tune script lives nowhere yet; this plan creates `evaluation/scripts/tune_kernels.py` from scratch.
- No `evaluation/retrieval/build_export.py` and **no `IndexWrapper` / `AttrIndexWrapper`** anywhere in the repo (current `grep` confirms; the previous plan's references to those names were already moot at v1, and remain moot). Phase 4 creates this scaffold fresh (originally Phase 3's job; silvertorch is out of scope so authoring shifts to the first remaining phase).
- silvertorch is no longer in scope as of 2026-05-19 — see [../plans-silvertorch-backup/](../plans-silvertorch-backup/) for the prior plan covering silvertorch + IVF + sharding.

The bar is therefore: **"after the refactor, a fresh [evaluation/retrieval/build_export.py](../../evaluation/retrieval/build_export.py) can be authored that calls `torch.export.export()` per linr algorithm-mode (backend=triton) and gets a clean `.pt2` per entry"**. A scaffold for `PrefilterKNN` is part of Phase 4's scope; Phase 5 extends it. AOTI compile + package wiring stays out of scope.

End-state, after all 5 phases (Phase 3 is now a no-op stub):

- Every in-scope Triton kernel has a **pure-launch core** with non-Optional tensor args and Python-int/bool scalars only — the function shape `triton_op` will eventually wrap.
- Every kernel's block-size / warp / stage parameters come from a **per-kernel `Config` dataclass + REGISTRY** the layer threads in. No `@triton.autotune`.
- Every linr retrieval layer exposes **mode-pinned forward signatures** — Optional combinatorics collapse into either a constructor `mode=` flag or a sibling method (`forward_candidates`). Each mode is one `.pt2`.
- Post-launch host code (topk, gather, pad-to-K) lives in the layer, not in the kernel host wrapper.
- The cudagraph_trees-compiled query-side helpers (`_build_query_signatures_compiled`, `_project_oporp_1bit_query_compiled`) are **bypassed under export**: the layers call the bit-identical eager bodies directly when traced. The compiled wrappers still serve the eager runtime path.
- No `.item()` on any export-reachable code path.
- A new `evaluation/scripts/tune_kernels.py` benchmarks each kernel's config grid on the local GPU; results are pasted into the in-code REGISTRY.
- A new `evaluation/retrieval/build_export.py` exists and produces one `.pt2` per (algo, mode) entry. AOTI wiring is the follow-up.

## Design principles (apply to all phases)

1. **Trace-boundary API is non-Optional, scalar-typed.** Kernel host functions take only `Tensor` (always present, dummies allowed) + `int` / `bool`. No `Tensor | None`, no kwargs that flip code paths.
2. **Config is plain Python data, frozen at export time.** A `@dataclass(frozen=True)` per kernel, looked up once in the layer's `__init__` (not in the host wrapper, not in `forward`), stored as a Python attribute on the layer. At trace time it's a graph constant.
3. **One forward signature per export entry.** Each Optional combination becomes a construction-time choice (`mode=` flag) or a sibling method (`forward_candidates`). Each becomes its own `.pt2`.
4. **Post-launch host code (topk, gather, pad-to-K) is in the layer.** The kernel produces a raw `[B, P]` score buffer or `[B, N]` mask; the layer composes `topk` / `gather` / pad. This isolates the `min(k, p)` / `actual_k < k` problem to one place per layer.
5. **`.item()` is banned on the export path.** Where it currently exists ([clause_compact.py:158](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L158), [bloom_compact.py:132](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L132), [one_bit_knn.py:156](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156)), propagate the `counts` tensor downstream rather than slicing or guarding. Consumers (`fused_masked_knn_topk`, `oporp_1bit_match_topk`) already accept `counts`. The early-return-on-empty-mask guard in `OneBitKNN._forward_triton` is replaced by always calling the kernel — the kernel already short-circuits per-row when `counts[b] == 0`.
6. **`torch.compile` is not on the export path.** Anywhere a `torch.compile(...)` wrapper is reached from `forward` under `backend="triton"`, the layer must call the eager body directly when traced. The compiled wrapper stays as a separate runtime helper for eager callers. Specifically: `_build_query_signatures_compiled` and `_project_oporp_1bit_query_compiled` get gated out of the export path; their eager twins (`_build_query_signatures_eager`, `_project_oporp_1bit_query_eager`) are bit-identical and already exist.
7. **No deprecation shims.** Old kwarg-style host wrappers are deleted in the same PR that introduces the new core. Tests in [retrieve/tests/correctness/](../../retrieve/tests/correctness/) and [retrieve/tests/parity/](../../retrieve/tests/parity/) are updated in the same PR.
8. **Per-kernel block-size policy is decided during the phase**, not pre-mandated. Each phase picks bucketed-compile vs single-conservative based on observed shape distribution; document the choice in the PR description and in [docs/system/kernels.md](../../docs/system/kernels.md).
9. **Tests must pass before AND after the refactor**, with no relaxed tolerances. Parity tests in particular (Triton-vs-torch reference) must continue to assert bitwise-or-near-bitwise equivalence.

## Shared conventions established in Phase 1

The following code patterns are introduced in Phase 1 and reused verbatim by every subsequent phase. Subsequent phases reference them by name; they are defined here once.

### `KernelConfig` dataclass shape

```python
# In each kernel file:
@dataclass(frozen=True)
class FooKernelConfig:
    block_p: int       # or block_n / block_m, kernel-specific
    num_warps: int
    num_stages: int = 3
    # additional kernel-specific bucket fields if applicable

DEFAULT_CONFIG = FooKernelConfig(...)  # conservative fallback that runs everywhere

REGISTRY: dict[tuple[str, str], FooKernelConfig] = {
    ("sm_80", "small"):  FooKernelConfig(...),
    ("sm_80", "large"):  FooKernelConfig(...),
    ("sm_90", "small"):  FooKernelConfig(...),
    ("sm_90", "large"):  FooKernelConfig(...),
}

def lookup(device: torch.device, problem_hint: int) -> FooKernelConfig:
    if device.type != "cuda":
        return DEFAULT_CONFIG
    cap = torch.cuda.get_device_capability(device)
    arch = f"sm_{cap[0]}{cap[1]}"
    regime = "large" if problem_hint >= THRESHOLD else "small"
    return REGISTRY.get((arch, regime), DEFAULT_CONFIG)
```

The `problem_hint` is whatever scalar best characterizes the workload size (e.g., `N*W` for bloom, `B*N*D` for matmul-topk). Each phase picks one and documents it.

### Pure-launch core signature shape

```python
def _foo_kernel_launch(
    # All input tensors — always present, caller allocates dummies for "absent"
    query: Tensor,
    item_data: Tensor,
    optional_input_or_dummy: Tensor,  # 1x1 if not used
    # Output buffer(s) — caller allocates, kernel writes in place
    out_scores: Tensor,
    # Python scalars
    has_optional: bool,
    block_p: int,
    num_warps: int,
    num_stages: int,
) -> None:
    """Pure launch. Mutates out_scores in place. No Python branches on tensor properties."""
    b, _ = query.shape
    p = item_data.shape[1]
    grid = (b, triton.cdiv(p, block_p))
    _foo_kernel[grid](
        query, item_data, optional_input_or_dummy, out_scores,
        ...,  # strides
        HAS_OPTIONAL=has_optional,
        BLOCK_P=block_p,
        num_warps=num_warps,
        num_stages=num_stages,
    )
```

The launch function returns `None` and mutates outputs in place. The layer allocates `out_scores`, calls `_launch`, then runs topk/gather/pad in pure torch. (Today's wrappers — e.g. [`fused_masked_knn_topk`](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L92) — both allocate AND run post-launch topk inside the wrapper; this refactor splits those two responsibilities.)

### Mode-flag forward shape

```python
class FooLayer(RetrievalModule):
    def __init__(self, *, mode: Literal["full", "filtered"], k: int, backend: Backend = "triton"):
        super().__init__()
        if mode not in {"full", "filtered"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.k = k
        self.backend = backend

    def forward(self, query: Tensor, *extra: Tensor) -> tuple[Tensor, Tensor]:
        # exactly one signature per mode; Optionals do not appear here
        if self.mode == "full":
            return self._forward_full(query)
        # mode == "filtered" — caller supplies attrs in *extra
        return self._forward_filtered(query, *extra)
```

The `if self.mode == "full"` branch is on a Python attribute set in `__init__` — it specializes at export time (one branch per `.pt2`). This is the only kind of `if` allowed in `forward`. The existing `backend` flag stays — it picks the kernel vs eager-torch body of the chosen mode, and is also a Python attribute (specializes at export time too).

### Eager-body bypass for cudagraph-trees compile

Where a layer reaches `_build_query_signatures` ([bloom.py:266](../../retrieve/src/retrieve/layers/filters/bloom.py#L266)) or `project_oporp_1bit_query` ([quantize.py:114](../../retrieve/src/retrieve/layers/utils/quantize.py#L114)) inside a forward path that will be exported, replace the call with the eager variant. The pattern:

```python
# OLD (works for eager runtime, fails under torch.export):
qb = _build_query_signatures(query_clause_attrs, ...)  # dispatches to compiled on CUDA

# NEW (export-safe; same numerical output):
qb = _build_query_signatures_eager(query_clause_attrs, ...)
```

The compiled wrappers remain in the module for callers that **don't** export (e.g. the eval harness's `backend="torch"` paths). The export-target layer methods are the only spots that need to switch. We do **not** delete the compiled wrappers — they buy ~4× wall-clock on eager-runtime ([docs/system/kernels.md:278-289](../../docs/system/kernels.md#L278-L289)).

---

## Notes for executing agents

Read this before starting any phase. These are cross-cutting facts and gotchas that don't fit a single phase but each phase agent needs to know.

### Phase dependencies

```
Phase 1 ─────────► Phase 2 ──┐
                             ├──► Phase 3
                             │
   └────────────────────────►┤
   │                         │
   └─────────────────────────┴──► Phase 4 ──► Phase 5
```

- **Phase 1** has no in-refactor dependencies; it establishes the conventions (KernelConfig + REGISTRY + lookup, pure-launch core shape, the `[B, N]` full-width `(ids, counts)` API for compact kernels, and the test patterns) every later phase clones.
- **Phase 2** depends on Phase 1's KernelConfig shape and the `(ids, counts)` API.
- **Phase 3** depends on Phase 2 (`_build_query_signatures_eager` swap) and on Phase 1's conventions.
- **Phases 4 and 5** depend on Phase 1's `(ids, counts)` API. Phase 5 additionally depends on Phase 2's eager-bypass pattern (`OneBitKNN._project_query` carries the same cudagraph-trees compile that `BloomFilter._build_query_sigs` does).

There is no Phase 6: `SimilarityMasking` and `FullScanKNN` are pure-torch with optional masks. They're cleaned up as a small addendum (Cross-cutting §6 below) — no kernel to refactor, just `forward` signature collapse if they ever need an export entry.

### Repo basics

- **Repo layout note**: the repo root is `/workspace/retrieve/`. The Python package lives at [`/workspace/retrieve/retrieve/`](../../retrieve/) — top-level (note: NOT nested under a `retrieve/retrieve/src/retrieve/` triple; the previous version of this plan said "nested", which was outdated). The source root is [`/workspace/retrieve/retrieve/src/retrieve/`](../../retrieve/src/retrieve/). The eval app is its own package at [`/workspace/retrieve/evaluation/`](../../evaluation/). Plan paths are written relative to `/workspace/retrieve/docs/plans/`, so `../../retrieve/src/retrieve/...` resolves to the package source and `../../evaluation/...` to the eval app.
- **Test runner**: `cd retrieve && uv run pytest tests/` (or scope to `tests/correctness/test_X.py`). Both projects use `uv`; do not invoke `pip` or raw `pytest`.
- **Eval entry point**: `cd evaluation && uv run evaluate --config conf/<yaml>`. Algorithms come from the YAML config; override with `--algorithms <name>`. Valid in-scope algorithm names are listed in [evaluation/retrieval/algos/__init__.py:36-44](../../evaluation/retrieval/algos/__init__.py#L36-L44): `torch_knn`, `triton_knn` (alias for `linr_v1_filter_mask`), `linr_v1_filter_mask`, `linr_v3`, `linr_v2`, `voyager_hnsw`. (`silvertorch` is out of scope.) Use `--backend triton` / `--backend torch` to pin a backend. Configs live under [evaluation/conf/](../../evaluation/conf/) (e.g. `conf/500m/d128-quality.yaml`, `conf/goodreads/d128-filter.yaml`). Run benchmarks before AND after the refactor for the affected algorithm and diff recall@k metrics.
- **No CI**. Locally-green is the entire bar. There is no `.github/workflows/` directory in this repo, so a phase PR cannot rely on CI to catch regressions — run the full `tests/` suite, not just the touched test files.
- **Triton ≥ 3.0, torch ≥ 2.4 (capped <3)** (see [retrieve/pyproject.toml](../../retrieve/pyproject.toml)). Do not rely on torch 2.5+ features (e.g. `torch.library.triton_op`) in this refactor — those land in a later AOTI-wiring effort.

### Pre-existing export status

There is **no `evaluation/retrieval/build_export.py`** today, no `IndexWrapper` / `AttrIndexWrapper`, no `torch.export.export(...)` call site anywhere in the tree (confirmed via `grep`). The bar is "after the refactor, an export wrapper can be authored cleanly," not "no regression in export." Phase 4 creates `build_export.py` (for `PrefilterKNN` initially; originally Phase 3's job before silvertorch was dropped); Phase 5 extends it with `OneBitKNN` entries. The scaffold calls `torch.export.export()` per (algo, mode) and saves `.pt2`; it does **not** call `aoti_compile_and_package` (that's the next, separate effort).

### Kernel-internal quirks worth knowing

These are intentional; do not "fix" them while refactoring:

- **`clause_compact` has no `@triton.autotune`** ([clause_compact.py:24-30](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L24-L30)) because `tl.atomic_add` ([line 91](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L91)) corrupts state across autotune trials (each trial would see partially-mutated buffers). The fixed `_BLOCK_N=256, _NUM_WARPS=4` is correct. Phase 1 must NOT add autotune; the REGISTRY has one row per arch. **Same gotcha applies to `bloom_compact`** ([bloom_compact.py:24-27](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L24-L27)): also no autotune, also fixed `_BLOCK_N=256, _NUM_WARPS=4`.
- **`clause_mask` has no autotune** but it *could* — atomics are absent. The fixed `_BLOCK_N=256, _NUM_WARPS=4` ([clause_mask.py:17-18](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py#L17-L18)) was chosen to match `clause_compact`'s body. Phase 1 should treat its REGISTRY the same way as `clause_compact` (one row) and only widen the grid if the tune script shows benefit.
- **`oporp_1bit_match_topk` has a custom `_popcount_int64`** bit-twiddle helper ([oporp_1bit_match_topk.py:10-19](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L10-L19)) instead of using libdevice. This is portability across Triton versions — leave it alone.
- **`fused_masked_knn_topk` has an autotune-cache stabilizer** via `_bucket_p` ([fused_masked_knn_topk.py:9-19](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L9-L19), [line 36](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L36)) that buckets `P` so autotune compiles once per bucket. Phase 4 strips the `@triton.autotune` line entirely and replaces it with the REGISTRY+lookup pattern; the bucket lookup table moves to the REGISTRY (one entry per `(arch, p_bucket)`).
- **`fused_masked_knn_topk` has an `if p == 0` early return** in the host wrapper ([fused_masked_knn_topk.py:118-122](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L118-L122)). Phase 4 removes this — the kernel already handles `count == 0` per row ([line 65-66 reads `count = tl.load(counts_ptr + bid)` then `in_count = n_offsets < count`](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L65-L66)). The `p == 0` shortcut exists only to skip the launch when *every* row is empty; the export-friendly version always launches and lets per-row guards do their job. Same removal applies to the `p == 0` return in [oporp_1bit_match_topk.py:150-154](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154).

### `interfaces.py` contract decision

[interfaces.py](../../retrieve/src/retrieve/interfaces.py) defines **three** ABCs: `FilterModule`, `RetrievalModule`, and `ScorerModule`. None should add `mode` to their abstract contract — modes are subclass-specific (e.g., `OneBitKNN` will get `full` / `masked` / `candidates`; `PrefilterKNN` will get `full` / `candidates`). Forcing a uniform `mode` in the base would either over-constrain or be too vague. Each subclass declares its own `mode: Literal[...]` in `__init__`. The bases stay as-is.

The one exception: if any phase needs `forward_candidates` to be part of `RetrievalModule`'s declared interface (so type-checkers and downstream consumers can rely on it), add it there. If only some subclasses have it, leave it as a subclass method.

`FilterModule` already provides default implementations of `evaluate_indices` (via `compact_mask(self.evaluate_mask(...))`) and `evaluate_subset` (via gather). The current filter classes already opt into the defaults where there's no fused-kernel win; keep that pattern.

### When you finish a phase

In addition to that phase's verification block, run from repo root:

```bash
cd retrieve && uv run pytest tests/ -v   # full suite, not just the phase's tests
```

If anything outside your phase's intended surface area is now red, you have a regression — investigate before merging. The full-suite run is the only safety net since there is no CI.

Document the phase's bucket / mode / config decisions in the PR description and in [docs/system/kernels.md](../../docs/system/kernels.md) (kernel-specific) and [docs/system/architecture.md](../../docs/system/architecture.md) (layer-API).

---

## Phase 1 — `clause_compact` + `clause_mask` + `ExactAttributeFilter` (prototype, establishes conventions)

**Why first**: introduces the `.item()`-removal pattern via `counts` propagation, which Phases 2, 4, and 5 reuse. Also introduces the shared `KernelConfig` + REGISTRY + `lookup` shape that every later phase clones. `ExactAttributeFilter` has the simplest forward path of the consumer classes (no `mode`, no Optionals in `evaluate_*` — just one signature each).

**Difficulty: easy.** No autotune to strip. No Optionals in the launch. No `mode` flag to add (the filter's API is already one signature per evaluator). The only real surgery is moving the `.item()` slice out of the host wrapper and switching downstream consumers to full-width `(ids, counts)`.

`ExactAttributeFilter` today already routes through the `backend` flag set in `__init__` ([exact_attribute.py:31](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L31), [exact_attribute.py:55-86](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L55-L86)) — no `is_cuda` branches to remove. Both `clause_compact` and `clause_mask` already have fixed configs (no autotune); the kernel files just need a `@dataclass` + REGISTRY wrapping over the existing `_BLOCK_N` / `_NUM_WARPS` constants.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) — split wrapper into `_launch` (mutates buffers) + thin `clause_compact` API (allocates + returns full-width). Add `ClauseCompactConfig`.
- [retrieve/src/retrieve/kernels/triton/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) — same surgery (no `.item()` to remove here, but `_launch` shape).
- [retrieve/src/retrieve/layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) — `evaluate_indices` returns `(full_width_indices [B, N], counts [B])` instead of `(indices [B, P], counts [B])`. Update docstring.
- [retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) — `combine_indices` consumer (already accepts full-width `(ids, counts)` and reads `counts[b]` to bound each row at [filters/__init__.py:54-59](../../retrieve/src/retrieve/layers/filters/__init__.py#L54-L59); verify it still works once `evaluate_indices` actually returns full-width buffers). The `int(new_counts.max().item())` slice at [line 61](../../retrieve/src/retrieve/layers/filters/__init__.py#L61) is OK to leave — `combine_indices` is not on the export path.
- [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) — `compact_mask` uses `int(counts.max().item())` to pick output width ([compact.py:14](../../retrieve/src/retrieve/layers/utils/compact.py#L14)). The `backend="torch"` evaluate_indices path goes through it ([exact_attribute.py:86](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L86)). Decide: either change `compact_mask` to take `full_width: bool = False`, or sidestep it in `ExactAttributeFilter.evaluate_indices` torch branch by manually composing the full-width result. Recommend the latter (keep `compact_mask` simple for non-export callers; `combine_indices` still uses the trimmed-width form).
- [retrieve/tests/correctness/test_filters.py](../../retrieve/tests/correctness/test_filters.py)
- [retrieve/tests/correctness/test_compact.py](../../retrieve/tests/correctness/test_compact.py)
- [retrieve/tests/correctness/test_combine_filters.py](../../retrieve/tests/correctness/test_combine_filters.py) — exercises `combine_indices`; will see the API-shape change (input width becomes `N` instead of `max counts`).
- [retrieve/tests/parity/test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py) — import `_clause_compact_launch` directly, allocate `out_indices [B, N]` and `counts [B]` in the test, assert against the torch reference.
- [retrieve/tests/parity/test_clause_mask.py](../../retrieve/tests/parity/test_clause_mask.py) — import `_clause_mask_launch` directly, allocate `out_mask [B, N]`, assert against torch broadcast reference.
- `evaluation/scripts/tune_kernels.py` — **create new** (no scripts directory exists today). Click CLI; first entries `tune_clause_compact` and `tune_clause_mask`. Use `triton.testing.do_bench`; output a console table formatted to paste into the REGISTRY plus a per-arch JSON dump.
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md) — document the full-width `(ids, counts)` convention and the KernelConfig shape.

### Steps

1. **Kill `.item()`** at [clause_compact.py:158](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L158):
   - Old: kernel writes `[B, N]` indices buffer; host slices `out_indices[:, :p]` where `p = max(int(counts.max().item()), 1)`.
   - New: kernel writes the same `[B, N]` buffer; host returns `(out_indices, counts)` with full `N` width. Consumers honor `counts` to know how many entries are valid per row.

2. **Define `ClauseCompactConfig` and `ClauseMaskConfig`** with `block_n: int = 256, num_warps: int = 4, num_stages: int = 3` each. Neither kernel autotunes today; one REGISTRY entry per arch suffices. Two parallel dataclasses (even if values match initially) — the kernels share inner-loop structure but have different epilogues, so keep them as separate configs.

3. **Pure-launch cores**:
   - `_clause_compact_launch(item_clause_attrs, clause_is_reverse, query_clause_attrs, out_indices, counts, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_indices [B, N] int64` (initialized to `-1` per the existing sentinel convention at [clause_compact.py:126-133](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L126-L133)) and `counts [B] int64` zeros, passes them in; kernel mutates.
   - `_clause_mask_launch(item_clause_attrs, clause_is_reverse, query_clause_attrs, out_mask, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_mask [B, N] bool` via `torch.empty`; kernel writes every slot via the existing `mask=n_valid` store.

   Keep a thin `clause_compact(...)` / `clause_mask(...)` wrapper next to the launch core for **non-export callers** (parity tests, `combine_indices`). The wrapper does the allocation, calls `_launch`, and returns the same outputs the new `evaluate_*` paths emit. Internal allocation lives in the wrapper; export paths bypass the wrapper and call `_launch` directly with caller-allocated buffers.

4. **Update consumers**:
   - [filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) — `combine_indices` should keep working unchanged because it already gates by `counts` per row at [line 54](../../retrieve/src/retrieve/layers/filters/__init__.py#L54). Verify (the existing test in [test_combine_filters.py](../../retrieve/tests/correctness/test_combine_filters.py) should still pass without source changes).
   - [exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) — `evaluate_indices` calls `clause_compact` which now returns full-width. Update the return-shape docstring at [line 72](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L72) to say `(indices [B, N], counts [B])`. For the `backend="torch"` branch ([line 86](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L86)): replace `compact_mask(self.evaluate_mask(...))` with a manual full-width compose:
     ```python
     mask = self.evaluate_mask(qa)
     counts = mask.sum(dim=1)
     # full-width indices, -1 sentinel past counts[b]
     indices = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
     # populate; argsort works for full-width since trailing -1s sort to end
     sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
     valid = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < counts.unsqueeze(1)
     indices = torch.where(valid, sorted_idx, indices)
     return indices, counts
     ```
   - Any `fused_masked_knn_topk`, `oporp_1bit_match_topk` call sites that consume these — both kernels iterate over `counts[b]` per row already ([fused_masked_knn_topk.py:65-66](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L65-L66), [oporp_1bit_match_topk.py:67-69](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L67-L69)), so they accept wider `positive_indices` without correctness change. Verify by running the V2 / V3 parity tests after Phase 1 lands.

5. **No class split needed.** Today's `ExactAttributeFilter` already does device routing via the `backend` flag in `__init__` ([exact_attribute.py:31-33](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L31-L33)). The old plan's "split into `ClauseIndexTorch` + `ClauseIndexTriton`" step is **already done in a different form** — a single class with a constructor flag instead of two sibling classes. For export, each chosen `backend` specializes at trace time exactly the same way. Skip this step.

6. **Add kernel-config plumbing to `ExactAttributeFilter.__init__`**: accept `compact_config: ClauseCompactConfig | None = None, mask_config: ClauseMaskConfig | None = None`. Resolve via `lookup(device, problem_hint=n)` at `register_index` time (when the device is known and `N` is set). Store as plain Python attributes (graph constants at export time). Pass them through to `_clause_compact_launch` / `_clause_mask_launch` via Python ints.

   No `build_clause_index(...)` factory is needed: the class already constructs cleanly and `register_index` already takes the buffer. The old plan called for a builder; with the `backend` flag already in `__init__`, the builder is redundant.

7. **Update tests**: same patterns as today, but `evaluate_indices` returns full-width `[B, N]` instead of trimmed-width `[B, P]`. Tests should assert on `counts` for the valid prefix and on the `-1` sentinel for trailing positions. Parity tests import the launch core directly.

8. **Stand up `evaluation/scripts/tune_kernels.py`**. Click CLI: `--kernel <name>`, `--device cuda:0`, `--shape-grid path.json`. Entries `tune_clause_compact` and `tune_clause_mask` emit the REGISTRY-pasteable rows. Use `triton.testing.do_bench`. Output: per-arch JSON (informational) + console output formatted as `("sm_80", "small"): ClauseCompactConfig(block_n=256, num_warps=4, num_stages=3),`.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_filters.py tests/correctness/test_compact.py tests/correctness/test_combine_filters.py tests/parity/test_clause_compact.py tests/parity/test_clause_mask.py -v
grep -rn "\.item()" retrieve/src/retrieve/layers/filters/exact_attribute.py retrieve/src/retrieve/kernels/triton/filters/clause_compact.py retrieve/src/retrieve/kernels/triton/filters/clause_mask.py
# Should have zero hits in forward / evaluate_* / launch paths.
```

### Acceptance criteria

- [ ] No `.item()` calls in [clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py), [clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py), or [exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py).
- [ ] `_clause_compact_launch` and `_clause_mask_launch` exist with the documented signature shape (caller-allocated buffers, mutates in place, returns None).
- [ ] `ClauseCompactConfig` / `ClauseMaskConfig` dataclasses + REGISTRY + `lookup` defined and wired through `ExactAttributeFilter`.
- [ ] Downstream consumers (`combine_indices`, `compact_mask`, `fused_masked_knn_topk`, `oporp_1bit_match_topk`) consume full-width indices + counts. Eager-mode behavior unchanged.
- [ ] All correctness + parity tests pass; recall@k on `linr_v3` (which uses `clause` / `bloom` cascades) unchanged on at least one [evaluation/conf/](../../evaluation/conf/) YAML.
- [ ] `evaluation/scripts/tune_kernels.py` exists with the two clause kernel entries.

---

## Phase 2 — `bloom_compact` + `BloomFilter` + the `_build_query_signatures` export bypass

**Difficulty: medium.** `_build_query_signatures` is wrapped in `torch.compile(mode="reduce-overhead")` and reached from `BloomFilter.evaluate_mask` / `evaluate_indices`; the export path has to switch to the eager twin without breaking the eager runtime caller.

`BloomFilter` today already routes via the `backend` flag in `__init__` ([bloom.py:32](../../retrieve/src/retrieve/layers/filters/bloom.py#L32)). Same as Phase 1: no `is_cuda` branches in `forward` / `evaluate_*` to remove; the class-split step from the old plan is already obviated.

> **Scope note:** `bloom_match` (silvertorch-only kernel) was originally part of this phase. It is out of scope; the `evaluate_mask` path that linr's V1 uses today calls `bloom_match` only on the silvertorch IVF path. For linr, `BloomFilter.evaluate_indices` (driven by `bloom_compact`) is the relevant route. The `_build_query_signatures` export bypass remains in scope because `OneBitKNN._project_query` / `BloomFilter.evaluate_indices` reach it on the linr path. See [../plans-silvertorch-backup/torch-export-refactor.md](../plans-silvertorch-backup/torch-export-refactor.md) for the original `bloom_match` bucket-policy write-up.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py)
- [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [retrieve/tests/correctness/test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py)
- [retrieve/tests/parity/test_bloom_compact.py](../../retrieve/tests/parity/test_bloom_compact.py)
- `evaluation/scripts/tune_kernels.py` (extend Phase 1's file)
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md)

### Steps

1. **Pre-flight check on `_build_signatures`** ([bloom.py:134](../../retrieve/src/retrieve/layers/filters/bloom.py#L134)): this is index-build-time only (`register_index`), not on the per-call export path. It's allowed to use `scatter_` and Python `range()` chunking.

2. **The `_build_query_signatures` export-path swap** ([bloom.py:266-282](../../retrieve/src/retrieve/layers/filters/bloom.py#L266-L282)). This **is** on the export path (every `BloomFilter.evaluate_*` call goes through `_build_query_sigs` ([line 67-74](../../retrieve/src/retrieve/layers/filters/bloom.py#L67-L74)) which dispatches to `_build_query_signatures`).
   - Approach: add a module-level Python flag (or, cleaner, an `__init__`-time `self._compile_query_sigs: bool` set from a constructor arg) on `BloomFilter` that picks between `_build_query_signatures` (current dispatch wrapper) and `_build_query_signatures_eager` (the bit-identical body without the `torch.compile` layer). Default: `True` (use compiled wrapper) so eager runtime keeps the cudagraph_trees 4× win documented at [docs/system/kernels.md:278-289](../../docs/system/kernels.md#L278-L289). Export call sites set it to `False`.
   - Alternative: thread the choice via `torch.compiler.is_compiling()` / `torch._dynamo.is_compiling()` — runtime check. Probably more brittle than a Python attribute set at construction time; recommend the constructor-flag approach.
   - Same swap pattern lands in Phase 5 (OPORP's `project_oporp_1bit_query`). Phase 2 establishes the pattern; Phase 5 clones it.

3. **Define `BloomCompactConfig`** with `block_n: int, num_warps: int, num_stages: int = 3`. **`bloom_compact` does *not* need bucketing** — it uses fixed `BLOCK_N=256` and tail-masks via `n_valid`; one REGISTRY row per arch suffices, and it must NOT autotune for the same atomic_add reason as `clause_compact`.

4. **Pure-launch core**:
   - `_bloom_compact_launch(qb, sigs, out_indices, counts, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_indices [B, N] int64` (initialized to `-1` per the existing sentinel convention) and `counts [B] int64` zeros. Same full-width-indices + counts-propagation policy as `_clause_compact_launch` from Phase 1 — no `.item()` slice (today at [bloom_compact.py:132](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L132)).

5. **`BloomFilter` plumbing**: accept `compact_config: BloomCompactConfig | None = None` in `__init__`. Resolve at `register_index` time. `evaluate_indices` uses the fixed `compact_config`.

6. **No class split needed.** Same reasoning as Phase 1: `BloomFilter` already has a `backend` flag ([bloom.py:32](../../retrieve/src/retrieve/layers/filters/bloom.py#L32)) and no `is_cuda` branches inside `evaluate_*`. Skip.

7. **Update tests**: `test_bloom_filter.py` and `test_bloom_compact.py` use the new launch signatures. The `_build_signatures` / `_build_query_signatures_eager` unit tests should remain unchanged. Add a smoke test that confirms `BloomFilter` constructed with the export-flag set bypasses the compiled wrapper (e.g. by patching `_build_query_signatures_compiled` to raise and verifying the eager path doesn't reach it).

8. **Tune script**: extend `evaluation/scripts/tune_kernels.py` with `tune_bloom_compact` (single row).

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_bloom_filter.py tests/parity/test_bloom_compact.py -v
grep -rn "\.item()" retrieve/src/retrieve/layers/filters/bloom.py retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py
# Should have zero hits in forward / evaluate_* / launch paths.
# (The `bool(clause_is_reverse.any().item())` in register_index is OK — index-build path, not export.)

# Smoke-export a BloomFilter:
uv run python -c "
import torch
from retrieve.layers.filters.bloom import BloomFilter
bf = BloomFilter(m_bits=1024, k_hash=5, backend='triton')  # plus the export flag once defined
attrs = torch.randint(0, 100, (1024, 4, 2), dtype=torch.long, device='cuda')
bf.register_index(attrs)
qa = torch.randint(0, 100, (8, 4), dtype=torch.long, device='cuda')
ep = torch.export.export(bf, (qa,))
print(ep.graph_module.code[:1000])
"
```

### Acceptance criteria

- [ ] No `.item()` calls in `bloom.py` evaluate paths or `bloom_compact.py`.
- [ ] `_bloom_compact_launch` exists with the documented signature shape.
- [ ] `BloomCompactConfig` dataclass + REGISTRY in place.
- [ ] `bloom_compact` documented as fixed single-config.
- [ ] `BloomFilter` has a way to bypass `_build_query_signatures_compiled` on the export path; eager runtime still uses the compiled wrapper (no regression on `evaluate.py` benchmarks).
- [ ] Tests pass.

---

## Phase 3 — `codesigned_probe_score` + `SilverTorch`

**Out of scope — silvertorch deprecated (2026-05-19).**

This phase originally exported `SilverTorch` and authored the `build_export.py` scaffold starting from silvertorch. With silvertorch out of scope, Phase 3 is dropped; the `build_export.py` scaffold is instead authored in Phase 4 (PrefilterKNN). The phase number is preserved so cross-doc references stay stable.

See [../plans-silvertorch-backup/torch-export-refactor.md](../plans-silvertorch-backup/torch-export-refactor.md) for the original Phase 3 write-up (autotune-strip on `codesigned_probe_score`, `SilverTorch` mode flag, three export entries, original `build_export.py` scaffold).

---

## Phase 4 — `fused_masked_knn_topk` + `PrefilterKNN`

**Difficulty: medium.** Strip autotune (move `_bucket_p`'s logic into REGISTRY); the `do_not_specialize=["P_REAL"]` JIT hint goes away once the autotune key does. Two coupled Optionals in `PrefilterKNN.forward` (`candidate_ids=None, counts=None`) collapse into two modes. **This phase also authors `evaluation/retrieval/build_export.py` from scratch** (originally Phase 3's job; Phase 3 is now a no-op stub, so the scaffold lands here).

Depends on Phase 1's `(indices, counts)` API stabilizing — `PrefilterKNN` calls `fused_masked_knn_topk` with `candidate_ids` and `counts` that come from `ExactAttributeFilter.evaluate_indices` upstream.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py)
- [retrieve/src/retrieve/layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- [retrieve/tests/parity/test_fused_masked_knn_topk.py](../../retrieve/tests/parity/test_fused_masked_knn_topk.py)
- `evaluation/retrieval/build_export.py` — **create new** (does not exist today). Phase 4 owns the scaffold; Phase 5 extends it.
- [evaluation/retrieval/algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) — `LinrV2Algo` adds `mode=` pass-through (always `"candidates"`, since the V2 algo path is only the sparse-rescore case).
- `evaluation/scripts/tune_kernels.py`

### Steps

1. **Strip autotune** at [fused_masked_knn_topk.py:36](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L36) and remove the `do_not_specialize=["P_REAL"]` JIT hint at [line 37](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L37) (the hint exists because autotune's cache key was different from the runtime `P` — without autotune, the kernel just specializes on `P` like every other constexpr arg). Define `FusedMaskedKnnTopkConfig` (`block_n, num_warps, num_stages`). Migrate the existing `_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)` ladder ([line 9](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L9)) into the REGISTRY: one row per `(arch, p_bucket)`. `problem_hint = p_bucket`.

2. **Pure-launch core**: `_fused_masked_knn_topk_launch(query, item_embs, positive_indices, counts, out_scores, *, d, block_n, num_warps, num_stages) -> None`. Caller allocates `out_scores [B, P] fp32 empty`. Kernel mutates. Indices and counts are always present (full-width from Phase 1).

3. **Remove the `if p == 0` early return** at [fused_masked_knn_topk.py:118-122](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L118-L122). The per-row `count == 0` guard inside the kernel ([line 65-66](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L65-L66)) already handles the all-empty case correctly. The host-side shortcut exists only as a perf optimization for empty inputs and breaks export tracing.

4. **`PrefilterKNN.__init__` mode flag**: add `mode: Literal["full", "candidates"]`. `mode="full"` exports as `forward(query)` and uses the existing `_forward_full` ([prefilter_knn.py:52-55](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L52-L55)) which is just `query @ item_embs.t()` + topk — pure torch, no kernel, trivially exportable. `mode="candidates"` exports as `forward(query, candidate_ids, counts)` and calls `_fused_masked_knn_topk_launch` (under `backend="triton"`) or `_forward_prefilter` (under `backend="torch"`).

   Drop the existing `candidate_ids is None` Optional branch at [prefilter_knn.py:46-47](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L46-L47). The `counts is None` Optional at [prefilter_knn.py:109-110](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L109-L110) (which fills counts with `p` when missing) also collapses — the new signature requires `counts` to be a non-Optional `Tensor`. Callers that don't have a real counts buffer (e.g. `LinrV3Algo` in `linr_v3.py:73-75`) pre-build one (`(cand_ids >= 0).sum(dim=1)`).

5. **Move post-launch topk + gather + pad to `PrefilterKNN.forward`** for the candidates mode. The current host wrapper at [fused_masked_knn_topk.py:164-194](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L164-L194) (topk + masked-fill-by-isfinite + cat-pad-to-K) moves into the layer. Same static-shape pad-to-K policy as Phase 3.

6. **Drop the early `if p == 0` return** in `PrefilterKNN._forward_prefilter_triton` and `_forward_prefilter` ([prefilter_knn.py:66-70, 103-108](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L66-L108)) — same reasoning as Step 3.

7. **Tests**: parametrize over mode; the `backend="torch"` `_forward_full` and `_forward_prefilter` paths stay reachable (different code path, same numerical reference).

8. **Create `evaluation/retrieval/build_export.py`** from scratch with `linr_v2` entries for `mode="full"` and `mode="candidates"`. Scaffold:
    ```python
    """Export linr retrieval modules to .pt2 — one file per (algo, mode).

    AOTI compile + package is the next, separate effort.
    """
    from __future__ import annotations
    from pathlib import Path
    from typing import Literal
    import click
    import torch

    from retrieve.layers.linr.prefilter_knn import PrefilterKNN
    # ... loaders to fetch item_embs / item_attrs from a checkpoint dir


    def export_linr_v2(
        item_embs: torch.Tensor,
        out_dir: Path,
        *,
        mode: Literal["full", "candidates"],
        k: int = 1024,
    ) -> Path:
        idx = PrefilterKNN(k=k, mode=mode, backend="triton")
        idx.register_index(item_embs)
        b, d = 8, item_embs.shape[1]
        query = torch.randn(b, d, device=item_embs.device)
        if mode == "full":
            ep = torch.export.export(idx, (query,))
        elif mode == "candidates":
            cand = torch.zeros(b, k, dtype=torch.long, device=item_embs.device)
            counts = torch.full((b,), k, dtype=torch.long, device=item_embs.device)
            ep = torch.export.export(idx, (query, cand, counts))
        out = out_dir / f"linr_v2_{mode}.pt2"
        torch.export.save(ep, out)
        return out


    @click.command()
    @click.option("--algo", required=True, type=click.Choice(["linr_v2"]))  # extended in Phase 5
    @click.option("--mode", required=True, type=str)
    @click.option("--checkpoint-dir", required=True, type=click.Path(exists=True, path_type=Path))
    @click.option("--out-dir", required=True, type=click.Path(path_type=Path))
    def main(algo, mode, checkpoint_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        # ... load item_embs from checkpoint_dir ...
        if algo == "linr_v2":
            path = export_linr_v2(item_embs, out_dir, mode=mode)
        else:
            raise click.UsageError(f"unknown algo: {algo}")
        click.echo(f"wrote {path}")


    if __name__ == "__main__":
        main()
    ```
    Plus a `[project.scripts]` entry in [evaluation/pyproject.toml](../../evaluation/pyproject.toml) (`build-export = "retrieval.build_export:main"`). Minimal scope: `linr_v2` modes; Phase 5 adds `linr_v3` entries.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py tests/parity/test_fused_masked_knn_topk.py -v
# End-to-end with clause_compact (Phase 1) → fused_masked_knn_topk via the V2 algo:
cd evaluation && uv run evaluate --config conf/goodreads/d128-filter.yaml --algorithms linr_v2 --backend triton
# Recall@k diff before/after should be zero or noise.
cd evaluation && uv run build-export --algo linr_v2 --mode candidates --checkpoint-dir <path> --out-dir <path>
```

### Acceptance criteria

- [ ] No autotune on `fused_masked_knn_topk`. Config + REGISTRY in place.
- [ ] No `if p == 0` early return in the host wrapper; per-row `in_count` guard inside the kernel does the work.
- [ ] No Optional args in `_fused_masked_knn_topk_launch`.
- [ ] `PrefilterKNN.__init__` accepts `mode: Literal["full", "candidates"]`.
- [ ] Cascade `linr_v3 → linr_v2` (the production filter cascade) still works end-to-end.
- [ ] `build_export.py` produces `linr_v2_full.pt2` and `linr_v2_candidates.pt2`.

---

## Phase 5 — `oporp_1bit_match_topk` + `OneBitKNN`

**Difficulty: high.** Last because it has the most combinations: `HAS_INDICES` constexpr drives an autotune key today, the layer has three modes (full / masked / candidates) folded into two coupled Optionals, the masked path has a `.item()` early-return guard, and the query-projection path goes through a `torch.compile(reduce-overhead)` wrapper (same as Phase 2's bloom-sig issue).

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py)
- [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)
- [retrieve/src/retrieve/layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py) — same eager-bypass pattern as Phase 2 (`project_oporp_1bit_query` / `_project_oporp_1bit_query_compiled` at [lines 109-130](../../retrieve/src/retrieve/layers/utils/quantize.py#L109-L130)).
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- [retrieve/tests/parity/test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)
- `evaluation/retrieval/build_export.py` (extend with `linr_v3` entries — three modes)
- [evaluation/retrieval/algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) — `LinrV3Algo` adds `mode=` pass-through.
- `evaluation/scripts/tune_kernels.py`

### Steps

1. **Strip autotune** (8 configs keyed on `N, W, HAS_INDICES` at [oporp_1bit_match_topk.py:36](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L36)). The `HAS_INDICES` key collapses into a constructor `mode` flag on the layer (not in the kernel autotune key). Define `OporpMatchTopkConfig` (`block_n, num_warps, num_stages`). `problem_hint = N`.

2. **Pure-launch core**: `_oporp_1bit_match_topk_launch(query_bits, item_bits, positive_indices_or_dummy, counts_or_dummy, out_scores, *, has_indices, d_total, block_n, num_warps, num_stages) -> None`. Dummies for absent inputs. The kernel already has `HAS_INDICES` constexpr at [line 55](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L55) and dummy-tensor binding at [host wrapper lines 166-169](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L166-L169) — move the dummy-binding out of the launch site into the layer.

3. **Remove the `if n_loop == 0` early return** at [oporp_1bit_match_topk.py:150-154](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154). Same reasoning as Phase 3 / Phase 4.

4. **`OneBitKNN.__init__` mode flag**: add `mode: Literal["full", "masked", "candidates"]`. Three modes collapse the (`mask is not None`, `candidate_ids is not None`) Optional matrix at [one_bit_knn.py:120-169](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L120-L169). Each mode is a separate export entry. The current dispatch is in `_forward_triton` ([lines 120-169](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L120-L169)) and `_forward_torch_eager` ([lines 89-118](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L89-L118)) — both have the same three-branch structure; refactor both to the mode-dispatched shape.

5. **Kill `int(counts.max().item()) == 0`** at [one_bit_knn.py:156](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156). The early-return-with-sentinel-values guard ([lines 156-162](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156-L162)) was a perf optimization for fully-masked inputs; the kernel's per-row `count == 0` guard already produces the same sentinel output (the masked path's `valid_score = in_count` at [line 80](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L80) emits `-inf` for empty rows, which after `topk` gives `-inf` scores → the layer-level `torch.where(isfinite, ids, -1)` sentinel logic at [oporp_1bit_match_topk.py:198-200](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L198-L200) does the rest). Remove the early return; always launch the kernel.

   Note that `compact_mask` ([compact.py:14](../../retrieve/src/retrieve/layers/utils/compact.py#L14)) — which the masked path calls at [one_bit_knn.py:155](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L155) — also uses `.item()`. The export path for `mode="masked"` must bypass `compact_mask` and consume the full-width `(indices, counts)` directly. Two options:
   - **(A)** Change `mode="masked"` to accept `(indices, counts)` instead of a `mask` (caller pre-compacts upstream — e.g. `ExactAttributeFilter.evaluate_indices` already returns this shape after Phase 1). This is the natural fit and removes `compact_mask` from the export path entirely.
   - **(B)** Add a full-width `compact_mask_full_width` helper. More overhead; less clean.

   Recommended: **(A)**. The mode is then effectively `"candidates"` with `(indices, counts)` shape — rename `mode="masked"` to `mode="indexed"` or fold into `mode="candidates"` with an additional `counts` arg. Pick one and document.

6. **Swap `project_oporp_1bit_query` → `_project_oporp_1bit_query_eager`** ([one_bit_knn.py:77](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L77)) on the export path. Same constructor-flag mechanism as Phase 2's `_build_query_signatures` swap.

7. **Move post-launch topk + gather + pad to `OneBitKNN.forward`** for each mode. Same pad-to-K policy as earlier phases.

8. **Tests + tune script**: parametrize tests over mode; add `tune_oporp_1bit_match_topk` to the tune script.

9. **Extend `build_export.py`**: add `linr_v3_full.pt2`, `linr_v3_indexed.pt2` (or whichever name from Step 5), `linr_v3_candidates.pt2` entries.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py tests/parity/test_oporp_1bit_match_topk.py -v
grep -rn "\.item()" retrieve/src/retrieve/layers/linr/one_bit_knn.py retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py
# Zero hits.
cd evaluation && uv run evaluate --config conf/500m/d128-quality.yaml --algorithms linr_v3 --backend triton
cd evaluation && uv run build-export --algo linr_v3 --mode full --checkpoint-dir <path> --out-dir <path>
```

### Acceptance criteria

- [ ] No autotune; three explicit modes on the layer.
- [ ] No `.item()` on the export path.
- [ ] `OneBitKNN` export path bypasses `_project_oporp_1bit_query_compiled`.
- [ ] `compact_mask` is not reached from any export-path forward.
- [ ] All tests pass; `linr_v3` recall@k unchanged.

---

## Cross-cutting work (rides along with the phases, no separate PR)

1. **`evaluation/scripts/tune_kernels.py`** — created fresh in Phase 1 (no `evaluation/scripts/` directory exists today). Click CLI with `--kernel <name>`, `--device cuda:0`, `--shape-grid path.json`. Each phase appends one or two `tune_<kernel>` functions. Uses `triton.testing.do_bench`. Output is per-arch JSON (informational) + console output formatted to paste into REGISTRY. Wire a `[project.scripts]` entry in [evaluation/pyproject.toml](../../evaluation/pyproject.toml): `tune-kernels = "scripts.tune_kernels:main"` (and add `scripts` to the `packages` list at [pyproject.toml:38](../../evaluation/pyproject.toml#L38)).

2. **`evaluation/retrieval/build_export.py`** — created in Phase 4, extended through Phase 5 (originally Phase 3's job before silvertorch was dropped):
   - Phase 3: out of scope (silvertorch).
   - Phase 4: scaffolds the file; adds `linr_v2` entries (`full`, `candidates`).
   - Phase 5: adds `linr_v3` entries (`full`, `indexed`-or-equivalent, `candidates`).
   - Single Click CLI; one `.pt2` per (algo, mode); no AOTI compile yet.
   - **Note**: `linr_v1` (`SimilarityMasking`) and `torch_knn` (`FullScanKNN`) don't need export entries yet — both are pure-torch dense matmul + topk, which exports trivially. Add them only if downstream consumers want a uniform `.pt2` zoo.

3. **[docs/system/](../../docs/system/)** updates per phase:
   - `kernels.md` for kernel-specific changes (per-kernel config + REGISTRY documented, per-kernel bucket policy if any).
   - `architecture.md` for layer API changes (the `mode` flag per class, the export-path bypass for the compiled query helpers).
   - `filtering.md` for the full-width `(ids, counts)` convention from Phase 1.
   - `testing.md` for the per-phase verification commands and the "no autotune in tests; configs are constants" rule.
   - All four files exist today and are claude-code-generated; keep them in sync.

4. **[retrieve/pyproject.toml](../../retrieve/pyproject.toml)** version pins stay (`torch>=2.4,<3`, `triton>=3.0`). The AOTI bump (`torch>=2.5`) is a separate later effort. The `[tool.ruff.lint.per-file-ignores]` for `kernels/triton/**/*.py` E731 (the `grid = lambda meta: ...` carve-out at [pyproject.toml:34-35](../../retrieve/pyproject.toml#L34-L35)) becomes unneeded once every kernel switches to a static `grid = (b, triton.cdiv(p, block_p))` — but leaving it in costs nothing; remove in a follow-up cleanup PR.

5. **[evaluation/retrieval/algos/](../../evaluation/retrieval/algos/)** call sites get adjusted as each phase lands:
   - Phase 1: no algo change (`ExactAttributeFilter` is built via [algos/filter.py](../../evaluation/retrieval/algos/filter.py) which already takes a `backend` flag).
   - Phase 2: no algo change (same — `BloomFilter` is built via the same helper).
   - Phase 3: out of scope (silvertorch deprecated).
   - Phase 4: [algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) passes `mode="candidates"` (the only V2 path).
   - Phase 5: [algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) — stage 1 is `OneBitKNN(mode="full")` or `OneBitKNN(mode="indexed")` (depending on whether a filter is wired; see the `make_mask` call at [algos/linr_v3.py:65](../../evaluation/retrieval/algos/linr_v3.py#L65)), and stage 2 is `PrefilterKNN(mode="candidates")`.

6. **`SimilarityMasking` and `FullScanKNN` addendum** (small, no separate phase): both have `mask: Tensor | None = None` (and `candidate_ids: Tensor | None = None` on `FullScanKNN`). For export, add a `mode` flag to each:
   - `SimilarityMasking`: `mode: Literal["full", "masked"]`. `mode="masked"` requires `mask` as a real tensor; `mode="full"` exports `forward(query)`.
   - `FullScanKNN`: `mode: Literal["full", "masked", "candidates"]`. Same shape as `OneBitKNN`. The current `_forward_candidates`'s `min(self.k, scores.shape[1])` at [retrieval.py:57](../../retrieve/src/retrieve/layers/utils/retrieval.py#L57) gets the same pad-to-K treatment.
   - Roll this in alongside Phase 4 (`SimilarityMasking` shares no kernels with anyone) and Phase 5 (`FullScanKNN` is structurally identical to `OneBitKNN` — same Optional matrix). Or do it as a separate cleanup PR after Phase 5 lands; it has no kernel surgery so it's pure layer-API churn.

## Files NOT in scope

- [retrieve/src/retrieve/layers/utils/kmeans.py](../../retrieve/src/retrieve/layers/utils/kmeans.py), the `_build_signatures` / `_generate_seeds` index-builders, `quantize_oporp_1bit` — all `register_index`-time only, not on the export path. (Silvertorch's `quantize_int8` and the IVF index-build helpers are out of scope; see [../plans-silvertorch-backup/torch-export-refactor.md](../plans-silvertorch-backup/torch-export-refactor.md).)
- [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) — `compact_mask` keeps its `.item()` for the trim-to-max path; it's a host utility for non-export callers. Phase 5 routes `OneBitKNN` around it via the `(indices, counts)` API; if any phase finds another caller on the export path, route around it there too rather than reworking the helper.
- [retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) `combine_indices` — sparse-cascade helper, uses `.item()` for compaction width. Not on the per-call export path; consumers are offline pipelines.
- The `_build_query_signatures_compiled` / `_project_oporp_1bit_query_compiled` wrappers themselves — they **stay** as eager-runtime helpers. Phases 2 and 5 only add a **bypass mechanism** so the export path calls the eager body; the compiled wrapper continues to win 4× wall-clock on the eager benchmark path documented in [docs/system/kernels.md:278-289](../../docs/system/kernels.md#L278-L289).
- [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) — base class signatures stay; `forward_candidates`, if it becomes worth declaring on `RetrievalModule`, is a small in-PR change inside whichever phase adds it (most likely Phase 3).
- [evaluation/retrieval/algos/torch_knn.py](../../evaluation/retrieval/algos/torch_knn.py), [voyager.py](../../evaluation/retrieval/algos/voyager.py), [linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) — pure-torch / external-library wrappers; no kernel surgery, no Optional collapsing required unless an export entry is wanted.

## End-state verification (after all 5 phases)

```bash
# 1. Full test suite green:
cd retrieve && uv run pytest tests/ -v

# 2. Eager-mode E2E unchanged for every algorithm in the registry, both backends:
cd evaluation && uv run evaluate --config conf/500m/d128-quality.yaml --backend triton
cd evaluation && uv run evaluate --config conf/500m/d128-quality.yaml --backend torch
cd evaluation && uv run evaluate --config conf/goodreads/d128-filter.yaml --backend triton

# 3. Export round-trips for every in-scope (algo, mode):
cd evaluation && uv run build-export --algo linr_v2      --mode full       --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run build-export --algo linr_v2      --mode candidates --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run build-export --algo linr_v3      --mode full       --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run build-export --algo linr_v3      --mode indexed    --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run build-export --algo linr_v3      --mode candidates --checkpoint-dir <path> --out-dir <path>

# 4. Sweeps clean across the in-scope codebase:
grep -rn "@triton.autotune" retrieve/src/retrieve/kernels/triton/filters/ retrieve/src/retrieve/kernels/triton/linr/  # zero hits
grep -rn "\.item()" retrieve/src/retrieve/kernels/triton/filters/ retrieve/src/retrieve/kernels/triton/linr/ retrieve/src/retrieve/layers/filters/ retrieve/src/retrieve/layers/linr/ retrieve/src/retrieve/layers/utils/  # only register_index / compact_mask / combine_indices hits, none in forward / launch
grep -rn "Tensor | None\|Optional\[Tensor\]" retrieve/src/retrieve/kernels/triton/filters/ retrieve/src/retrieve/kernels/triton/linr/  # zero hits in launch signatures
grep -rn "is_cuda" retrieve/src/retrieve/layers/filters/ retrieve/src/retrieve/layers/linr/  # no new hits beyond the existing in bloom.py / quantize.py, both of which feed the compile-bypass dispatchers
```

## Phase-difficulty summary (one-line each)

| Phase | Kernels | Layer | Difficulty | Why                                                                                                                |
|-------|---------|-------|------------|--------------------------------------------------------------------------------------------------------------------|
| 1     | `clause_compact`, `clause_mask` | `ExactAttributeFilter` | **easy**   | No autotune to strip, no Optionals in launch. Just `.item()` removal + KernelConfig plumbing.                       |
| 2     | `bloom_compact`                 | `BloomFilter`          | **medium** | Introduces the cudagraph-trees compile bypass pattern (reused in Phase 5). `bloom_match` was originally here — out of scope (silvertorch). |
| 3     | — (out of scope)                | — (silvertorch)        | **n/a**    | Originally `codesigned_probe_score` + `SilverTorch`. Out of scope; `build_export.py` scaffold authored in Phase 4 instead. |
| 4     | `fused_masked_knn_topk`         | `PrefilterKNN`         | **medium** | Strip autotune (migrate `_bucket_p` to REGISTRY), one mode flag, two coupled Optionals collapse, `p==0` removal. Authors `build_export.py`. |
| 5     | `oporp_1bit_match_topk`         | `OneBitKNN`            | **high**   | Three modes, `.item()` guard removal, `compact_mask` bypass, OPORP query-side compile bypass.                       |
