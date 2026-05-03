# Concern-separation refactor: retrieve module → torch.export-ready

> **Document audience**: Each phase below is a self-contained brief intended to be executed by a separate SWE agent without access to this conversation's history. Phases are ordered and depend on prior phases for shared conventions; do not run later phases out of order, but each phase's PR is reviewed and merged independently.

## Context

The `retrieve` module ships **7 Triton kernels** and ~9 `nn.Module` retrieval/filter layers under [retrieve/src/retrieve/](../../retrieve/src/retrieve/). The kernels and layers carry a set of patterns that block clean `torch.export.export()` traces and the future `torch.library.triton_op` + `register_fake` + `aoti_compile_and_package` wrap-up:

- **5 of 7 kernels** use `@triton.autotune` with lambda-meta grids (`grid=lambda meta: (b, triton.cdiv(p, meta["BLOCK_P"]))`), which inductor's compile-time autotuner cannot serialize portably. The 2 atomics-tail kernels (`clause_compact`, `bloom_compact`) deliberately do not autotune — they share a tail that would corrupt under autotune trials.
- **3 host wrappers** (`codesigned_probe_score`, `oporp_1bit_match_topk`) branch on `Optional[Tensor]` arguments and rebind to dummy `1×1` tensors keyed by `HAS_X: tl.constexpr` flags.
- Filter layer `forward`/`evaluate_*` methods branch on `is_cuda` ([clause.py](../../retrieve/src/retrieve/layers/filters/clause.py), [bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)).
- Retrieval layer `forward` methods branch on Optional tensor args (`candidate_ids is not None`, `query_clause_attrs is not None`, `mask is not None`) — **`SilverTorch.forward` now carries three Optionals: `query_clause_attrs`, `mask`, `candidate_ids`** (see [silvertorch/main.py:126](../../retrieve/src/retrieve/layers/silvertorch/main.py#L126)).
- Two host wrappers compute Python-int values via `.item()` and use them to slice tensors (`out_indices[:, :p]` at [clause_compact.py:151](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L151) and [bloom_compact.py:128](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L128)).
- Several layers compute `min(self.k, scores.shape[1])` for variable-K topk, producing data-dependent output shapes (`SilverTorch._forward_candidates`, `LiNR_V2._forward_prefilter`, `LiNR_V3._forward_candidates`, `FullScanKNN._forward_candidates`).
- `LiNR_V3_Triton.forward` calls `int(counts.max().item()) == 0` as an early-return guard ([v3_triton.py:52](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L52)). Same `.item()` ban on the export path applies.
- `SilverTorch.register_index` calls `int(cluster_sizes.max().item())` ([main.py:87](../../retrieve/src/retrieve/layers/silvertorch/main.py#L87)) — fine, `register_index` is not on the export path.
- `combine_indices` ([filters/__init__.py:61](../../retrieve/src/retrieve/layers/filters/__init__.py#L61)) calls `int(new_counts.max().item())` for sparse cascade compaction. Not on the export path itself, but consumers should prefer the full-width `(ids, counts)` API where possible.

**Goal of this refactor**: separate concerns so that adding `torch.library.triton_op` + `register_fake` + `aoti_compile_and_package` later becomes mechanical. **Non-goal**: actually wiring AOTI yet. The refactor itself stays on torch ≥ 2.4 (current pin in [retrieve/pyproject.toml](../../retrieve/pyproject.toml): `torch>=2.4,<3`, `triton>=3.0`) and does not require the libtorch / AOTI bump.

### Status note: no current export entry point

The previous version of this plan referenced `evaluation/retrieval/build_export.py` and its `IndexWrapper` / `AttrIndexWrapper` adapters. **That file no longer exists.** The current `evaluation/retrieval/` directory contains the Yambda benchmark (`benchmark.py`), the dataset eval entry points, and an algorithm registry ([registry.py](../../evaluation/retrieval/algo_registry.py)) that constructs layers directly via `build_silvertorch` / `LiNR_V*_Triton(...)` / `FullScanKNN(...)`. There is no `torch.export` consumer in-tree today.

This changes the refactor's framing slightly: the bar is no longer "no regression in export status." It is now **"after the refactor, a fresh `evaluation/retrieval/build_export.py` can be authored that calls `torch.export.export()` per algorithm-mode and gets a clean `.pt2` per entry."** A scaffold for that file is part of Phase 3's scope (for `SilverTorch`); subsequent phases extend it (one mode per `.pt2`). The actual AOTI compile + package wiring stays out of scope.

End-state, after all 6 phases:

- Every Triton kernel has a **pure-launch core** with non-Optional tensor args and Python-int/bool scalars only — the function shape `triton_op` will eventually wrap.
- Every kernel's block-size / warp / stage parameters come from a **per-kernel `Config` dataclass + REGISTRY** the layer threads in. No `@triton.autotune`.
- Every retrieval layer exposes **mode-pinned forward signatures** — Optional combinatorics collapse into either a constructor `mode=` flag or distinct subclasses with a build factory.
- Device routing (`is_cuda` branches in `BloomFilter` / `ClauseIndex`) lives in build factories, not in `forward`.
- Post-launch host code (topk, gather, pad-to-K) lives in the layer, not in the kernel host wrapper.
- No `.item()` on any export-reachable code path.
- A standalone tuning script at `evaluation/scripts/tune_kernels.py` (renamed/extended from the existing [bench_filter_kernels.py](../../evaluation/scripts/bench_filter_kernels.py)) benchmarks each kernel's config grid on the local GPU; results are pasted into the in-code REGISTRY.
- A new `evaluation/retrieval/build_export.py` exists and produces one `.pt2` per (algo, mode) entry. AOTI wiring is the follow-up.

## Design principles (apply to all phases)

1. **Trace-boundary API is non-Optional, scalar-typed.** Kernel host functions take only `Tensor` (always present, dummies allowed) + `int` / `bool`. No `Tensor | None`, no kwargs that flip code paths.
2. **Config is plain Python data, frozen at export time.** A `@dataclass(frozen=True)` per kernel, looked up once in the layer's `__init__` (not in the host wrapper, not in `forward`), stored as a Python attribute on the layer. At trace time it's a graph constant.
3. **One forward signature per export entry.** Each Optional combination becomes a construction-time choice (`mode=` flag) or a sibling method (`forward_candidates`). Each becomes its own `.pt2`.
4. **Post-launch host code (topk, gather, pad-to-K) is in the layer.** The kernel produces a raw `[B, P]` score buffer or `[B, N]` mask; the layer composes `topk` / `gather` / pad. This isolates the `min(k, p)` / `actual_k < k` problem to one place per layer.
5. **`.item()` is banned on the export path.** Where it currently exists ([clause_compact.py:151](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L151), [bloom_compact.py:128](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L128), [v3_triton.py:52](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L52)), the fix is to propagate the `counts` tensor downstream rather than slicing or guarding. Consumers (`fused_masked_knn_topk`, `oporp_1bit_match_topk`) already accept `counts`. The early-return-on-empty-mask guard in `LiNR_V3_Triton` is replaced by always calling the kernel — the kernel already short-circuits per-row when `counts[b] == 0`.
6. **No deprecation shims.** Old kwarg-style host wrappers are deleted in the same PR that introduces the new core. Tests in [retrieve/tests/correctness/](../../retrieve/tests/correctness/) and [retrieve/tests/parity/](../../retrieve/tests/parity/) are updated in the same PR.
7. **Per-kernel block-size policy is decided during the phase**, not pre-mandated. Each phase picks bucketed-compile vs single-conservative based on observed shape distribution; document the choice in the PR description and in [docs/system/kernels.md](../../docs/system/kernels.md).
8. **Tests must pass before AND after the refactor**, with no relaxed tolerances. Parity tests in particular (Triton-vs-torch reference) must continue to assert bitwise-or-near-bitwise equivalence.
9. **Builder factories follow the existing naming convention.** The repo already has `build_silvertorch`, `build_linr_v1`, `build_linr_v1_triton`, `build_linr_v2[_triton]`, `build_linr_v3[_triton]`, `build_linr_index` — new factories use the same `build_xxx` shape and live alongside their class definitions ([linr/v1.py](../../retrieve/src/retrieve/layers/linr/v1.py), [silvertorch/main.py:202](../../retrieve/src/retrieve/layers/silvertorch/main.py#L202)).

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

The `problem_hint` is whatever scalar best characterizes the workload size (e.g., `P` for IVF, `N*W` for bloom, `B*N*D` for matmul-topk). Each phase picks one and documents it.

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

The launch function returns `None` and mutates outputs in place. The layer allocates `out_scores`, calls `_launch`, then runs topk/gather/pad in pure torch.

### Build-factory shape (for layers with device routing)

```python
def build_foo_filter(
    item_clause_attrs: Tensor,
    *,
    device: torch.device | None = None,
    kernel_config: FooKernelConfig | None = None,
) -> FooFilterTorch | FooFilterTriton:
    device = device or item_clause_attrs.device
    if device.type == "cuda":
        cfg = kernel_config or lookup(device, problem_hint=item_clause_attrs.shape[0])
        layer = FooFilterTriton(kernel_config=cfg)
    else:
        layer = FooFilterTorch()
    layer.register_index(item_clause_attrs)
    return layer.to(device).eval()
```

### Mode-flag forward shape

```python
class FooLayer(RetrievalModule):
    def __init__(self, *, mode: Literal["full", "filtered"], k: int, kernel_config=None):
        super().__init__()
        if mode not in {"full", "filtered"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.k = k
        self._cfg = kernel_config  # resolved in register_index when device is known

    def forward(self, query: Tensor, *extra: Tensor) -> tuple[Tensor, Tensor]:
        # exactly one signature per mode; Optionals do not appear here
        if self.mode == "full":
            return self._forward_full(query)
        # mode == "filtered" — caller supplies attrs in *extra
        return self._forward_filtered(query, *extra)
```

The `if self.mode == "full"` branch is on a Python attribute set in `__init__` — it specializes at export time (one branch per `.pt2`). This is the only kind of `if` allowed in `forward`.

---

## Notes for executing agents

Read this before starting any phase. These are cross-cutting facts and gotchas that don't fit a single phase but each phase agent needs to know.

### Phase dependencies

The phases are ordered numerically, but the real dependency graph is:

```
Phase 1 ────────► Phase 2 ─────────► Phase 3   (device factory pattern; Phase 3 also depends on Phase 2 BloomFilter split)
   │
   └──────────────────────────────► Phase 5   (counts-propagation API)
   │
   └──────────────────────────────► Phase 6   (counts-propagation API)
```

Phase 1 has no in-refactor dependencies and can be started independently. Phases 2, 3, 5, 6 must wait on their listed predecessors. Phase 3 has the most dependencies and the largest blast radius — that ordering is deliberate. **There is no Phase 4: the deleted Triton kernel for V1 means LiNR_V1 has no kernel to refactor.** [v1_triton.py](../../retrieve/src/retrieve/layers/linr/v1_triton.py) is a documented pure-torch alias kept only so the `triton_knn` benchmark algorithm and `build_linr_index(version=1, backend="triton")` continue to resolve. It does not participate in this refactor.

### Repo basics

- **Repo layout note**: the repo root is `/workspace/retrieve/`; the Python package lives at `/workspace/retrieve/retrieve/` (yes, nested). The eval app is its own package at `/workspace/retrieve/evaluation/`. Plan paths are written relative to `/workspace/retrieve/docs/plans/`, so `../../retrieve/src/retrieve/...` resolves to the package source.
- **Test runner**: `cd retrieve && uv run pytest tests/` (or scope to `tests/correctness/test_X.py`). Both projects use `uv`; do not invoke `pip` or raw `pytest`.
- **Eval baseline runner**: `cd evaluation && uv run evaluate --config conf/<yaml>` (the new shape — algorithms come from a YAML config, not `--algo`). To override algorithms on the fly: `--algorithms <name>`. Valid algorithm names live in [evaluation/retrieval/algo_registry.py:46](../../evaluation/retrieval/algo_registry.py#L46): `torch_fullscan`, `triton_knn`, `linr_v3_then_v2`, `silvertorch`, `voyager_hnsw`. **There is no `--algo linr_v2` or `--algo linr_v3`** any more — V2 and V3 are exercised end-to-end via the `linr_v3_then_v2` cascade. Run benchmarks before AND after the refactor for the affected algorithm and diff recall@k metrics.
- **No CI**. Locally-green is the entire bar. There is no `.github/workflows/` directory in this repo, so a phase PR cannot rely on CI to catch regressions — run the full `tests/` suite, not just the touched test files.
- **Triton ≥ 3.0, torch ≥ 2.4 (capped <3)** (see [retrieve/pyproject.toml](../../retrieve/pyproject.toml)). Do not rely on torch 2.5+ features (e.g. `torch.library.triton_op`) in this refactor — those land in a later AOTI-wiring effort.

### Existing-export status caveat

There is **no `evaluation/retrieval/build_export.py`** today, and no `IndexWrapper` / `AttrIndexWrapper`. The previous version of this plan assumed they existed. They do not. This means:

- There is no pre-existing export-success baseline to preserve. The bar is "after the refactor, an export wrapper can be authored cleanly," not "no regression in export."
- Phase 3 owns creation of a new `evaluation/retrieval/build_export.py` scaffold (just for `SilverTorch` initially). Phases 5 and 6 extend it for LiNR variants. The scaffold calls `torch.export.export()` per (algo, mode); it does not call `aoti_compile_and_package` (that's the next, separate effort).
- The end-state grep over `IndexWrapper` / `AttrIndexWrapper` in the original plan is moot — those names never need to appear.

### Kernel-internal quirks worth knowing

These are intentional; do not "fix" them while refactoring:

- **`clause_compact` has no `@triton.autotune`** because `tl.atomic_add` ([clause_compact.py:91](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L91)) corrupts state across autotune trials (each trial would see partially-mutated buffers). The fixed `BLOCK_N=256, num_warps=4` is correct. Phase 1 must NOT add autotune; the REGISTRY has one row. **Same gotcha applies to `bloom_compact`** ([bloom_compact.py:71](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L71)): also no autotune, also fixed `BLOCK_N=256, num_warps=4`, same atomic_add tail.
- **`clause_mask` has no autotune** but it *could* — atomics are absent, so a future autotune pass is safe. The fixed `_BLOCK_N=256, _NUM_WARPS=4` was chosen to match `clause_compact`'s body. Phase 1 should treat its REGISTRY the same way as `clause_compact` (one row) and only widen the grid if the tune script shows benefit.
- **`oporp_1bit_match_topk` has a custom `_popcount_int64`** bit-twiddle helper ([oporp_1bit_match_topk.py:10-19](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L10-L19)) instead of using libdevice. This is portability across Triton versions — leave it alone.
- **`bloom_match` picks BLOCK from input shape** (`block_n = 128 if n >= 128 else triton.next_power_of_2(n)` at [bloom_match.py:68](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py#L68)) because Triton requires `BLOCK` to be a power of two ≥ the working size. Phase 2 must address this via bucketed-compile (see "Design principles" #7), not by stripping the shape-dependent logic. **`bloom_compact` does *not* have this quirk** — it uses fixed `BLOCK_N=256` and tail-masks via `n_valid`; Phase 2's bucket policy applies to `bloom_match` only.
- **`fused_masked_knn_topk` has an `if p == 0` early return** in the host wrapper ([fused_masked_knn_topk.py:104-108](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L104-L108)). Phase 5 removes this — the kernel already handles `count == 0` per row (line 51-52 reads `count = tl.load(counts_ptr + bid)` and bounds-checks `n_offsets < count`). The `p == 0` shortcut exists only to skip the launch when *every* row is empty; the export-friendly version always launches and lets per-row guards do their job.

### `interfaces.py` contract decision

[retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) defines **three** abstract base classes today: `FilterModule`, `RetrievalModule`, and `ScorerModule` (the latter is consumed by `DotProductScorer` in [layers/utils/scorers.py](../../retrieve/src/retrieve/layers/utils/scorers.py)). None of them should add `mode` to their abstract contract. Modes are subclass-specific (e.g., `SilverTorch` will get `ivf_only` / `ivf_bloom`; `LiNR_V3_Triton` will get `full` / `masked` / `candidates`); forcing a uniform `mode` in the base would either over-constrain or be too vague to be useful. Each subclass declares its own `mode: Literal[...]` in `__init__`. The bases stay as-is.

The one exception: if any phase needs `forward_candidates` to be part of `RetrievalModule`'s declared interface (so type-checkers and downstream consumers can rely on it), add it there. If only some subclasses have it, leave it as a subclass method.

`FilterModule` already provides default implementations of `evaluate_indices` (via `compact_mask(self.evaluate_mask(...))`) and `evaluate_subset` (via gather). The Torch fallback split classes Phases 1–2 introduce should rely on those defaults rather than re-implementing them, except where a more efficient torch path exists.

### When you finish a phase

In addition to that phase's verification block, run from repo root:

```bash
cd retrieve && uv run pytest tests/ -v   # full suite, not just the phase's tests
```

If anything outside your phase's intended surface area is now red, you have a regression — investigate before merging. The full-suite run is the only safety net since there is no CI.

---

## Phase 1 — `clause_compact` + `clause_mask` + `ClauseIndex` (prototype, establishes conventions)

**Why first**: introduces the `.item()`-removal pattern via `counts` propagation, which Phases 3, 5, and 6 reuse. Also introduces the device-routing-via-build-factory pattern that Phases 2 and 3 reuse, and the shared `KernelConfig` + REGISTRY + `lookup` shape that every later phase clones.

`ClauseIndex` today routes BOTH `evaluate_indices` (→ `clause_compact`) AND `evaluate_mask` (→ `clause_mask`) on `is_cuda` ([clause.py:43](../../retrieve/src/retrieve/layers/filters/clause.py#L43), [clause.py:69](../../retrieve/src/retrieve/layers/filters/clause.py#L69)). Both kernels are part of this phase — they share inner-loop structure (`clause_mask` is `clause_compact` minus the cumsum + atomic_add tail), so refactoring them together avoids re-deriving the same conventions twice.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py)
- [retrieve/src/retrieve/kernels/triton/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py)
- [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py)
- [retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) — `combine_indices` consumer (it uses the full-width `(ids, counts)` API today via `evaluate_indices` + `evaluate_subset`; verify the cascade still works after the API change)
- [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) — `compact_mask` is the torch-side fallback for `clause_compact`; verify the `(ids, counts)` shape still matches between the Triton path and `compact_mask` after the change (today they both return full-width-or-trimmed; standardize on full-width per Step 1 below)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py) — re-exports
- [retrieve/src/retrieve/layers/__init__.py](../../retrieve/src/retrieve/layers/__init__.py) — re-exports
- [retrieve/tests/correctness/test_filters.py](../../retrieve/tests/correctness/test_filters.py)
- [retrieve/tests/correctness/test_compact.py](../../retrieve/tests/correctness/test_compact.py)
- [retrieve/tests/correctness/test_combine_filters.py](../../retrieve/tests/correctness/test_combine_filters.py) — exercises `combine_indices`; will see API-shape change
- [retrieve/tests/parity/test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py)
- [retrieve/tests/parity/test_clause_mask.py](../../retrieve/tests/parity/test_clause_mask.py)
- `evaluation/scripts/tune_kernels.py` — **either rename + extend** [evaluation/scripts/bench_filter_kernels.py](../../evaluation/scripts/bench_filter_kernels.py) (which already benches `clause_mask` and `bloom_compact`) **or create a new file alongside it**. Recommended: rename + extend, so we have one tuning entry point. Phase 1 wires `tune_clause_compact` and `tune_clause_mask` into the resulting harness.
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md)

### Steps

1. **Kill `.item()`** at [clause_compact.py:151](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L151):
   - Old behavior: kernel writes `[B, N]` indices buffer, host slices `out_indices[:, :p]` where `p = max(int(counts.max().item()), 1)`.
   - New behavior: kernel writes the same `[B, N]` buffer; host returns `(out_indices, counts)` with full `N` width. Consumers honor `counts` to know how many entries are valid per row.

2. **Define `ClauseCompactConfig` and `ClauseMaskConfig`** with `block_n: int = 256, num_warps: int = 4` each. Neither kernel autotunes today; one REGISTRY entry per arch suffices (likely the same `BLOCK_N=256` everywhere). Two parallel dataclasses + REGISTRY + `lookup()` shapes — the kernels share inner-loop structure but have different output epilogues, so keep them as separate configs even if the values match initially. (`clause_mask` *could* eventually autotune since it has no atomics — Phase 1 doesn't need to, but the config split leaves room for that later without re-touching `clause_compact`.)

3. **Pure-launch cores**: replace the existing wrappers with
   - `_clause_compact_launch(item_clause_attrs, clause_is_reverse, query_clause_attrs, out_indices, counts, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_indices [B, N] int64` and `counts [B] int64` and passes them in; kernel mutates.
   - `_clause_mask_launch(item_clause_attrs, clause_is_reverse, query_clause_attrs, out_mask, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_mask [B, N] bool`; kernel mutates.

4. **Update consumers** of the old `clause_compact(...)` / `clause_mask(...)` API:
   - [retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py) — `combine_indices` already consumes full-width `(ids, counts)` (it explicitly handles `ids past counts[b]` as scratch; see the `safe_ids` line at [filters/__init__.py:57](../../retrieve/src/retrieve/layers/filters/__init__.py#L57)). Verify it still works once `evaluate_indices` returns *truly* full-width buffers instead of `[:, :p]`-sliced ones. The internal `int(new_counts.max().item())` slice at [filters/__init__.py:61](../../retrieve/src/retrieve/layers/filters/__init__.py#L61) is fine for now (combine_indices itself is not on the export path), but if any consumer wants to call it from a traced graph, this needs the same full-width treatment in a follow-up.
   - [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) — `compact_mask` similarly uses `int(counts.max().item())` to pick output width. Decide whether the `ClauseIndexTorch.evaluate_indices` path goes through the trimmed-width `compact_mask` or returns full-width; consistency with the Triton path argues for full-width. Either change `compact_mask` to take a `full_width: bool = False` flag, or sidestep it in `ClauseIndexTorch` by manually composing the full-width result.
   - [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py) — `evaluate_indices` returns `(indices [B, N], counts [B])` instead of `(indices [B, p], counts [B])`. Update the docstring.
   - Any `fused_masked_knn_topk`, `oporp_1bit_match_topk` call sites that consume these — verify they accept full-width indices keyed by `counts`. Both kernels iterate over `counts[b]` per row already.

5. **Split `ClauseIndex`** at [clause.py](../../retrieve/src/retrieve/layers/filters/clause.py):
   - Rename current `ClauseIndex` body's torch fallback path → `ClauseIndexTorch(FilterModule)`. `evaluate_mask` is the pure-torch broadcast; `evaluate_indices` falls back to `compact_mask(self.evaluate_mask(...))` (full-width variant).
   - Extract the Triton-using path → `ClauseIndexTriton(FilterModule)`. Constructor takes `compact_config: ClauseCompactConfig | None = None, mask_config: ClauseMaskConfig | None = None`. `evaluate_mask` calls `_clause_mask_launch`; `evaluate_indices` calls `_clause_compact_launch`.
   - Both implement the same `FilterModule` interface (`register_index`, `evaluate_mask`, `evaluate_indices`, `evaluate_subset`).
   - **Drop both `is_cuda` branches from each method** — there are now two branches, one in `evaluate_mask` ([clause.py:43](../../retrieve/src/retrieve/layers/filters/clause.py#L43)) and one in `evaluate_indices` ([clause.py:69](../../retrieve/src/retrieve/layers/filters/clause.py#L69)). `ClauseIndexTorch` does only the torch implementation for both; `ClauseIndexTriton` does only the Triton implementation for both.

6. **Add build factory** `build_clause_index(item_clause_attrs, clause_is_reverse=None, *, device=None, compact_config=None, mask_config=None) -> ClauseIndexTorch | ClauseIndexTriton`:
   - If `device.type == "cuda"`: return `ClauseIndexTriton`.
   - Else: return `ClauseIndexTorch`.
   - Both call `register_index` before returning.
   - Place it next to the class definitions in `clause.py` (matches the in-file builder pattern used by `silvertorch/main.py` and `linr/v*.py`).

7. **Update [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)**: add `ClauseIndexTorch`, `ClauseIndexTriton`, `build_clause_index`. The current `ClauseIndex` re-export ([__init__.py:4](../../retrieve/src/retrieve/__init__.py#L4)) can be deleted in this PR since the only consumers are tests under `retrieve/tests/` and the eval app — both updated in the same PR. (No external API consumers documented.)

8. **Update tests**:
   - [test_filters.py](../../retrieve/tests/correctness/test_filters.py), [test_compact.py](../../retrieve/tests/correctness/test_compact.py): instantiate via the build factory; assertions over `(indices, counts)` use full-width indices.
   - [test_combine_filters.py](../../retrieve/tests/correctness/test_combine_filters.py): exercises the `combine_indices` cascade; should still pass without changes once the cascade gets full-width inputs (it already handles them).
   - [test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py): import `_clause_compact_launch` directly, allocate `out_indices` and `counts` in the test, assert against torch reference.
   - [test_clause_mask.py](../../retrieve/tests/parity/test_clause_mask.py): import `_clause_mask_launch` directly, allocate `out_mask` in the test, assert against torch broadcast reference.

9. **Stand up `tune_kernels.py`** (rename of `bench_filter_kernels.py` if going that route). Add `tune_clause_compact` and `tune_clause_mask` entries that emit the REGISTRY-pasteable rows. Keep the existing `bench_filter_kernels.py` shape (Click CLI, `triton.testing.do_bench`, per-arch JSON output).

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_filters.py tests/correctness/test_compact.py tests/correctness/test_combine_filters.py tests/parity/test_clause_compact.py tests/parity/test_clause_mask.py -v
grep -rn "\.item()" retrieve/src/retrieve/layers/filters/clause.py retrieve/src/retrieve/kernels/triton/filters/clause_compact.py retrieve/src/retrieve/kernels/triton/filters/clause_mask.py
grep -rn "is_cuda" retrieve/src/retrieve/layers/filters/clause.py
# All three greps should have zero hits in forward / evaluate_* paths.
```

### Acceptance criteria

- [ ] No `.item()` calls in [clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py), [clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py), or [clause.py](../../retrieve/src/retrieve/layers/filters/clause.py).
- [ ] No `is_cuda` checks in `ClauseIndexTorch` / `ClauseIndexTriton` methods (both `evaluate_mask` and `evaluate_indices` had branches that need removing).
- [ ] `build_clause_index` factory exists and returns the correct subclass per device.
- [ ] Downstream consumers (`combine_indices`, `compact_mask`) updated or verified to consume full-width indices + counts. Eager-mode behavior unchanged.
- [ ] All correctness + parity tests pass.

---

## Phase 2 — `bloom_match` + `bloom_compact` + `BloomFilter`

`BloomFilter` today routes BOTH `evaluate_mask` (→ `bloom_match`) AND `evaluate_indices` (→ `bloom_compact`) on `is_cuda` ([bloom.py:64](../../retrieve/src/retrieve/layers/filters/bloom.py#L64), [bloom.py:80](../../retrieve/src/retrieve/layers/filters/bloom.py#L80)). Both kernels are part of this phase. They share inner structure (`bloom_compact` is `bloom_match`'s subset-test inner loop + `clause_compact`'s cumsum + atomic_add tail), so the bucket policy decided for `bloom_match` directly informs whether `bloom_compact` needs the same bucket family.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py)
- [retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py)
- [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)
- [retrieve/src/retrieve/layers/__init__.py](../../retrieve/src/retrieve/layers/__init__.py)
- [retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py)
- [retrieve/tests/correctness/test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py)
- [retrieve/tests/parity/test_bloom_match.py](../../retrieve/tests/parity/test_bloom_match.py)
- [retrieve/tests/parity/test_bloom_compact.py](../../retrieve/tests/parity/test_bloom_compact.py)
- `evaluation/scripts/tune_kernels.py`
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md)

### Steps

1. **Pre-flight check**: verify `_build_signatures` / `_generate_seeds` / `_build_query_sigs` (host helpers in [bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) using `scatter_` into `[B, m_bits+1]` bool grids) trace cleanly under `torch.export`. Run a one-off `torch.export.export(BloomFilterTriton(...), (example_query_attrs,))` and inspect for tracer-incompatible ops. **If they don't trace**, this phase grows to add a query-side bloom-build kernel; document and re-scope before proceeding. Estimated 50/50 odds based on the `scatter_` usage. Note that `SilverTorch.forward` ([main.py:160](../../retrieve/src/retrieve/layers/silvertorch/main.py#L160)) reuses `_build_signatures` for the per-call query bloom — Phase 3 inherits whatever decision Phase 2 makes here.

2. **Define `BloomMatchConfig` and `BloomCompactConfig`**, each with `block_n: int, num_warps: int, num_stages: int = 3`. The current `bloom_match` kernel picks `BLOCK_N` from input shape (`128 if n >= 128 else triton.next_power_of_2(n)` at [bloom_match.py:68](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py#L68)); decide bucket strategy. Recommend **bucketed-compile**: 3-4 buckets `BLOCK_N ∈ {16, 64, 128, 256}` selected by a small Python if-ladder in the layer. Each bucket exports as one variant; one `.pt2` per bucket per arch. Document. **`bloom_compact` does *not* need bucketing** — it uses fixed `BLOCK_N=256` and tail-masks via `n_valid` (same shape as `clause_compact`); one REGISTRY row per arch suffices, and it must NOT autotune for the same atomic_add reason as `clause_compact`.

3. **Pure-launch cores**:
   - `_bloom_match_launch(qb, sigs, out_mask, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_mask [B, N] bool`. Kernel mutates.
   - `_bloom_compact_launch(qb, sigs, out_indices, counts, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_indices [B, N] int64` and `counts [B] int64` zeros. Kernel mutates. Same full-width-indices + counts-propagation policy as `_clause_compact_launch` from Phase 1 — there must be no `.item()` slice (today at [bloom_compact.py:128](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L128)), downstream consumers honor `counts` per row.

4. **Split `BloomFilter`** into `BloomFilterTorch` (torch-only fallback: `evaluate_mask` is `(qb.unsqueeze(1) & sigs) == qb` then `.all(dim=-1)` per the current CPU branch at [bloom.py:67](../../retrieve/src/retrieve/layers/filters/bloom.py#L67); `evaluate_indices` is `compact_mask(self.evaluate_mask(...))` — full-width form) and `BloomFilterTriton` (CUDA + both kernels). Both implement `FilterModule`. **Drop both `is_cuda` branches** — `evaluate_mask` (routes to `bloom_match`) and `evaluate_indices` (routes to `bloom_compact`).

5. **Build factory**: `build_bloom_filter(item_clause_attrs, *, m_bits, k_hash, device=None, match_config=None, compact_config=None)`. Place next to the class in `bloom.py`.

6. **Bucket dispatcher**: in `BloomFilterTriton.evaluate_mask`, the `bloom_match` bucket selection (`next_pow2(n)` clamped) is a Python `if` ladder on `n` (which is a SymInt under dynamic export). For export, the layer commits to one bucket per export — passed at construction time as `match_config`. The runtime dispatcher (in eager mode) picks the bucket from input `n`; the export trace specializes to one bucket via the constructor arg. `bloom_compact` has no bucket — `compact_config` is a single fixed config.

7. **Update tests**: `test_bloom_filter.py`, `test_bloom_match.py`, and `test_bloom_compact.py` use the factory + the new launch signatures. The `_build_signatures` helper unit tests should remain unchanged.

8. **Tune script**: add `tune_bloom_match` (per bucket) and `tune_bloom_compact` (single row, parallels `tune_clause_compact`).

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_bloom_filter.py tests/parity/test_bloom_match.py tests/parity/test_bloom_compact.py -v
grep -rn "is_cuda" retrieve/src/retrieve/layers/filters/bloom.py
grep -rn "\.item()" retrieve/src/retrieve/layers/filters/bloom.py retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py
# Both greps should be zero hits.
# Verify export of one BloomFilterTriton instance works (smoke):
uv run python -c "import torch; from retrieve.layers.filters.bloom import build_bloom_filter, BloomFilterTriton; ..."
```

### Acceptance criteria

- [ ] `BloomFilterTorch` and `BloomFilterTriton` exist; old `BloomFilter` either deleted or aliased.
- [ ] `build_bloom_filter` factory exists.
- [ ] No `is_cuda` checks in the layer's forward path (both `evaluate_mask` and `evaluate_indices` had branches).
- [ ] No `.item()` calls in `bloom_compact.py` or `bloom.py`.
- [ ] Bucket policy chosen and documented for `bloom_match`; `bloom_compact` documented as fixed single-config.
- [ ] Tests pass.

---

## Phase 3 — `codesigned_probe_score` + `SilverTorch`

**Largest blast radius**: 3 Optionals (`query_bits`, `bloom_sigs`, `mask`) inside the kernel host wrapper, plus the `candidate_ids` branch and the `query_clause_attrs is not None` / `mask is not None` triple in `SilverTorch.forward`. Save until Phases 1–2 have settled the conventions.

The current `SilverTorch.forward` signature is:

```python
forward(query, query_clause_attrs=None, mask=None, candidate_ids=None) -> tuple[Tensor, Tensor]
```

Three Optionals. The bloom kernel (`codesigned_probe_score`) takes another three Optionals (`query_bits`, `bloom_sigs`, `mask`) as kwargs. The `candidate_ids` branch dispatches to `_forward_candidates` ([main.py:140-141](../../retrieve/src/retrieve/layers/silvertorch/main.py#L140-L141)), which has its own `min(self.k, scores.shape[1])` variable-K topk.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py)
- [retrieve/src/retrieve/layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)
- [retrieve/tests/correctness/test_silvertorch.py](../../retrieve/tests/correctness/test_silvertorch.py)
- [retrieve/tests/parity/test_codesigned_probe_score.py](../../retrieve/tests/parity/test_codesigned_probe_score.py)
- `evaluation/scripts/tune_kernels.py`
- `evaluation/retrieval/build_export.py` — **create new** (does not exist today). Phase 3 owns the scaffold. See "Status note" above.
- [evaluation/retrieval/algo_registry.py](../../evaluation/retrieval/algo_registry.py) — `silvertorch` builder gets a `mode=` kwarg pass-through; the cascade entry doesn't change
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/architecture.md](../../docs/system/architecture.md)

### Steps

1. **Strip `@triton.autotune`** from [codesigned_probe_score.py:23](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L23). The current key includes `HAS_QB, HAS_MASK` — these become construction-time mode flags, not autotune keys.

2. **Define `CodesignedProbeScoreConfig`**: `block_p, num_warps, num_stages`. `problem_hint = P (n_probe * max_cluster_size)`. REGISTRY.

3. **Pure-launch core**: `_codesigned_probe_score_launch(query, query_bits_or_dummy, flat_items, item_codes, item_scales, bloom_sigs_or_dummy, mask_or_dummy, out_scores, *, has_qb: bool, has_mask: bool, block_p, num_warps, num_stages) -> None`. Caller allocates dummies for unused tensors (size `1×1`); `has_qb`/`has_mask` become `tl.constexpr` flags inside the kernel. No Python `if x is None` in the launch. (The kernel already guards on `HAS_QB`/`HAS_MASK` constexpr flags internally; the Optional → dummy-tensor binding is what moves out of the launch site.)

4. **`SilverTorch.__init__` mode flag**:
   ```python
   class SilverTorch(RetrievalModule):
       def __init__(self, *, mode: Literal["ivf_only", "ivf_bloom"], ...):
           ...
           self.mode = mode
   ```
   Drop the `if self.has_bloom and query_clause_attrs is not None` branch in `forward` ([main.py:159](../../retrieve/src/retrieve/layers/silvertorch/main.py#L159)). `mode="ivf_only"` exports as one `.pt2` (signature: `forward(query, mask)`); `mode="ivf_bloom"` exports as another (signature: `forward(query, query_clause_attrs, mask)`). The `mask` parameter is independent of mode and remains in both signatures (today `mask` is honored independently regardless of bloom config — see the [main.py docstring at line 22](../../retrieve/src/retrieve/layers/silvertorch/main.py#L22)).

   Note: today `SilverTorch.__init__` already has bloom-optional construction (`m_bits`/`k_hash` default to `None`) — `has_bloom` ([main.py:46](../../retrieve/src/retrieve/layers/silvertorch/main.py#L46)) is the existing semantic equivalent of the `mode` flag. The refactor formalizes it into a `Literal["ivf_only", "ivf_bloom"]` for export specialization; the underlying allocation logic in `register_index` is already conditional and stays the same.

5. **Mask handling**: `mask` stays an Optional today across both bloom configs. To keep the trace clean, either:
   - Keep `mask` as an explicit *required* arg in the per-mode forward signatures (`forward(query, mask)` for ivf_only; `forward(query, query_clause_attrs, mask)` for ivf_bloom). Callers pass an all-True mask when no mask is desired. **Recommended.**
   - Or split `mode` into 4 (`ivf_only` × `{masked, unmasked}`, `ivf_bloom` × `{masked, unmasked}`). More `.pt2` files, but each is a tighter trace.

   Pick option 1 unless the all-True mask path measures noticeably slower than the no-mask path; if it does, fall back to option 2.

6. **Always-allocate buffers**: in `register_index` ([main.py:111-124](../../retrieve/src/retrieve/layers/silvertorch/main.py#L111-L124)), when `mode="ivf_bloom"`, populate `bloom_sigs` (zeros if `item_clause_attrs` is None — the existing `has_bloom` branch already does this at [main.py:113-114](../../retrieve/src/retrieve/layers/silvertorch/main.py#L113-L114)). When `mode="ivf_only"`, skip allocation entirely (also existing). The `mode` flag controls whether the kernel reads them, via `has_qb` constexpr.

7. **Drop `_forward_candidates` from `SilverTorch.forward`**. Either:
   - Add a `mode="candidates"` variant (separate export entry), or
   - Extract to a sibling `forward_candidates(self, query, candidate_ids)` method that's a separate export entry.

   Recommended: separate method, kept on the same class. Three export entries total: `forward(q, mask)` for `ivf_only`, `forward(q, attrs, mask)` for `ivf_bloom`, `forward_candidates(q, cand)` for either. The `min(self.k, scores.shape[1])` at [main.py:196](../../retrieve/src/retrieve/layers/silvertorch/main.py#L196) inside `_forward_candidates` needs the pad-to-K treatment from Step 8.

8. **Move post-launch topk + gather + pad to `SilverTorch.forward`**. Pad-to-K policy: build `flat_items` with at least K slots per query (allocate `n_probe × max_size ≥ K`); then `topk(scores, k)` always returns K columns. If P < K is genuinely possible for some configurations, fall back to topk over P then `torch.cat`-pad with `-1` / `-inf` to width K — produces a static `[B, K]` shape because K is a Python int.

9. **Update `build_silvertorch`** ([main.py:202-224](../../retrieve/src/retrieve/layers/silvertorch/main.py#L202-L224)) to require `mode` as a kwarg (or to derive it from `m_bits/k_hash` for backward-compat with existing call sites in [registry.py:105](../../evaluation/retrieval/algo_registry.py#L105)). The registry currently builds `SilverTorch` directly, not via the builder — update that call site to pass `mode="ivf_only"` (since Yambda has no item attributes; see the [registry comment block at line 16-20](../../evaluation/retrieval/algo_registry.py#L16-L20)).

10. **Create `evaluation/retrieval/build_export.py`**. Scaffold:
    ```python
    def export_silvertorch(
        checkpoint_path: Path,
        out_dir: Path,
        *,
        mode: Literal["ivf_only", "ivf_bloom", "candidates"],
        ...,
    ) -> Path:
        idx = build_silvertorch(item_embs, k=K, mode=mode, ...)
        if mode == "ivf_only":
            ep = torch.export.export(idx, (query, mask), dynamic_shapes=...)
        elif mode == "ivf_bloom":
            ep = torch.export.export(idx, (query, query_clause_attrs, mask), dynamic_shapes=...)
        elif mode == "candidates":
            ep = torch.export.export(idx.forward_candidates, (query, candidate_ids), dynamic_shapes=...)
        out = out_dir / f"silvertorch_{mode}.pt2"
        torch.export.save(ep, out)
        return out
    ```
    Plus a Click CLI surface to drive it. Minimal scope: just `silvertorch` for Phase 3; Phases 5–6 add `linr_v*` entries.

11. **Update tests**: split tests by mode where applicable; parametrize over `mode`. Add a smoke-test that the new `build_export.py` round-trips for each mode (export → save → load → run → compare to eager).

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_silvertorch.py tests/parity/test_codesigned_probe_score.py -v
cd evaluation && uv run python -m retrieval.build_export --algo silvertorch --mode ivf_only --checkpoint-dir <path> --out-dir <path>
# Should produce silvertorch_ivf_only.pt2.
# End-to-end recall@k baseline:
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms silvertorch
```

### Acceptance criteria

- [ ] No `@triton.autotune`. Config dataclass + REGISTRY in place.
- [ ] No `Optional[Tensor]` in pure-launch core.
- [ ] `SilverTorch.__init__` accepts `mode: Literal["ivf_only", "ivf_bloom"]`.
- [ ] `forward` and `forward_candidates` are separate methods with non-Optional signatures.
- [ ] New `evaluation/retrieval/build_export.py` produces a `.pt2` for each (mode) entry.
- [ ] All tests pass; `silvertorch` benchmark recall@k unchanged.

---

## Phase 5 — `fused_masked_knn_topk` + LiNR V2 / V2_Triton

Depends on Phase 1's `(indices, counts)` API stabilizing. (No Phase 4: V1 has no kernel — see Notes.)

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py)
- [retrieve/src/retrieve/layers/linr/v2.py](../../retrieve/src/retrieve/layers/linr/v2.py), [v2_triton.py](../../retrieve/src/retrieve/layers/linr/v2_triton.py)
- [retrieve/src/retrieve/layers/linr/builder.py](../../retrieve/src/retrieve/layers/linr/builder.py) — already exists; `build_linr_v2_triton` may need a `mode=` kwarg pass-through
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- [retrieve/tests/parity/test_fused_masked_knn_topk.py](../../retrieve/tests/parity/test_fused_masked_knn_topk.py)
- `evaluation/retrieval/build_export.py` — extend the Phase-3 scaffold with a `linr_v2` entry
- `evaluation/scripts/tune_kernels.py`

### Steps

1. **Strip autotune** (8 configs keyed on `P, D` at [fused_masked_knn_topk.py:23](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L23)). Define config + REGISTRY.

2. **Pure-launch core**: takes `(query, item_embs, positive_indices, counts, out_scores, out_ids)` — full-width indices from Phase 1, kernel iterates per `counts[b]` per row. Remove the `if p == 0` early return at [fused_masked_knn_topk.py:104-108](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L104-L108) — the per-row `count == 0` guard inside the kernel ([line 51-52](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L51-L52)) already handles the all-empty case correctly. The host-side shortcut exists only as a perf optimization for empty inputs and breaks export tracing.

3. **`LiNR_V2_Triton.__init__`**: add `mode: Literal["full", "candidates"]` (collapses the `candidate_ids is None` branch at [v2_triton.py:27](../../retrieve/src/retrieve/layers/linr/v2_triton.py#L27)). `mode="full"` delegates to `super()._forward_full()` (V2's pure-torch `q @ x.T`-then-topk path). `mode="candidates"` calls `_fused_masked_knn_topk_launch`.

4. **Pad-to-K policy**: same as Phase 3 (caller pre-pads or commits to ≥ K candidates). The current `LiNR_V2._forward_prefilter` ([v2.py:71](../../retrieve/src/retrieve/layers/linr/v2.py#L71)) computes `actual_k = min(self.k, p)` and pads if `actual_k < self.k`; replace with the static-shape pad-to-K from Phase 3.

5. **Drop the early `if p == 0` return** in `LiNR_V2_Triton.forward` ([v2_triton.py:30-35](../../retrieve/src/retrieve/layers/linr/v2_triton.py#L30-L35)) — same reasoning as Step 2.

6. **Tests**: parametrize over mode; the torch-only `LiNR_V2` reference is unchanged.

7. **Extend `build_export.py`**: add `linr_v2` entries for `mode="full"` and `mode="candidates"`.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py tests/parity/test_fused_masked_knn_topk.py -v -k v2
# Verify Phase 1's full-width indices flow end-to-end via the cascade:
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms linr_v3_then_v2
```

### Acceptance criteria

- [ ] No autotune; no `if p == 0` early return; no Optional args in launch.
- [ ] `LiNR_V2_Triton` has `mode` flag.
- [ ] End-to-end with `clause_compact` (Phase 1) → `fused_masked_knn_topk` works (exercised via `linr_v3_then_v2`).
- [ ] `build_export.py` produces `linr_v2_full.pt2` and `linr_v2_candidates.pt2`.

---

## Phase 6 — `oporp_1bit_match_topk` + LiNR V3 / V3_Triton

Last because it has the most Optional combinations and a `int(counts.max().item()) == 0` early-return guard in the layer that needs the same fix as Phase 1.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py)
- [retrieve/src/retrieve/layers/linr/v3.py](../../retrieve/src/retrieve/layers/linr/v3.py), [v3_triton.py](../../retrieve/src/retrieve/layers/linr/v3_triton.py)
- [retrieve/src/retrieve/layers/linr/builder.py](../../retrieve/src/retrieve/layers/linr/builder.py) — `build_linr_v3_triton` may need a `mode=` kwarg pass-through
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- [retrieve/tests/parity/test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)
- `evaluation/retrieval/build_export.py` — extend with `linr_v3` entries
- `evaluation/scripts/tune_kernels.py`

### Steps

1. **Strip autotune** (8 configs keyed on `N, W, HAS_INDICES` at [oporp_1bit_match_topk.py:36](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L36)). The `HAS_INDICES` key collapses into a constructor `mode` flag.

2. **Pure-launch core**: `_oporp_1bit_match_topk_launch(query_bits, item_bits, positive_indices_or_dummy, counts_or_dummy, out_scores, out_ids, *, has_indices, k, block_n, num_warps, num_stages)`. Dummies for absent inputs (kernel already has `HAS_INDICES` constexpr at [line 55](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L55) and dummy-tensor binding at [host wrapper lines 166-169](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L166-L169) — move the dummy-binding out of the launch site).

3. **`LiNR_V3_Triton.__init__`**: add `mode: Literal["full", "masked", "candidates"]`. Three modes collapse the (`mask is not None`, `candidate_ids is not None`) Optional matrix at [v3_triton.py:33-49](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L33-L49).

4. **Kill `int(counts.max().item()) == 0`** at [v3_triton.py:52](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L52). The early-return-with-sentinel-values guard ([lines 55-58](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L55-L58)) was a perf optimization for fully-masked inputs; the kernel's per-row `count == 0` guard already produces the same sentinel output. Remove the early return; always launch the kernel.

5. **Tests + tune script**.

6. **Extend `build_export.py`**: add `linr_v3_full.pt2`, `linr_v3_masked.pt2`, `linr_v3_candidates.pt2` entries.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py tests/parity/test_oporp_1bit_match_topk.py -v -k v3
grep -rn "\.item()" retrieve/src/retrieve/layers/linr/v3_triton.py retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py
# Zero hits.
```

### Acceptance criteria

- [ ] No autotune; three explicit modes.
- [ ] No `.item()` on the export path.
- [ ] All tests pass.

---

## Cross-cutting work (rides along with the phases, no separate PR)

1. **`evaluation/scripts/tune_kernels.py`** — start by **renaming and extending** the existing [evaluation/scripts/bench_filter_kernels.py](../../evaluation/scripts/bench_filter_kernels.py), which already times `bloom_compact` vs torch and `clause_mask` vs torch. Add a Click CLI with `--kernel <name>`, `--device cuda:0`, `--shape-grid path.json`. Each phase appends one or two `tune_<kernel>` functions. Uses `triton.testing.do_bench`. Output is per-arch JSON (informational) + console output formatted to paste into REGISTRY.

2. **`evaluation/retrieval/build_export.py`** is **created in Phase 3** and extended through Phases 5–6:
   - Phase 3: scaffolds the file; adds `silvertorch` entries (one per mode).
   - Phase 5: adds `linr_v2` entries.
   - Phase 6: adds `linr_v3` entries.
   - Single Click CLI; one `.pt2` per (algo, mode); no AOTI compile yet.
   - **Note**: `LiNR_V1` / `LiNR_V1_Triton` does not need an export entry — it has no Triton kernel, the trace is just `q @ x.T` followed by `topk`, which exports trivially. Add it only if downstream consumers want a uniform `.pt2` zoo.

3. **[docs/system/](../../docs/system/)** updated per phase: `kernels.md` for kernel-specific changes, `architecture.md` for layer API changes, `filtering.md` for the bloom/clause split, `testing.md` for the per-phase verification commands. All four files exist today.

4. **[retrieve/pyproject.toml](../../retrieve/pyproject.toml)** version pins stay (`torch>=2.4,<3`, `triton>=3.0`). The AOTI bump (`torch>=2.5`) is a separate later effort.

5. **[evaluation/retrieval/algo_registry.py](../../evaluation/retrieval/algo_registry.py)** call sites get adjusted as each phase lands:
   - Phase 1: `ClauseIndex` is no longer in the registry's direct construction path (the registry only builds `SilverTorch` / `LiNR_V*_Triton` / `FullScanKNN` / `VoyagerHNSW`), so this phase has no registry change.
   - Phase 2: same — `BloomFilter` is not registered as a top-level algo.
   - Phase 3: `silvertorch` builder gains `mode="ivf_only"` (Yambda has no attrs — see the [registry comment at lines 16-20](../../evaluation/retrieval/algo_registry.py#L16-L20)).
   - Phase 5/6: `linr_v3_then_v2` cascade construction stays the same shape; just thread `mode="full"` (V3) and `mode="candidates"` (V2) through.

## Files NOT in scope

- `FullScanKNN` ([retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py)), `DotProductScorer` ([retrieve/src/retrieve/layers/utils/scorers.py](../../retrieve/src/retrieve/layers/utils/scorers.py)) — no Triton, no autotune, minimal Optionals. `FullScanKNN` does have the same `(mask, candidate_ids)` Optional pattern as LiNR; retrofit in the same PR as its first export consumer if needed. `DotProductScorer` is a `ScorerModule` with a non-Optional candidate-set signature already.
- `KMeansTorch` ([retrieve/src/retrieve/layers/utils/kmeans.py](../../retrieve/src/retrieve/layers/utils/kmeans.py)), `quantize_int8`, `quantize_oporp_1bit` ([layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py)), `_build_signatures`, `_generate_seeds` — host helpers in `register_index`. Not on the export path.
- `compact_mask` ([retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py)) — uses `.item()` for output width, but it's a host utility and not part of any kernel launch. Phase 1 makes it produce full-width output for trace-compat; deeper rework not needed.
- `combine_indices` ([retrieve/src/retrieve/layers/filters/__init__.py](../../retrieve/src/retrieve/layers/filters/__init__.py)) — sparse-cascade helper that uses `.item()` for compaction width. Not on the export path itself; the consumers (filter cascades) live in offline pipelines, not in the per-call retrieval forward path.
- `LiNR_V1` / `LiNR_V1_Triton` ([linr/v1.py](../../retrieve/src/retrieve/layers/linr/v1.py), [linr/v1_triton.py](../../retrieve/src/retrieve/layers/linr/v1_triton.py)) — no Triton kernel; pure torch matmul + topk. Out of scope.
- [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) — base class signatures may need a `forward_candidates` declaration; treated as a small change inside the relevant phase's PR.

## End-state verification (after all 6 phases)

```bash
# 1. Full test suite green:
cd retrieve && uv run pytest tests/ -v

# 2. Eager-mode E2E unchanged for every algorithm in the registry:
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms torch_fullscan
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms triton_knn
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms linr_v3_then_v2
cd evaluation && uv run evaluate --config conf/<yaml> --algorithms silvertorch

# 3. Export round-trips for every (algo, mode):
cd evaluation && uv run python -m retrieval.build_export --algo silvertorch --mode ivf_only --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo silvertorch --mode ivf_bloom --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo silvertorch --mode candidates --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo linr_v2 --mode full --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo linr_v2 --mode candidates --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo linr_v3 --mode full --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo linr_v3 --mode masked --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export --algo linr_v3 --mode candidates --checkpoint-dir <path> --out-dir <path>

# 4. Sweeps clean across the codebase:
grep -rn "@triton.autotune" retrieve/src/retrieve/kernels/triton/  # zero hits
grep -rn "\.item()" retrieve/src/retrieve/kernels/triton/ retrieve/src/retrieve/layers/  # only register_index hits, none in forward / launch
grep -rn "Tensor | None\|Optional\[Tensor\]" retrieve/src/retrieve/kernels/triton/  # zero hits in launch signatures
grep -rn "is_cuda" retrieve/src/retrieve/layers/  # zero hits in forward / evaluate paths
```

After all phases, adding `triton_op` + `register_fake` to each kernel is a localized follow-up: each kernel's pure-launch core has the right shape; each layer's `forward` has the right shape; AOTI compile becomes the next, separate piece of work.
