# Concern-separation refactor: retrieve module → torch.export-ready

> **Document audience**: Each phase below is a self-contained brief intended to be executed by a separate SWE agent without access to this conversation's history. Phases are ordered and depend on prior phases for shared conventions; do not run later phases out of order, but each phase's PR is reviewed and merged independently.

## Context

The `retrieve` module ships 7 Triton kernels and ~9 `nn.Module` retrieval layers under [retrieve/src/retrieve/](../../retrieve/src/retrieve/). `evaluation/retrieval/build_export.py` already calls `torch.export.export()` on those layers via thin `IndexWrapper` / `AttrIndexWrapper` adapters, but the export is fragile:

- Kernels use `@triton.autotune` with lambda-meta grids (`grid=lambda meta: (b, triton.cdiv(p, meta["BLOCK_P"]))`), which inductor's compile-time autotuner cannot serialize portably.
- Host wrappers branch on `Optional[Tensor]` arguments and rebind to dummy `1×1` tensors keyed by `HAS_X: tl.constexpr` flags.
- Layer `forward` methods branch on `is_cuda`, `candidate_ids is not None`, `query_clause_attrs is not None`, `mask is not None`.
- Two host wrappers compute Python-int values via `.item()` and use them to slice tensors (`out_indices[:, :p]`).
- Several layers compute `min(self.k, scores.shape[1])` for variable-K topk, producing data-dependent output shapes.
- `SilverTorch.register_index` calls `int(cluster_sizes.max().item())` — fine because `register_index` is not on the export path, but the same pattern leaks into `forward` in `LiNR_V3_Triton`.

**Goal of this refactor**: separate concerns so that adding `torch.library.triton_op` + `register_fake` + `aoti_compile_and_package` later becomes mechanical. **Non-goal**: actually wiring AOTI yet. The refactor itself stays on torch ≥ 2.4 (current pin in [retrieve/pyproject.toml](../../retrieve/pyproject.toml)) and does not require the libtorch / AOTI bump.

End-state, after all 6 phases:

- Every Triton kernel has a **pure-launch core** with non-Optional tensor args and Python-int/bool scalars only — the function shape `triton_op` will eventually wrap.
- Every kernel's block-size / warp / stage parameters come from a **per-kernel `Config` dataclass + REGISTRY** the layer threads in. No `@triton.autotune`.
- Every retrieval layer exposes **mode-pinned forward signatures** — Optional combinatorics collapse into either a constructor `mode=` flag or distinct subclasses with a build factory.
- Device routing (`is_cuda` branches in `BloomFilter` / `ClauseIndex`) lives in build factories, not in `forward`.
- Post-launch host code (topk, gather, pad-to-K) lives in the layer, not in the kernel host wrapper.
- No `.item()` on any export-reachable code path.
- A standalone tuning script at `evaluation/scripts/tune_kernels.py` benchmarks each kernel's config grid on the local GPU; results are pasted into the in-code REGISTRY.

## Design principles (apply to all phases)

1. **Trace-boundary API is non-Optional, scalar-typed.** Kernel host functions take only `Tensor` (always present, dummies allowed) + `int` / `bool`. No `Tensor | None`, no kwargs that flip code paths.
2. **Config is plain Python data, frozen at export time.** A `@dataclass(frozen=True)` per kernel, looked up once in the layer's `__init__` (not in the host wrapper, not in `forward`), stored as a Python attribute on the layer. At trace time it's a graph constant.
3. **One forward signature per export entry.** Each Optional combination becomes a construction-time choice (`mode=` flag) or a sibling method (`forward_candidates`). Each becomes its own `.pt2`.
4. **Post-launch host code (topk, gather, pad-to-K) is in the layer.** The kernel produces a raw `[B, P]` score buffer or `[B, N]` mask; the layer composes `topk` / `gather` / pad. This isolates the `min(k, p)` / `actual_k < k` problem to one place per layer.
5. **`.item()` is banned on the export path.** Where it currently exists ([clause_compact.py:151](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L151), [v3_triton.py](../../retrieve/src/retrieve/layers/linr/v3_triton.py)), the fix is to propagate the `counts` tensor downstream rather than slicing. Consumers (`fused_masked_knn_topk`, `oporp_1bit_match_topk`) already accept `counts`.
6. **No deprecation shims.** Old kwarg-style host wrappers are deleted in the same PR that introduces the new core. Tests in [retrieve/tests/correctness/](../../retrieve/tests/correctness/) and [retrieve/tests/parity/](../../retrieve/tests/parity/) are updated in the same PR.
7. **Per-kernel block-size policy is decided during the phase**, not pre-mandated. Each phase picks bucketed-compile vs single-conservative based on observed shape distribution; document the choice in the PR description and in [docs/system/kernels.md](../../docs/system/kernels.md).
8. **Tests must pass before AND after the refactor**, with no relaxed tolerances. Parity tests in particular (Triton-vs-torch reference) must continue to assert bitwise-or-near-bitwise equivalence.

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

Phase 4 ──────────────────────────► Phase 5   (V2 mode="full" composes _fused_matmul_topk_launch from Phase 4)
```

Phases 1 and 4 have no in-refactor dependencies and can be started independently. Phases 2, 3, 5, 6 must wait on their listed predecessors. Phase 3 has the most dependencies and the largest blast radius — that ordering is deliberate.

### Repo basics

- **Test runner**: `cd retrieve && uv run pytest tests/` (or scope to `tests/correctness/test_X.py`). Both projects use `uv`; do not invoke `pip` or raw `pytest`.
- **Eval baseline runner**: `cd evaluation && uv run python -m retrieval.benchmark --algo <name> --k 100`. Run this before AND after the refactor for the affected algo and diff recall@k metrics.
- **No CI**. Locally-green is the entire bar. There is no GitHub Actions workflow in this repo, so a phase PR cannot rely on CI to catch regressions — run the full `tests/` suite, not just the touched test files.
- **Triton ≥ 3.0, torch ≥ 2.4** (see [retrieve/pyproject.toml](../../retrieve/pyproject.toml)). Do not rely on torch 2.5+ features (e.g. `torch.library.triton_op`) in this refactor — those land in a later AOTI-wiring effort.

### Existing-export status caveat

`evaluation/retrieval/build_export.py` calls `torch.export.export()` on each algo, but I have not verified that every algo's export currently succeeds end-to-end. It is plausible that one or more existing exports (especially `silvertorch` with attrs, or `linr_v3`) are already broken before this refactor begins. **If your phase's export was already broken pre-refactor, document that in the PR description; do not try to fix the pre-existing breakage as part of the refactor.** The refactor's bar is "no regression in export status," not "make export work where it didn't before."

### Kernel-internal quirks worth knowing

These are intentional; do not "fix" them while refactoring:

- **`clause_compact` has no `@triton.autotune`** because `tl.atomic_add` corrupts state across autotune trials (each trial would see partially-mutated buffers). The fixed `BLOCK_N=256, num_warps=4` is correct. Phase 1 must NOT add autotune; the REGISTRY has one row.
- **`oporp_1bit_match_topk` has a custom `_popcount_int64`** bit-twiddle helper instead of using libdevice. This is portability across Triton versions — leave it alone.
- **`bloom_match` picks BLOCK from input shape** (`128 if n >= 128 else next_power_of_2(n)`) because Triton requires `BLOCK` to be a power of two ≥ the working size. Phase 2 must address this via bucketed-compile (see "Design principles" #7), not by stripping the shape-dependent logic.
- **`int8_ann_fused.py` is parity-only dead code** — it has no production caller (the silvertorch path runs through `codesigned_probe_score`, see [architecture.md](../system/architecture.md)). It is *out of scope* for this refactor; whether to delete it is a separate cleanup decision.

### `interfaces.py` contract decision

`RetrievalModule` and `FilterModule` base classes ([retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py)) should NOT add `mode` to their abstract contract. Modes are subclass-specific (e.g., `SilverTorch` has `ivf_only` / `ivf_bloom`; `LiNR_V3_Triton` has `full` / `masked` / `candidates`); forcing a uniform `mode` in the base would either over-constrain or be too vague to be useful. Each subclass declares its own `mode: Literal[...]` in `__init__`. The base class stays as-is.

The one exception: if any phase needs `forward_candidates` to be part of `RetrievalModule`'s declared interface (so type-checkers and downstream consumers can rely on it), add it there. If only some subclasses have it, leave it as a subclass method.

### When you finish a phase

In addition to that phase's verification block, run from repo root:

```bash
cd retrieve && uv run pytest tests/ -v   # full suite, not just the phase's tests
```

If anything outside your phase's intended surface area is now red, you have a regression — investigate before merging. The full-suite run is the only safety net since there is no CI.

---

## Phase 1 — `clause_compact` + `ClauseIndex` (prototype, establishes conventions)

**Why first**: introduces the `.item()`-removal pattern via `counts` propagation, which Phases 3, 5, and 6 reuse. Also introduces the device-routing-via-build-factory pattern that Phases 2 and 3 reuse, and the shared `KernelConfig` + REGISTRY + `lookup` shape that every later phase clones.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py)
- [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py)
- [retrieve/src/retrieve/layers/utils/__init__.py](../../retrieve/src/retrieve/layers/utils/__init__.py) — `combine_indices` AND `combine_masks` consumers (verify both, the latter may need a parallel update)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py) — re-exports
- [retrieve/src/retrieve/layers/__init__.py](../../retrieve/src/retrieve/layers/__init__.py) — re-exports
- [retrieve/tests/correctness/test_filters.py](../../retrieve/tests/correctness/test_filters.py)
- [retrieve/tests/correctness/test_compact.py](../../retrieve/tests/correctness/test_compact.py)
- [retrieve/tests/parity/test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py)
- `evaluation/scripts/tune_kernels.py` — **create new**, add `tune_clause_compact`
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md)

### Steps

1. **Kill `.item()`** at [clause_compact.py:151](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L151):
   - Old behavior: kernel writes `[B, N]` indices buffer, host slices `out_indices[:, :p]` where `p = int(counts.max().item())`.
   - New behavior: kernel writes the same `[B, N]` buffer; host returns `(out_indices, counts)` with full `N` width. Consumers honor `counts` to know how many entries are valid per row.

2. **Define `ClauseCompactConfig`** with `block_n: int = 256, num_warps: int = 4`. The kernel does not autotune currently; one REGISTRY entry per arch suffices (likely the same `BLOCK_N=256` everywhere). Add the dataclass + REGISTRY + `lookup()` per shared shape.

3. **Pure-launch core**: replace the existing `clause_compact(...)` function with `_clause_compact_launch(item_clause_attrs, clause_is_reverse, query_clause_attrs, out_indices, counts, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_indices [B, N] int64` and `counts [B] int64` and passes them in; kernel mutates.

4. **Update consumers** of the old `clause_compact(...)` API:
   - [retrieve/src/retrieve/layers/utils/__init__.py](../../retrieve/src/retrieve/layers/utils/__init__.py) — `combine_indices`: update call sites to pass full-width `(indices, counts)` through, no slicing.
   - [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py) — `evaluate_indices` returns `(indices [B, N], counts [B])` instead of `(indices [B, p], counts [B])`. Update the docstring.
   - Any `fused_masked_knn_topk`, `oporp_1bit_match_topk` call sites that consume these — verify they accept full-width indices keyed by `counts`. (They should already; the kernels already iterate over `counts[b]` per row.)

5. **Split `ClauseIndex`** at [clause.py](../../retrieve/src/retrieve/layers/filters/clause.py):
   - Rename current `ClauseIndex` body's torch fallback path → `ClauseIndexTorch(FilterModule)`.
   - Extract the Triton-using path → `ClauseIndexTriton(FilterModule)`. Constructor takes `kernel_config: ClauseCompactConfig | None = None`.
   - Both implement the same `FilterModule` interface (`register_index`, `evaluate_mask`, `evaluate_indices`, `evaluate_subset`).
   - **Drop the `is_cuda` branch from each method** — `ClauseIndexTorch` does only the torch implementation; `ClauseIndexTriton` does only the Triton implementation.

6. **Add build factory** `build_clause_index(item_clause_attrs, clause_is_reverse=None, *, device=None, kernel_config=None) -> ClauseIndexTorch | ClauseIndexTriton`:
   - If `device.type == "cuda"`: return `ClauseIndexTriton`.
   - Else: return `ClauseIndexTorch`.
   - Both call `register_index` before returning.

7. **Update [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)**: add `ClauseIndexTorch`, `ClauseIndexTriton`, `build_clause_index`. Keep the old `ClauseIndex` export as `ClauseIndex = ClauseIndexTriton` alias **only if** there are external consumers to consider — otherwise delete the old name and update [evaluation/](../../evaluation/) call sites in this PR.

8. **Update tests**:
   - [test_filters.py](../../retrieve/tests/correctness/test_filters.py), [test_compact.py](../../retrieve/tests/correctness/test_compact.py): instantiate via the build factory; assertions over `(indices, counts)` use full-width indices.
   - [test_clause_compact.py](../../retrieve/tests/parity/test_clause_compact.py): import `_clause_compact_launch` directly, allocate `out_indices` and `counts` in the test, assert against torch reference.

9. **Add `tune_clause_compact`** to `tune_kernels.py`.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_filters.py tests/correctness/test_compact.py tests/parity/test_clause_compact.py -v
grep -rn "\.item()" retrieve/src/retrieve/layers/filters/clause.py retrieve/src/retrieve/kernels/triton/filters/clause_compact.py
grep -rn "is_cuda" retrieve/src/retrieve/layers/filters/clause.py
# All three should have zero hits in forward / evaluate_* paths.
```

### Acceptance criteria

- [ ] No `.item()` calls in [clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) or [clause.py](../../retrieve/src/retrieve/layers/filters/clause.py).
- [ ] No `is_cuda` checks in `ClauseIndexTorch` / `ClauseIndexTriton` methods.
- [ ] `build_clause_index` factory exists and returns the correct subclass per device.
- [ ] Downstream consumers (`combine_indices`, others) updated to consume full-width indices + counts. Eager-mode behavior unchanged.
- [ ] All correctness + parity tests pass.

---

## Phase 2 — `bloom_match` + `BloomFilter`

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py)
- [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)
- [retrieve/src/retrieve/layers/__init__.py](../../retrieve/src/retrieve/layers/__init__.py)
- [retrieve/tests/correctness/test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py)
- [retrieve/tests/parity/test_bloom_match.py](../../retrieve/tests/parity/test_bloom_match.py)
- `evaluation/scripts/tune_kernels.py`
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/filtering.md](../../docs/system/filtering.md)

### Steps

1. **Pre-flight check**: verify `_build_signatures` / `_build_query_sigs` (host helpers in [bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) using `scatter_` into `[B, m_bits+1]` bool grids) trace cleanly under `torch.export`. Run a one-off `torch.export.export(BloomFilterTriton(...), (example_query_attrs,))` and inspect for tracer-incompatible ops. **If they don't trace**, this phase grows to add a query-side bloom-build kernel; document and re-scope before proceeding. Estimated 50/50 odds based on the `scatter_` usage.

2. **Define `BloomMatchConfig`** with `block_n: int, num_warps: int, num_stages: int = 3`. The current kernel picks `BLOCK_N` from input shape (`128 if n >= 128 else next_power_of_2(n)`); decide bucket strategy. Recommend **bucketed-compile**: 3-4 buckets `BLOCK_N ∈ {16, 64, 128, 256}` selected by a small Python if-ladder in the layer. Each bucket exports as one variant; one `.pt2` per bucket per arch. Document.

3. **Pure-launch core**: `_bloom_match_launch(qb, sigs, out_mask, *, block_n, num_warps, num_stages) -> None`. Caller allocates `out_mask [B, N] bool`. Kernel mutates.

4. **Split `BloomFilter`** into `BloomFilterTorch` (torch-only fallback: `(qb.unsqueeze(1) & sigs) == qb` then `.all(dim=-1)`) and `BloomFilterTriton` (CUDA + kernel). Both implement `FilterModule`. Drop the `is_cuda` branch from each `evaluate_mask`.

5. **Build factory**: `build_bloom_filter(item_clause_attrs, *, m_bits, k_hash, device=None, kernel_config=None)`.

6. **Bucket dispatcher**: in `BloomFilterTriton.evaluate_mask`, the bucket selection (`next_pow2(n)` clamped) is a Python `if` ladder on `n` (which is a SymInt under dynamic export). For export, the layer commits to one bucket per export — passed at construction time as part of `kernel_config`. The runtime dispatcher (in eager mode) picks the bucket from input `n`; the export trace specializes to one bucket via the constructor arg.

7. **Update tests**: `test_bloom_filter.py` and `test_bloom_match.py` use the factory + the new launch signature. The `_build_signatures` helper unit tests should remain unchanged.

8. **Tune script**: add `tune_bloom_match` per kernel + per bucket.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_bloom_filter.py tests/parity/test_bloom_match.py -v
grep -rn "is_cuda" retrieve/src/retrieve/layers/filters/bloom.py
# Should be zero hits.
# Verify export of one BloomFilterTriton instance works (smoke):
uv run python -c "import torch; from retrieve.layers.filters.bloom import build_bloom_filter, BloomFilterTriton; ..."
```

### Acceptance criteria

- [ ] `BloomFilterTorch` and `BloomFilterTriton` exist; old `BloomFilter` either deleted or aliased.
- [ ] `build_bloom_filter` factory exists.
- [ ] No `is_cuda` checks in the layer's forward path.
- [ ] Bucket policy chosen and documented.
- [ ] Tests pass.

---

## Phase 3 — `codesigned_probe_score` + `SilverTorch`

**Largest blast radius**: 3 Optionals (`query_bits`, `bloom_sigs`, `mask`) + the `candidate_ids` branch. Save until Phases 1–2 have settled the conventions.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py)
- [retrieve/src/retrieve/layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py)
- [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)
- [retrieve/tests/correctness/test_silvertorch.py](../../retrieve/tests/correctness/test_silvertorch.py)
- [retrieve/tests/parity/test_codesigned_probe_score.py](../../retrieve/tests/parity/test_codesigned_probe_score.py)
- `evaluation/scripts/tune_kernels.py`
- `evaluation/retrieval/build_export.py` — `AttrIndexWrapper` may retire here
- [docs/system/kernels.md](../../docs/system/kernels.md), [docs/system/architecture.md](../../docs/system/architecture.md)

### Steps

1. **Strip `@triton.autotune`** from [codesigned_probe_score.py:23](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L23). The current key includes `HAS_QB, HAS_MASK` — these become construction-time mode flags, not autotune keys.

2. **Define `CodesignedProbeScoreConfig`**: `block_p, num_warps, num_stages`. `problem_hint = P (n_probe * max_cluster_size)`. REGISTRY.

3. **Pure-launch core**: `_codesigned_probe_score_launch(query, query_bits_or_dummy, flat_items, item_codes, item_scales, bloom_sigs_or_dummy, mask_or_dummy, out_scores, *, has_qb: bool, has_mask: bool, block_p, num_warps, num_stages) -> None`. Caller allocates dummies for unused tensors (size `1×1`); `has_qb`/`has_mask` become `tl.constexpr` flags inside the kernel. No Python `if x is None` in the launch.

4. **`SilverTorch.__init__` mode flag**:
   ```python
   class SilverTorch(RetrievalModule):
       def __init__(self, *, mode: Literal["ivf_only", "ivf_bloom"], ...):
           ...
           self.mode = mode
   ```
   Drop the `if query_clause_attrs is not None:` branch in `forward`. `mode="ivf_only"` exports as one `.pt2` (signature: `forward(query)`); `mode="ivf_bloom"` exports as another (signature: `forward(query, query_clause_attrs)`).

   Note: today `SilverTorch.__init__` already has bloom-optional construction (`m_bits`/`k_hash` default to `None`) — `has_bloom` is the existing semantic equivalent of the `mode` flag. The refactor formalizes it into a `Literal["ivf_only", "ivf_bloom"]` for export specialization; the underlying allocation logic in `register_index` is already conditional and stays the same.

5. **Always-allocate buffers**: in `register_index`, when `mode="ivf_bloom"`, populate `bloom_sigs` (zeros if `item_clause_attrs` is None — the existing `has_bloom` branch behavior). When `mode="ivf_only"`, skip allocation entirely (also existing). The `mode` flag controls whether the kernel reads them, via `has_qb` constexpr.

6. **Drop `_forward_candidates` from `SilverTorch.forward`**. Either:
   - Add a `mode="candidates"` variant (separate export entry), or
   - Extract to a sibling `forward_candidates(self, query, candidate_ids)` method that's a separate export entry.

   Recommended: separate method, kept on the same class. Three export entries total: `forward(q)` for `ivf_only`, `forward(q, attrs)` for `ivf_bloom`, `forward_candidates(q, cand)` for either.

7. **Move post-launch topk + gather + pad to `SilverTorch.forward`**. Pad-to-K policy: build `flat_items` with at least K slots per query (allocate `n_probe × max_size ≥ K`); then `topk(scores, k)` always returns K columns. If P < K is genuinely possible for some configurations, fall back to topk over P then `torch.cat`-pad with `-1` / `-inf` to width K — produces a static `[B, K]` shape because K is a Python int.

8. **Update `build_export.py`**: `AttrIndexWrapper` exists only because `SilverTorch.forward` had `Optional` args. Now retires. The export path becomes:
   ```python
   if mode == "ivf_only":
       ep = export(silvertorch, (query,), dynamic_shapes={"query": {0: batch}})
   elif mode == "ivf_bloom":
       ep = export(silvertorch, (query, query_clause_attrs), dynamic_shapes=...)
   ```

9. **Update tests**: split tests by mode where applicable; parametrize over `mode`.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_silvertorch.py tests/parity/test_codesigned_probe_score.py -v
cd evaluation && uv run python -m retrieval.build_export --checkpoint-dir <path> --index silvertorch --attrs-path <p>
# Should produce silvertorch_ivf_bloom.pt2 (or whatever the new naming is)
```

### Acceptance criteria

- [ ] No `@triton.autotune`. Config dataclass + REGISTRY in place.
- [ ] No `Optional[Tensor]` in pure-launch core.
- [ ] `SilverTorch.__init__` accepts `mode: Literal["ivf_only", "ivf_bloom"]`.
- [ ] `forward` and `forward_candidates` are separate methods with non-Optional signatures.
- [ ] `AttrIndexWrapper` retired in `build_export.py`.
- [ ] All tests pass; `benchmark --algo silvertorch` recall@k unchanged.

---

## Phase 4 — `fused_matmul_topk` + LiNR V1 / V1_Triton

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/fused_matmul_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_matmul_topk.py)
- [retrieve/src/retrieve/layers/linr/v1.py](../../retrieve/src/retrieve/layers/linr/v1.py), [v1_triton.py](../../retrieve/src/retrieve/layers/linr/v1_triton.py)
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)
- `evaluation/scripts/tune_kernels.py`

### Steps

1. **Strip `@triton.autotune`** (12 configs keyed on `B, N, D`). Define `FusedMatmulTopkConfig` (block_m, block_n, num_warps, num_stages). REGISTRY by arch + `B*N*D` problem_hint regime.

2. **Pure-launch core**: `_fused_matmul_topk_launch(query, item_embs, mask_or_dummy, out_scores, out_ids, *, has_mask, k, block_m, block_n, num_warps, num_stages) -> None`. Caller allocates output buffers of shape `[B, k]`.

3. **`LiNR_V1_Triton.__init__`**: add `mode: Literal["unmasked", "masked"]` and `kernel_config`. Drop the `mask is not None` branch.

4. **Tests**: parametrize over mode; `LiNR_V1` (torch-only reference) is unchanged.

5. **Tune script**: add `tune_fused_matmul_topk` (largest config grid — 12 configs × shape grid).

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py -v -k v1
grep -rn "@triton.autotune" retrieve/src/retrieve/kernels/triton/linr/fused_matmul_topk.py
# Zero hits.
```

### Acceptance criteria

- [ ] No autotune. Config + REGISTRY.
- [ ] `LiNR_V1_Triton` has `mode` flag, no Optional args in `forward`.
- [ ] Tests pass.

---

## Phase 5 — `fused_masked_knn_topk` + LiNR V2 / V2_Triton

Depends on Phase 1's `(indices, counts)` API stabilizing.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py)
- [retrieve/src/retrieve/layers/linr/v2.py](../../retrieve/src/retrieve/layers/linr/v2.py), [v2_triton.py](../../retrieve/src/retrieve/layers/linr/v2_triton.py)
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)

### Steps

1. **Strip autotune** (8 configs keyed on `P, D`). Define config + REGISTRY.

2. **Pure-launch core**: takes `(query, item_embs, positive_indices, counts, out_scores, out_ids)` — full-width indices from Phase 1, kernel iterates per `counts[b]` per row. No `if p == 0` early return — handled by `counts == 0` per-row at kernel level (it already is, mostly).

3. **`LiNR_V2_Triton.__init__`**: add `mode: Literal["full", "candidates"]` (collapses the `candidate_ids is not None` branch). `mode="full"` calls into `_fused_matmul_topk_launch` from Phase 4 — verify the cross-kernel composition works.

4. **Pad-to-K policy**: same as Phase 3 (caller pre-pads or commits to ≥ K candidates).

5. **Tests**: same pattern as Phase 4.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py -v -k v2
# Verify Phase 1's full-width indices flow end-to-end:
cd evaluation && uv run python -m retrieval.benchmark --algo linr_v2 --use-clauses
```

### Acceptance criteria

- [ ] No autotune; no `if p == 0` early return; no Optional args in launch.
- [ ] `LiNR_V2_Triton` has `mode` flag.
- [ ] End-to-end with `clause_compact` (Phase 1) → `fused_masked_knn_topk` works.

---

## Phase 6 — `oporp_1bit_match_topk` + LiNR V3 / V3_Triton

Last because it has the most Optional combinations and an `int(counts.max().item())` pattern in the layer that needs the same fix as Phase 1.

### Files

Modify:
- [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py)
- [retrieve/src/retrieve/layers/linr/v3.py](../../retrieve/src/retrieve/layers/linr/v3.py), [v3_triton.py](../../retrieve/src/retrieve/layers/linr/v3_triton.py)
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)

### Steps

1. **Strip autotune** (8 configs keyed on `N, W, HAS_INDICES`). The `HAS_INDICES` key collapses into a constructor `mode` flag.

2. **Pure-launch core**: `_oporp_1bit_match_topk_launch(query_bits, item_bits, positive_indices_or_dummy, counts_or_dummy, out_scores, out_ids, *, has_indices, k, block_n, num_warps, num_stages)`.

3. **`LiNR_V3_Triton.__init__`**: add `mode: Literal["full", "masked", "candidates"]`. Three modes collapse the (`mask is not None`, `candidate_ids is not None`) Optional matrix.

4. **Kill `int(counts.max().item())`** — pass `counts` straight through as Phase 1.

5. **Tests + tune script**.

### Verification

```bash
cd retrieve && uv run pytest tests/correctness/test_linr.py -v -k v3
grep -rn "\.item()" retrieve/src/retrieve/layers/linr/v3_triton.py retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py
# Zero hits.
```

### Acceptance criteria

- [ ] No autotune; three explicit modes.
- [ ] No `.item()` on the export path.
- [ ] All tests pass.

---

## Cross-cutting work (rides along with the phases, no separate PR)

1. **`evaluation/scripts/tune_kernels.py`** grows incrementally: Phase 1 creates the file with `tune_clause_compact`; each later phase adds its kernel. Common harness — Click CLI with `--kernel <name>`, `--device cuda:0`, `--shape-grid path.json`. Uses `triton.testing.do_bench`. Output is per-arch JSON (informational) + console output formatted to paste into REGISTRY.

2. **`evaluation/retrieval/build_export.py`** cleanup is incremental:
   - Phase 3 retires `AttrIndexWrapper` for `SilverTorch`.
   - Phases 4-6 retire `IndexWrapper` for LiNR variants.
   - `EncoderWrapper` stays (it adapts `predict_last`).
   - The `ATTR_AWARE` and `SUPPORTS_CLAUSES` sets become a per-mode export-entry registry.

3. **[docs/system/](../../docs/system/)** updated per phase: `kernels.md` for kernel-specific changes, `architecture.md` for layer API changes, `filtering.md` for the bloom/clause split, `testing.md` for the per-phase verification commands.

4. **[retrieve/pyproject.toml](../../retrieve/pyproject.toml)** version pins stay (`torch >= 2.4`). The AOTI bump (`torch >= 2.5`) is a separate later effort.

## Files NOT in scope

- `FullScanKNN` ([retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py)), `DotProductScorer` ([retrieve/src/retrieve/layers/utils/scorers.py](../../retrieve/src/retrieve/layers/utils/scorers.py)) — no Triton, no autotune, minimal Optionals. Retrofit in the same PR as their first export consumer if needed.
- `KMeansTorch` ([retrieve/src/retrieve/layers/utils/kmeans.py](../../retrieve/src/retrieve/layers/utils/kmeans.py)), `quantize_int8`, `quantize_oporp_1bit`, `_build_signatures`, `_generate_seeds` — host helpers in `register_index`. Not on the export path.
- [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) — base class signatures may need a `forward_candidates` declaration; treated as a small change inside the relevant phase's PR.
- [retrieve/src/retrieve/kernels/triton/silvertorch/int8_ann_fused.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/int8_ann_fused.py) — parity-only kernel with no production caller; deleting it (or refactoring) is a separate cleanup, not part of this refactor.

## End-state verification (after all 6 phases)

```bash
# 1. Full test suite green:
cd retrieve && uv run pytest tests/ -v
# 2. Eager-mode E2E unchanged for every algo:
cd evaluation && uv run python -m retrieval.benchmark --algo fullscan --k 100
cd evaluation && uv run python -m retrieval.benchmark --algo linr_v1 --k 100
cd evaluation && uv run python -m retrieval.benchmark --algo linr_v2 --k 100
cd evaluation && uv run python -m retrieval.benchmark --algo linr_v3 --k 100
cd evaluation && uv run python -m retrieval.benchmark --algo silvertorch --k 100
# 3. Export round-trips for every (algo, mode):
cd evaluation && uv run python -m retrieval.build_export --checkpoint-dir <path> --index <each>
# 4. Sweeps clean across the codebase:
grep -rn "@triton.autotune" retrieve/src/retrieve/kernels/triton/  # zero hits
grep -rn "\.item()" retrieve/src/retrieve/kernels/triton/ retrieve/src/retrieve/layers/  # only register_index hits, none in forward / launch
grep -rn "Tensor | None\|Optional\[Tensor\]" retrieve/src/retrieve/kernels/triton/  # zero hits
grep -rn "is_cuda" retrieve/src/retrieve/layers/  # zero hits in forward / evaluate paths
```

After all phases, adding `triton_op` + `register_fake` to each kernel is a localized follow-up: each kernel's pure-launch core has the right shape; each layer's `forward` has the right shape; AOTI compile becomes the next, separate piece of work.
