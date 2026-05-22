# Stage 2 — `torch.compile` fixes via `triton_op` / `custom_op`

> See [00-roadmap.md](00-roadmap.md). Second main-thread stage. Stage 1 (autotune separation) has shipped — `@triton.autotune` has been lifted out of every in-scope linr/filter kernel into `<Name>Config + DEFAULT_CONFIG` per [../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation), so the wrappers are already clean for the custom_op decoration step.

> **Scope note (2026-05-19):** silvertorch is out of scope; see [00-roadmap.md](00-roadmap.md). The original migration order had **7 kernels** (1–7); steps for `bloom_match` (originally step 2) and `codesigned_probe_score` (originally step 7) are stubbed out below. Step numbers are preserved so cross-doc references stay stable. Active migration order: 5 kernels.

## Context

Every linr eval-side algo (`linr_v1`, `linr_v2`, `linr_v3`) is an `nn.Module` that calls `self.compile(dynamic=True, mode="reduce-overhead")` in `__init__`. The intent is to capture the whole algo forward (filter + index + cascade) as a single cudagraph_trees graph.

Today this trips a dynamo bug: when the compiled forward reaches a Triton kernel host wrapper that passes `tensor.stride(i)` as a kwarg, dynamo fails inside `_method_size_stride` with `AssertionError: Cannot construct ConstantVariable for value of type torch.SymInt`. The failure precedes any `int(...)` cast at the call site, so caller-side casts don't help.

The current workaround: `@torch._dynamo.disable` on every Triton kernel host wrapper. Functional but causes a graph break per kernel call, fragmenting cudagraph capture. Measured cost on `linr_v1+clause`: **0.314 ms/call vs 0.170 ms/call** when single-graph is preserved — ~85% slower under `@disable`.

This stage replaces every `@torch._dynamo.disable` with a `torch.library.custom_op` registration so dynamo treats the kernel as an opaque traced op (no graph break) while still letting the compiled forward stitch into one cudagraph.

[migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md) covered the "clean" kernels (`clause_mask`, `fused_masked_knn_topk`) and deferred the others because of `Optional[Tensor]` args or data-dependent output shapes. Both blockers go away here:

- **Data-dependent output shape** (`P = max(counts.max().item(), 1)` in `bloom_compact` / `clause_compact`) → return full-width `[B, N]` indices + `counts` tensor. Caller honors `counts` per row. Downstream kernels (`fused_masked_knn_topk`, `oporp_1bit_match_topk`) already iterate over `counts[b]` per row, so they accept wider `positive_indices` without correctness change. Removes the `.item()` host sync — a side-benefit that subsumes [mask-compact-kernel.md](mask-compact-kernel.md) and most of `kernel-optimization-research.md` §2.
- **`Optional[Tensor]` args** (`positive_indices` / `counts` in `oporp_1bit_match_topk`) → pre-allocate dummy `1×1` tensors at the layer, always pass them in, gate the kernel body via a Python `bool` → `tl.constexpr`. The kernel body already does this (`HAS_INDICES` constexpr); only the host wrapper / layer caller need to change.

After this stage, the full algo forward captures as a single cudagraph_trees graph for every linr algo × backend × filter combo.

## Approach

### Decoration choice: `custom_op` over `triton_op` (default)

Reading [`torch/_library/triton.py:200-202`](../../retrieve/.venv/lib/python3.11/site-packages/torch/_library/triton.py): `triton_op` auto-registers the user's `fn` itself as the fake/meta impl. Under `dynamic=True`, the fake runs with `FakeTensor`s carrying SymInt shapes. Most of our kernel wrappers branch on shape ints (`if p == 0`, etc.), which would fail or specialize under SymInt tracing.

`custom_op` treats the body as fully opaque (no fake re-tracing), with an explicit `register_fake` for shape derivation. This matches the "opaque-but-not-graph-breaking" intent and avoids any subtle dynamic-shape interaction in the kernel bodies. Inductor can't inline/fuse the kernel into surrounding ops — for these kernels there's nothing meaningful to inline anyway (the kernel IS the work).

Use `triton_op` only when the body is trivially traceable (a single kernel launch with no shape branches) AND there's a real benefit to letting Inductor see the kernel. Default: `custom_op`.

### Standard recipe

For each kernel:

1. Add `from torch.library import custom_op` import at the top of the kernel file.
2. Replace `@torch._dynamo.disable` on the host wrapper with `@custom_op("retrieve::<name>", mutates_args=())`.
3. Keep the existing body as-is (kernel launch, input validation, `.contiguous()` calls, all `stride(i)` kwargs — all inside the now-opaque op).
4. Define a `register_fake` impl right after the function that returns same-shape, same-dtype, same-device meta tensors derived from input shapes.

Example skeleton (from `clause_mask`):

```python
from torch.library import custom_op

@custom_op("retrieve::clause_mask", mutates_args=())
def clause_mask(
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
) -> Tensor:
    # body unchanged from today
    ...
    return out

@clause_mask.register_fake
def _(item_clause_attrs, clause_is_reverse, query_clause_attrs):
    n = item_clause_attrs.shape[0]
    b = query_clause_attrs.shape[0]
    return torch.empty((b, n), dtype=torch.bool, device=query_clause_attrs.device)
```

### Refactors to unblock the dirty kernels

#### Full-width `(ids, counts)` from compact kernels

`bloom_compact` and `clause_compact` today materialize `[B, N]` indices, slice via `int(counts.max().item())`, return `[B, P]`. After stage 2 they return `[B, N]` directly (no `.item()`; the trailing `-1` sentinel already exists). Downstream callers don't change because they already gate by `counts[b]` per row:

- `fused_masked_knn_topk` kernel body: `count = tl.load(counts_ptr + bid); in_count = n_offsets < count` ([line 65-66](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L65-L66)).
- `oporp_1bit_match_topk` kernel body: same pattern ([line 67-69](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L67-L69)).
- `combine_indices` in [layers/filters/__init__.py:54-59](../../retrieve/src/retrieve/layers/filters/__init__.py#L54-L59): already row-bounded by `counts`. Stays untouched.

The `register_fake` for the compact kernels then returns:

```python
@bloom_compact.register_fake
def _(qb, sigs):
    b = qb.shape[0]
    n = sigs.shape[0]
    return (
        torch.empty((b, n), dtype=torch.long, device=qb.device),
        torch.empty((b,), dtype=torch.long, device=qb.device),
    )
```

#### Optional → dummy-tensor at the layer

The `oporp_1bit_match_topk` host wrapper currently takes `positive_indices: Tensor | None`, `counts: Tensor | None`, binds to a `1×1` dummy when None, and gates via a `HAS_INDICES` kwarg. After stage 2:

- Host wrapper signature drops the `Optional`: every Tensor arg is required.
- Caller (the `OneBitKNN` layer) pre-allocates a `1×1` dummy tensor when the arg is semantically absent, passes a Python `bool` to a constexpr flag.
- The kernel body's existing `HAS_INDICES: tl.constexpr` gate stays.

This is a caller-side refactor only. The kernel itself doesn't change.

#### `if p == 0` removal

Two in-scope kernel host wrappers have `if p == 0: return (...)` early-return paths. The per-row `count == 0` guard inside the kernel already produces the correct sentinel output (`-inf` scores → topk → `where(isfinite, ...)` sentinel). The host short-circuit exists only as a per-row-empty perf optimization and breaks `custom_op`'s opaque-body semantics. Remove.

Files: [fused_masked_knn_topk.py:118-122](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L118-L122), [oporp_1bit_match_topk.py:150-154](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154).

## Per-kernel migration order

Order: existing "clean" kernels first (smallest blast radius, validates the recipe end-to-end including the measurement protocol), then refactor + migrate the "dirty" ones. Original numbering is preserved; step 2 (`bloom_match`) and step 7 (`codesigned_probe_score`) are stubbed out as silvertorch-only.

### 1. `clause_mask`

Already specified in detail in [migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md). Use that plan verbatim. The decoration step is decorator + `register_fake` only — no body changes.

### 2. `bloom_match` — (removed — silvertorch out of scope)

See [../plans-silvertorch-backup/02-triton-op-migration.md](../plans-silvertorch-backup/02-triton-op-migration.md) for the prior write-up.

### 3. `fused_masked_knn_topk`

Already specified in detail in [migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md). Use that plan verbatim. The decoration step is decorator + `register_fake` only — no body changes. Stage 1 already lifted `@triton.autotune` out of `fused_masked_knn_topk` into a `FusedMaskedKnnTopkConfig + DEFAULT_CONFIG + per-bucket bucketed N` pattern, so the wrapper is already clean.

### 4. `bloom_compact` — refactor compact API + migrate

Step A: change return shape from `[B, P]` to `[B, N]` (full-width). Drop `.item()` slice at [bloom_compact.py:132](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L132). The kernel body already writes to full width before the slice — just don't slice.

Step B: verify callers (`BloomFilter.evaluate_indices`, downstream kernels) still work. The downstream kernels already row-bound by `counts`. The torch fallback in `BloomFilter.evaluate_indices` (CPU path) needs updating to compose a full-width result manually (see `clause_compact` example in `torch-export-refactor.md` Phase 1 step 4 for the pattern).

Step C: apply standard `custom_op` recipe.

### 5. `clause_compact` — same as bloom_compact

Same three-step refactor + migration. Files mirror.

### 6. `oporp_1bit_match_topk` — Optional collapse + migrate

Step A: at the caller side in [layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py), pre-allocate dummy tensors for `positive_indices` / `counts` when the layer's mode is `full` (no indices) or `masked` (after stage 2's compact-kernel refactor returns `(ids, counts)`, this mode passes real values).

Step B: change host wrapper signature in [oporp_1bit_match_topk.py:92](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L92) — `positive_indices: Tensor`, `counts: Tensor`, `has_indices: bool`. Drop the `Optional`. The dummy-tensor binding ([lines 166-169](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L166-L169)) moves into the layer.

Step C: remove the `if n_loop == 0` early return ([lines 150-154](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154)).

Step D: apply standard `custom_op` recipe. `register_fake` returns `(ids[B, K], scores[B, K])` with `K = self.k` (Python int at construction time).

### 7. `codesigned_probe_score` — (removed — silvertorch out of scope)

See [../plans-silvertorch-backup/02-triton-op-migration.md](../plans-silvertorch-backup/02-triton-op-migration.md) for the prior write-up (Optional collapse + dummies for `query_bits` / `bloom_sigs`).

## Critical files

| File | Touch |
|---|---|
| [retrieve/src/retrieve/kernels/triton/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) | Decorator + register_fake. Per [migrate-clean-triton-custom-op.md](migrate-clean-triton-custom-op.md). |
| [retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py) | Decorator + register_fake. Remove `if p == 0` early return. |
| [retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py) | Return full-width `(ids[B, N], counts[B])`. Drop `.item()` slice. Decorator + register_fake. |
| [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) | Same. |
| [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py) | Drop `Optional[Tensor]` signature; require dummies. Remove `if n_loop == 0`. Decorator + register_fake. |
| [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | `evaluate_indices` torch-fallback composes full-width. No change to triton path beyond accepting the new return shape. |
| [retrieve/src/retrieve/layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | Same. |
| [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) | Pre-allocate dummy tensors for the no-indices mode; pass `has_indices: bool` to the host wrapper. Drop `int(counts.max().item()) == 0` guard ([line 156](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156)) — kernel handles per-row empty correctly now. |
| [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) | `compact_mask` stays for non-export callers; not on the cudagraph path after stage 2. If any caller is on the compiled forward, route around it via the `(ids, counts)` shape coming from the compact kernels. |
| [retrieve/tests/parity/test_*.py](../../retrieve/tests/parity/) | Update parity tests to assert on full-width `(ids, counts)` (set equality per row, not sequence). Existing tests for `bloom_compact` / `clause_compact` already assert set-equality due to the atomic-add unordered writes — recipe is established. |

## Verification (per kernel and end-of-stage)

Mirrors and extends the protocol in `migrate-clean-triton-custom-op.md`.

### Per-kernel

1. **Correctness**: `cd retrieve && uv run pytest tests/parity/test_<kernel>.py tests/correctness/test_<consumer>.py -v`.
2. **Eager-mode timing regression**: bench the affected algo path before and after (median ms/call, 200 iter, 20 warmup); within ±5% of baseline.
3. **Compiled-mode timing improvement**: same bench but with the algo's `self.compile(...)` engaged. Expect ~40-90% improvement on the call path that previously broke into the kernel.

### End-of-stage

```bash
# 1. Full test suite green.
cd retrieve && uv run pytest tests/ -v

# 2. No @torch._dynamo.disable left on any in-scope kernel host wrapper.
grep -rn "@torch._dynamo.disable" retrieve/src/retrieve/kernels/triton/filters/ retrieve/src/retrieve/kernels/triton/linr/   # zero hits

# 3. Cudagraph warnings absent on every linr algo path.
#    Run each (algo × backend="triton" × filter_kind) combo under verbose logging;
#    grep for "skipping cudagraphs" and "graph break". Zero hits for any migrated kernel.

# 4. End-to-end recall@k unchanged on the eval harness.
cd evaluation && uv run evaluate --config conf/<one yaml per dataset> --backend triton

# 5. Top-K parity against eager reference for LinrV2 + bloom (the most complex
#    code path — fused_masked_knn_topk's tuple return + topk + gather + where + cat
#    tail). Build a compiled algo and parallel uncompiled instance with same
#    weights; assert ids match and scores allclose(rtol=1e-5).
```

A wrong `register_fake` shape will silently shift top-K under compile. Step 5 is the only safety net for that class of bug.

## What this stage explicitly does NOT do

- **No layer-side `mode` flag collapse.** The Optional handling is local: host wrapper signature drops `Optional`, layer caller binds dummies. The layer's public `forward` signature still has Optionals. Mode-flag collapse belongs to [torch-export-refactor.md](torch-export-refactor.md) (deferred).
- **No `_build_query_signatures_eager` / `_project_oporp_1bit_query_eager` swap.** The cudagraph_trees-compiled query helpers stay as-is; they're not on the graph-break-causing path (they're already inside their own compiled region).
- **No `build_export.py`.** Same.
- **No post-launch ops (`topk`, `gather`, `cat`-pad) moved out of the kernel host wrappers.** They stay inside the `custom_op` body. The opaque-body semantics mean Inductor can't fuse them with surrounding ops, but they were already opaque under `@torch._dynamo.disable` so we're not regressing.

## Risks

- **`register_fake` mismatch with real output shape under dynamic shape**. If any fake returns a shape that's a constant where the real impl returns a SymInt-derived shape (or vice versa), torch.compile will produce wrong code. Top-K parity check (verification step 5) catches it; if it fires, the diff is always in the fake impl.
- **Dummy-tensor allocation cost**. Each call to a former-Optional kernel now allocates a `1×1` dummy when the arg is absent. On a hot path this is a tiny but real cost. Mitigate by stashing reusable dummies on the layer (`self._dummy_indices = torch.empty(1, 1, ...)` set in `__init__`).
- **Compact-kernel callers that DO want trimmed-width output**. `combine_indices` ([layers/filters/__init__.py:61](../../retrieve/src/retrieve/layers/filters/__init__.py#L61)) trims via `int(new_counts.max().item())`. It's not on the compiled forward (offline pipeline), so keep `.item()` there. Verify by running `test_combine_filters.py`.
- **`OneBitKNN.forward` `int(counts.max().item()) == 0` early-return**: removing it changes the call shape (always launches the kernel). The kernel's per-row guard handles correctness, but if the all-empty case had a non-trivial perf win from skipping the launch entirely, eager-mode timing on a fully-masked workload may regress. Likely a non-issue (fully-masked is pathological); verify on a synthetic test.
