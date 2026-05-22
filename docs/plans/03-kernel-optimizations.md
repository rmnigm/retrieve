# Stage 3 — Kernel optimizations on the clean base

> See [00-roadmap.md](00-roadmap.md). Third main-thread stage. Stage 1 (autotune separation) has shipped — see [../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation). Depends on [02-triton-op-migration.md](02-triton-op-migration.md).

## Context

Stages 1 and 2 subsumed all of the big-lever kernel-optimization items that were in flight before they shipped: the `counts.max().item()` host sync across the compact family is gone (compact kernels now return full-width `(ids, counts)`), the `int(counts.max().item()) == 0` short-circuit in `OneBitKNNTriton.forward` will be retired with the oporp `@custom_op` migration, the `out_indices[:, :p].contiguous()` slice no longer exists, and offline `DEFAULT_CONFIG` selection replaces the autotune-`BLOCK_N` and `clause_mask`-autotune items (re-tune via `uv run tune-kernels --kernel <name>` on a new arch).

What remains is small and focused: hardware popcount investigation + allocator hygiene in two host wrappers.

## In scope — two items, each independent

### 1. Hardware popcount in `oporp_1bit_match_topk` — investigation, then maybe swap

[oporp_1bit_match_topk.py:10-19](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L10-L19) implements a 5-step SWAR popcount inside the `@triton.jit` body. NVIDIA hardware has single-cycle `popc.b64` (PTX); if the SWAR doesn't already lower to it, swapping in `tl.extra.libdevice.popcll` (or whatever the current Triton API surfaces) is a real win on a kernel that dominates the V3 hot path.

**Steps**:

1. **Dump compiled PTX**: `triton.compile(_oporp_1bit_match_topk_kernel, ...).asm["ptx"]`. Grep for `popc.b64`. If the SWAR is already lowered (Triton peephole or LLVM backend), STOP — no change needed.

2. If not, probe `tl.extra.libdevice.popcll` in the current Triton version. If exposed, swap inside `_popcount_int64` behind a feature check; keep the SWAR as a fallback.

3. Bit-exactness with `popcount_int64` in [layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py) is preserved either way — popcount is deterministic. [test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) covers parity.

4. If the swap happens, re-dump PTX to confirm `popc.b64` is actually emitted. Bench the V3 hot path before / after on the local arch.

**Files**: [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py).

### 2. Allocator hygiene in `oporp_1bit_match_topk` and `fused_masked_knn_topk`

Three sub-items, all mechanical.

#### 2a. Drop the `torch.full(-inf)` pre-fill on the HAS_INDICES path

[oporp_1bit_match_topk.py:155-157](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L155-L157): pre-fills with `-inf` even though every in-bounds slot is overwritten by the kernel. Switch to `torch.empty`, matching the analogous allocation in [fused_masked_knn_topk.py:141](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L141). The `where(isfinite, ...)` mask at [line 200](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L200) already handles padding lanes.

#### 2b. Replace `cat`-to-pad with pre-allocate-and-slice

[oporp_1bit_match_topk.py:204-224](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L204-L224) and [fused_masked_knn_topk.py:178-193](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L178-L193) do 4 allocations under `P < K`. Apply a 2-allocation pattern: pre-allocate `[B, K]` outputs, slice-assign the kernel result, leave the tail as `-1` / `-inf`. The two host wrappers share the same shape; lift the helper rather than duplicating.

#### 2c. Stale-comment cleanup

[fused_masked_knn_topk.py:170-171](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L170-L171) says `clause_compact / bloom_compact allocate via torch.empty`, but both now allocate via `torch.full(..., -1)`. The defensive `where(isfinite, ...)` is still correct (rows with `counts[b] < k` need it) but the rationale is wrong. Fix the comment.

**Files**: as inlined.

## Out of scope (don't bundle here)

- **In-kernel top-K to eliminate the `[B, n]` HBM round-trip.** At our K range (10-200) CUB's `torch.topk` is near-HBM-bandwidth; the realistic saving is one read of a `[B, n]` fp32 buffer, not the full round-trip. The trade-off doc in [docs/system/kernels.md](../system/kernels.md) (top-K-not-in-kernel convention) still holds.
- **CPU `mask_compact` primitive.** Not the target audience.
- **Subsuming `clause_compact` / `bloom_compact` via `evaluate_mask` + `mask_compact`.** Re-introduces the `[B, N]` bool intermediate that the fused kernels exist to avoid.
- **Anything in [live-update-api.md](live-update-api.md), [torch-export-refactor.md](torch-export-refactor.md).** Separate concerns.
- **Silvertorch kernel optimizations (`codesigned_probe_score` tensor-core, phase-1 centroid topk).** Silvertorch is out of scope; see [../plans-silvertorch-backup/03-kernel-optimizations.md](../plans-silvertorch-backup/03-kernel-optimizations.md) for the prior write-up.

## Verification

For each item independently (they're orthogonal):

- **Item 1 (popcount)**: parity test [test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) stays bit-exact. PTX dump before/after to confirm the swap actually emits `popc.b64`. V3 hot-path bench (median ms/call, 200 iter) to confirm the swap is net positive.
- **Item 2 (allocator hygiene)**: existing parity tests stay green. Bench cell from [evaluation/retrieval/bench_tools.py](../../evaluation/retrieval/bench_tools.py) on V3 (oporp) and V2 (fused_masked_knn_topk) hot paths; expect a small but measurable reduction in allocator overhead (manifests as lower median latency at low B / K).

End-of-stage: full pytest green, no recall@k regression on the eval harness for any algo.

## Risks

- **Popcount swap regresses on some arches**. `popc.b64` is single-cycle on sm_80+, but earlier arches or specific Triton lowerings may not benefit. Mitigation: feature-check + SWAR fallback; bench per-arch via `tune-kernels` (already added in stage 1).
- **Allocator hygiene changes mask a real correctness bug elsewhere**. The pre-fill `-inf` was defensive; removing it relies on the `where(isfinite, ...)` mask being correct. Parity tests cover this — they were already exercising the same path.

## What this stage explicitly does NOT do

- Per-kernel autotune (stage 1's REGISTRY is the autotune story; widening rows is allowed if `tune-kernels` shows benefit, but that's REGISTRY maintenance, not a structural change).
- Layer-API changes (mode flags, Optional collapse).
- Kernel-body restructuring beyond the popcount swap.
- New kernels.
