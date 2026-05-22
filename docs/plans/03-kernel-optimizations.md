# Stage 3 — Kernel optimizations on the clean base

> See [00-roadmap.md](00-roadmap.md). Third main-thread stage. Stage 1 (autotune separation) has shipped — see [../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation). Depends on [02-triton-op-migration.md](02-triton-op-migration.md).

## Context

[kernel-optimization-research.md](kernel-optimization-research.md) enumerates a leverage-ranked list of per-kernel optimizations. Most of that list assumes the kernel base before stages 1+2: autotune at call sites, host-side `.item()` syncs, `[B, P]` trimmed-width outputs, etc. After stages 1+2, several items shrink or vanish entirely:

| Research item | Status after stages 1+2 |
|---|---|
| §1 Drop empty-batch `.item()` guard in `OneBitKNNTriton.forward` | **Done in stage 2** (the `int(counts.max().item()) == 0` short-circuit at [one_bit_knn.py:156](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L156) is removed when the compact kernels return full-width `(ids, counts)`). |
| §2 Kill `counts.max().item()` across compact family — *biggest single lever* | **Done in stage 2** (full-width `(ids, counts)` API from `bloom_compact` / `clause_compact`). |
| §3 Hardware popcount in `oporp_1bit_match_topk` | **Survives.** Independent of stages 1+2. |
| §4a Drop redundant `torch.full(-inf)` pre-fill in `oporp_1bit_match_topk` | **Survives.** Local to the host wrapper body. |
| §4b Pre-allocate-and-slice the pad-to-K tail in `oporp_1bit_match_topk` / `fused_masked_knn_topk` | **Survives.** Local to the host wrapper bodies. |
| §4c Replace `out_indices[:, :p].contiguous()` with `narrow()` view | **Eliminated by stage 2** (no `.contiguous()` slice in the new full-width API). |
| §4d Stale comment cleanup in `fused_masked_knn_topk` | **Survives** (mechanical). |
| §5 Dynamic `BLOCK_N` in `bloom_compact` / `clause_compact` for small-N callers | **Subsumed by stage 1** (kernel-file `DEFAULT_CONFIG` now picked offline against real-eval shapes; re-tune via `uv run tune-kernels --kernel bloom_compact` / `clause_compact` if shapes shift). |
| §6 Autotune `clause_mask` | **Subsumed by stage 1** (kernel-file `DEFAULT_CONFIG` shipped, re-tune the same way). |
| §7 Doc fix in `mask-compact-kernel.md` | **Survives** (annotation note; or just archive the doc — see below). |

What remains as actual stage 3 work is small and focused: hardware popcount investigation + allocator hygiene in two host wrappers + a doc fix.

## In scope — three items, each independent

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

### 3. Archive `mask-compact-kernel.md`

After stage 2, the only hot caller of `compact_mask` (`OneBitKNNTriton` masked path) is rewired through the full-width `(ids, counts)` API and doesn't go through `compact_mask` at all. The "When to actually do this" triggers in [mask-compact-kernel.md](mask-compact-kernel.md) should both be unmet. Re-check on the current tree:

- Trigger 1: profile shows `compact_mask` ≥ 10% of V3 masked-path wall time. Unlikely after stage 2; `compact_mask` isn't reached.
- Trigger 2: a new `FilterModule` subclass without a fused compact kernel. None today; would be a future event.

If both unmet, annotate the doc as "subsumed by 02-triton-op-migration.md" with a one-line pointer and stop.

## Out of scope (don't bundle here)

- **In-kernel top-K to eliminate the `[B, n]` HBM round-trip.** `kernel-optimization-research.md` "Explicit non-recommendations" §1 covers the rationale: at our K range (10-200) CUB's `torch.topk` is near-HBM-bandwidth; the realistic saving is one read of a `[B, n]` fp32 buffer, not the full round-trip. The trade-off doc in [docs/system/kernels.md §"Top-K selection is not in-kernel"](../system/kernels.md) still holds.
- **CPU `mask_compact` primitive.** Not the target audience.
- **Subsuming `clause_compact` / `bloom_compact` via `evaluate_mask` + `mask_compact`.** Re-introduces the `[B, N]` bool intermediate that the fused kernels exist to avoid.
- **Anything in [live-update-api.md](live-update-api.md), [torch-export-refactor.md](torch-export-refactor.md).** Separate concerns.
- **Silvertorch kernel optimizations (`codesigned_probe_score` tensor-core, phase-1 centroid topk).** Silvertorch is out of scope; see [../plans-silvertorch-backup/03-kernel-optimizations.md](../plans-silvertorch-backup/03-kernel-optimizations.md) for the prior write-up.

## Verification

For each item independently (they're orthogonal):

- **Item 1 (popcount)**: parity test [test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) stays bit-exact. PTX dump before/after to confirm the swap actually emits `popc.b64`. V3 hot-path bench (median ms/call, 200 iter) to confirm the swap is net positive.
- **Item 2 (allocator hygiene)**: existing parity tests stay green. Bench cell from [evaluation/retrieval/bench_tools.py](../../evaluation/retrieval/bench_tools.py) on V3 (oporp) and V2 (fused_masked_knn_topk) hot paths; expect a small but measurable reduction in allocator overhead (manifests as lower median latency at low B / K).
- **Item 3 (doc fix)**: no code changes; just `git log` showing the annotation.

End-of-stage: full pytest green, no recall@k regression on the eval harness for any algo.

## Risks

- **Popcount swap regresses on some arches**. `popc.b64` is single-cycle on sm_80+, but earlier arches or specific Triton lowerings may not benefit. Mitigation: feature-check + SWAR fallback; bench per-arch via `tune-kernels` (already added in stage 1).
- **Allocator hygiene changes mask a real correctness bug elsewhere**. The pre-fill `-inf` was defensive; removing it relies on the `where(isfinite, ...)` mask being correct. Parity tests cover this — they were already exercising the same path.

## What this stage explicitly does NOT do

- Per-kernel autotune (stage 1's REGISTRY is the autotune story; widening rows is allowed if `tune-kernels` shows benefit, but that's REGISTRY maintenance, not a structural change).
- Layer-API changes (mode flags, Optional collapse).
- Kernel-body restructuring beyond the popcount swap.
- New kernels.
