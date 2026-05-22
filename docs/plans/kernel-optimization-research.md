# Kernel optimization research — proposals across the Triton tree

> **Scope note (2026-05-19):** silvertorch is out of scope; see [00-roadmap.md](00-roadmap.md). Original research surveyed 7 kernels; items below have been trimmed to the 5 in-scope kernels. Silvertorch-only kernels (`bloom_match`, `codesigned_probe_score`) and silvertorch-only proposals (INT8 IVF tensor-core, phase-1 centroid topk) are preserved in [../plans-silvertorch-backup/kernel-optimization-research.md](../plans-silvertorch-backup/kernel-optimization-research.md).

> **Stage 1 status (2026-05-22):** §5 and §6 (autotune-shape work on the compact + mask filter kernels) are **shipped** as part of the autotune-separation effort. Every in-scope kernel now has a `<Name>Config` dataclass + `DEFAULT_CONFIG` tuned offline against real-eval shapes; the filter compact kernels also gained a 3D launch grid so they scale to the 15M-row catalog. See [../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation). The remaining items below stand as written.

## Context

User asked for a research-only pass over every Triton kernel in `retrieve/`,
with a specific question on whether the deferred `mask_compact` plan
([docs/plans/mask-compact-kernel.md](/workspace/retrieve/docs/plans/mask-compact-kernel.md))
should ship now. No benches were run; arxiv was consulted for prior art on
GPU stream compaction, batched top-K, and 1-bit Hamming retrieval.

Goal: a prioritized list of optimizations with cost/benefit notes that the
user can pick from, plus a clear go/no-go on `mask_compact`.

## Kernels reviewed

| Kernel | File | Status |
|---|---|---|
| `fused_masked_knn_topk` | [linr/fused_masked_knn_topk.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py) | well-tuned; minor allocator wins |
| `oporp_1bit_match_topk` | [linr/oporp_1bit_match_topk.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py) | well-tuned; popcount + allocator wins |
| `bloom_compact` | [filters/bloom_compact.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py) | host-sync + scratch-buffer leverage |
| `clause_compact` | [filters/clause_compact.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) | same as bloom_compact |
| `clause_mask` | [filters/clause_mask.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) | autotune candidate (low priority) |

## Headline decision: **defer `mask_compact`**

Neither of the plan's own "When to actually do this" triggers
([mask-compact-kernel.md §When to actually do this](/workspace/retrieve/docs/plans/mask-compact-kernel.md))
has fired:

1. No `OneBitKNNTriton` production / benchmark profile shows `compact_mask`
   ≥ 10% of masked-path wall time. The 30% wins in
   [`_bench_results/optimized.json`](/workspace/retrieve/_bench_results/optimized.json)
   came from the `_bucket_p` autotune-cache stabilization in
   [fused_masked_knn_topk.py:9-19,36](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L9-L36) —
   a `fused_masked_knn_topk` improvement on a path `mask_compact` doesn't touch.
2. No new `FilterModule` subclass without a fused predicate kernel. Both
   shipping filters override `evaluate_indices` on CUDA with their own fused
   compact kernels.

Stale-doc note: [mask-compact-kernel.md](/workspace/retrieve/docs/plans/mask-compact-kernel.md)
references `v3_triton.py` / `LiNR_V3_Triton`. Real path is
[`one_bit_knn_triton.py:52`](/workspace/retrieve/retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py#L52)
and class `OneBitKNNTriton`. Substance unchanged. Fix the doc.

The deeper reason to defer: **the dominant cost on the only hot CUDA caller
is the `counts.max().item()` host sync, not the fp32 argsort.** Killing the
sync (item 1 below) closes most of the gap and benefits every compact-family
kernel — not just `compact_mask`. After that lands, `mask_compact` may be
unnecessary altogether.

## Recommended changes, ranked by leverage

### 1. Drop the empty-batch `.item()` guard in `OneBitKNNTriton.forward` — **TRIVIAL, low risk, ship now**

[`one_bit_knn_triton.py:53`](/workspace/retrieve/retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py#L53):
the `int(counts.max().item()) == 0` guard is redundant. `oporp_1bit_match_topk`
already handles `n_loop == 0`
([line 150-154](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L150-L154))
and the floor `max(.., 1)` in `compact_mask` keeps `P ≥ 1`, so a single-tile
launch with every lane producing `-inf` returns the same `(-1, -inf)` rows
the manual short-circuit does.

Net: one host sync removed per masked forward, no behavioral change. A
companion fix lives in `compact_mask` itself — the early-exit on
[`compact.py:14-20`](/workspace/retrieve/retrieve/src/retrieve/layers/utils/compact.py#L14-L20)
is the same idea inside the helper; the `.item()` there cannot be removed
without item 2.

**Files**:
[layers/linr/one_bit_knn_triton.py](/workspace/retrieve/retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py)

### 2. Investigate and (likely) remove the `counts.max().item()` host sync across the compact family — **MEDIUM effort, MEDIUM risk, biggest real win**

The same `.item()` blocks four call paths:
[`compact.py:14`](/workspace/retrieve/retrieve/src/retrieve/layers/utils/compact.py#L14),
[`bloom_compact.py:132`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L132),
[`clause_compact.py:158`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L158),
and [`combine_indices` at __init__.py:61](/workspace/retrieve/retrieve/src/retrieve/layers/filters/__init__.py#L61).

Critically, downstream scoring kernels **already** tolerate full-width
`positive_indices` because they bound reads by `counts[bid]`:
- [`fused_masked_knn_topk.py:65-72`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L65-L72): `count = tl.load(counts_ptr + bid); in_count = n_offsets < count`.
- [`oporp_1bit_match_topk.py:68-80`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L68-L80): same pattern.

What still needs `P` on the host:
- The `[B, P]` `all_scores` allocation in both scoring wrappers.
- `torch.topk(all_scores, k, dim=1)` width.

Fix path: pass through `counts` (a CUDA tensor) without ever materializing
`P` on the host; pre-size `all_scores` to a bucket via the existing
`_bucket_p` ladder ([fused_masked_knn_topk.py:9-19](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L9-L19))
and pass `P_REAL` to `torch.topk` (which takes a Python int but can use
`min(k, P_BUCKET)` since extra `-inf` lanes only make topk return more
`-inf` rows that the existing `where(isfinite, …)` mask drops).

This subsumes `mask_compact` for the hot path: with the sync gone, the
remaining `mask.float().argsort` on the masked `OneBitKNNTriton` call
reduces to the cost of one `[B, N]` fp32 alloc + one stable argsort. If
that residual cost ever shows up in a profile, *then* ship `mask_compact`.

**Files** (all under `/workspace/retrieve/retrieve/src/retrieve/`):
[layers/utils/compact.py](/workspace/retrieve/retrieve/src/retrieve/layers/utils/compact.py),
[layers/filters/__init__.py](/workspace/retrieve/retrieve/src/retrieve/layers/filters/__init__.py),
[kernels/triton/filters/bloom_compact.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py),
[kernels/triton/filters/clause_compact.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_compact.py),
[kernels/triton/linr/fused_masked_knn_topk.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py),
[kernels/triton/linr/oporp_1bit_match_topk.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py).

### 3. Investigate hardware popcount for `oporp_1bit_match_topk` — **SMALL effort (investigation), real win if applicable**

[`oporp_1bit_match_topk.py:10-19`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L10-L19)
implements a 5-step SWAR popcount. NVIDIA hardware has single-cycle
`popc.b64` (PTX). Steps:

1. Dump compiled PTX for the kernel (`triton.compile(...).asm["ptx"]`).
   If the SWAR is already lowered to `popc.b64` via peephole, **skip** —
   no change needed.
2. If not, probe `tl.extra.libdevice.popcll` (or equivalent in this
   Triton version). If exposed, swap inside `_popcount_int64` behind a
   feature check; keep the SWAR as fallback.

Bit-exactness with the torch reference ([`popcount_int64`](/workspace/retrieve/retrieve/src/retrieve/layers/utils/quantize.py))
is preserved either way — popcount is a deterministic function. The
parity test in [`test_oporp_1bit_match_topk.py`](/workspace/retrieve/retrieve/tests/parity/test_oporp_1bit_match_topk.py)
will continue to pass.

**Files**:
[kernels/triton/linr/oporp_1bit_match_topk.py](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py).

### 4. Allocator / wrapper hygiene across scoring kernels — **TRIVIAL, mechanical**

a. [`oporp_1bit_match_topk.py:155-157`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L155-L157):
   `torch.full(..., -inf)` pre-fill on the HAS_INDICES path is redundant
   — every in-bounds slot is overwritten. Switch to `torch.empty`,
   matching [fused_masked_knn_topk.py:141](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L141).
   The `where(isfinite, ...)` mask at line 200 already covers padding lanes.

b. The `torch.cat`-to-pad tail in
   [oporp_1bit_match_topk.py:204-224](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L204-L224)
   and [fused_masked_knn_topk.py:178-193](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L178-L193)
   does 4 allocations under `P < K`. Apply a 2-allocation pre-allocate-and-slice-assign
   pattern: pre-allocate `[B, K]` outputs, slice-assign the kernel result, leave
   the tail as `-1` / `-inf`. The two host wrappers share the same shape; lift
   the helper rather than duplicating.

c. [`bloom_compact.py:133`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py#L133)
   and [`clause_compact.py:159`](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_compact.py#L159):
   `out_indices[:, :p].contiguous()` triggers a copy of `[B, P]` int64.
   Downstream kernels never write to it; return a `narrow()` view
   (zero-copy) instead. Subsumed by item 2 if/when `out_indices` returns
   at full `N`-width.

d. Stale comment cleanup: [fused_masked_knn_topk.py:170-171](/workspace/retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L170-L171)
   says `clause_compact / bloom_compact allocate via torch.empty`, but
   both now allocate via `torch.full(..., -1)` (bloom_compact.py:110,
   clause_compact.py:133). The defensive `where(isfinite, ...)` is still
   correct (rows with `counts[b] < k` need it) but the rationale is wrong.

**Files**: as inlined above.

### 5. Adopt dynamic `BLOCK_N` in `bloom_compact` / `clause_compact` for small-N callers — **shipped**

Stage 1 (autotune separation) replaced the fixed `BLOCK_N=256` constants
with `BloomCompactConfig` / `ClauseCompactConfig` dataclasses + a
`DEFAULT_CONFIG` tuned offline by `uv run tune-kernels --kernel <name>`
on real-eval shape regimes. Re-tune if a small-N caller becomes hot.

### 6. Autotune `clause_mask` — **shipped**

Same story: `ClauseMaskConfig` + offline-tuned `DEFAULT_CONFIG`. See
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation).

### 7. Doc-only fix — `mask-compact-kernel.md`

Update [docs/plans/mask-compact-kernel.md](/workspace/retrieve/docs/plans/mask-compact-kernel.md):
`v3_triton.py` → `one_bit_knn_triton.py`; `LiNR_V3_Triton` → `OneBitKNNTriton`.
Optionally annotate the file with the validated triggers status (neither
fired) so a future reader doesn't re-evaluate from scratch.

## Explicit non-recommendations

- **Tile-online / in-kernel top-K to eliminate the `[B, n]` HBM round-trip.**
  Considered for all four scoring kernels. Rejected. Triton lacks an
  ergonomic warp-level radix-select primitive
  ([RadiK arxiv:2501.14336](https://arxiv.org/abs/2501.14336),
  [RTop-K ICLR 2025](https://arxiv.org/pdf/2409.00822) — both rely on
  CUDA-level primitives or assume large-K). At our K range (10–200)
  CUB's `torch.topk` runs at near-HBM-bandwidth read; the realistic
  saving is one read of a `[B, n]` fp32 buffer, not the full round-trip.
  The doc at
  [docs/system/kernels.md §"Top-K selection is not in-kernel"](/workspace/retrieve/docs/system/kernels.md)
  already records this trade-off explicitly; the analysis still holds.
- **Silvertorch-only proposals** (phase-1 centroid topk Triton port, INT8 `tl.dot`
  tensor-core for `codesigned_probe_score`). Silvertorch is out of scope; see
  [../plans-silvertorch-backup/kernel-optimization-research.md](../plans-silvertorch-backup/kernel-optimization-research.md)
  for the prior write-up.
- **CPU `mask_compact` primitive.** Same rationale as the deferred plan.
- **Subsume `clause_compact` / `bloom_compact` via `evaluate_mask` +
  `mask_compact`.** Re-introduces the `[B, N]` bool intermediate the
  fused kernels exist to avoid.
- **Ship `mask_compact` speculatively.** Plan's own discipline; both
  triggers unmet; item 2 is the larger lever and would render
  `mask_compact` mostly redundant on the only hot path.

## Verification

For each change actually picked up:

- **Item 1**: existing `test_one_bit_knn_triton` and `test_linr.py` correctness
  tests cover the empty-mask case. Add a CUDA-only assertion that
  `OneBitKNNTriton.forward(query, mask=all_false_mask)` issues zero
  `cudaStreamSynchronize` events (e.g., wrap in `torch.cuda.set_sync_debug_mode("warn")`).
- **Item 2**: keep all parity tests in [tests/parity/](/workspace/retrieve/retrieve/tests/parity/)
  passing; add a single regression test using
  `torch.cuda.set_sync_debug_mode("error")` around a `BloomFilter →
  PrefilterKNN` call to fail on any new host sync. Bench cell to confirm
  end-to-end masked-path latency drop on the same shapes captured in
  [_bench_results/optimized.json](/workspace/retrieve/_bench_results/optimized.json).
- **Item 3**: re-run [test_oporp_1bit_match_topk.py](/workspace/retrieve/retrieve/tests/parity/test_oporp_1bit_match_topk.py)
  bit-exact assertion. If swapping in libdevice popcll, also dump PTX
  before and after to confirm the swap actually emits `popc.b64`.
- **Items 4 & 5**: existing parity tests cover correctness; no new
  assertions needed.

## Suggested sequencing

1. Item 7 (doc fix to `mask-compact-kernel.md`) — 5 minutes.
2. Item 1 (drop `.item()` guard in `OneBitKNNTriton`) — small diff, low
   risk, immediate sync removal.
3. Item 4a + 4b + 4d (allocator hygiene) — mechanical pass across two
   files; low risk.
4. Item 5 (dynamic `BLOCK_N` in compacts) — `block_n = 128 if n >= 128 else next_power_of_2(n)` host-side ladder.
5. Item 3 (PTX dump for popcount; swap if libdevice exposes `popcll`).
6. Item 2 (kill the host sync end-to-end). Largest design change; do
   last so it can build on the cleaned-up wrappers from steps 2–4.
7. Item 4c (replace `.contiguous()` slice with `narrow()`) — only if
   item 2 isn't done; otherwise subsumed.
8. Revisit `mask_compact` only if (a) item 2 lands and the residual
   `mask.float().argsort` still shows ≥10% in a profile, or (b) a new
   `FilterModule` subclass ships without a fused predicate kernel.

## Sources consulted

- [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](https://arxiv.org/abs/2501.14336)
  — informed the rejection of in-kernel top-K at our K range.
- [RTop-K (ICLR 2025)](https://arxiv.org/pdf/2409.00822) — same.
- [NVIDIA: Optimized Filtering with Warp-Aggregated Atomics](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)
  — confirmed `tl.cumsum + tl.atomic_add` per tile is already the
  optimal stream-compaction shape (no win from a different primitive).
- [Faster Population Counts Using AVX2 Instructions, arxiv:1611.07612](https://arxiv.org/pdf/1611.07612)
  — context for the SWAR vs hardware popcount investigation.
