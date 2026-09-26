# Predictions, written before each change was made

Rule: agent-orchestration §5, a performance change states its expected effect before it is
measured. Measured outcomes sit next to each prediction once taken, and a miss stays recorded
as a miss.

Baseline: `phase0_aa.md`, an A/A interleaved run (baseline snapshot of `ded8125` against the
unchanged working tree, 3 rounds each, alternating processes). Every A/A ratio lies in
0.994–1.010. The round-to-round range per case is 0.1–10 % (bloom_match_b16 has one outlier round).
Decision rule used below: **a regression is new/base > 1.03 and outside the case's A/A noise
band**. That means stop and report.

## Phase 1: int64 row bases

Change: every scalar row base (`bid * stride_b`, `tile_row0 * stride_n`) becomes int64
(`.to(tl.int64)` on the scalar before the multiply). Lane offsets (`tl.arange`) stay int32
relative to that base. `bloom_match` also moves to the 3-D `grid_batch_tiles` grid with
runtime `N` (the Phase 5 batch-1 item). Its tile axis leaves grid_y, which caps at 65,535.

Prediction: **same within noise (|Δ| < 1 %) for every case.** The extra work is one 64-bit
scalar multiply per program. The per-lane address arithmetic is unchanged, because pointers are
already 64-bit and an int32 lane offset is sign-extended either way. `bloom_match`'s grid change
keeps batch on grid_x and the same BLOCK_N=128, so its program order is the same. Runtime `N`
instead of constexpr removes one constant from the mask compare. Expected: same.
Risk: if Triton turns an int64 scalar base into 64-bit per-lane arithmetic, register pressure
rises on `bloom_compact` (W=16, the kernel with the spill cliff). That would show as > 3 %
on bloom_compact_b16.

**Outcome: the prediction missed twice before it held.**

1. `phase1_first_attempt.md`, always-int64 row bases with relative lanes: `bloom_compact_b16`
   +11.8 %, `fmkt_b16` +7.1 %, `bloom_match_b16` +6.3 %, all outside noise. That is a stop.
   Diagnosis by `torch.profiler` and PTX: (a) `fmkt` has 1.5M 32-lane programs, and a
   zero-extended program id times a sign-extended stride compiles to `mul.lo.s64`+`shl`+`add`
   instead of one `mad.wide.s32`, which costs +6.5 % on the kernel by itself. (b) `bloom_compact` goes
   from 40 to 46 registers (6 → 5 blocks/SM at 8 warps). (c) `bloom_match` is runtime `N`, not
   the int64 at all: constexpr `N` measures 1269 µs against 1418 µs runtime.
2. `phase1_second_attempt.md`, a `row_base` helper with `tl.assume(row, stride >= 0)` (one
   `mul.wide.u32`), widened only where the extent can reach 2³¹, `bloom_match` back to
   constexpr `N`. Everything within noise except `clause_compact` at B=1, +3.4 % on the kernel,
   separated across runs. Bisect: that is the *formulation*, a pointer shifted by
   `row0·stride` plus relative lanes, which costs the same even in int32.
3. `phase1.md`, final. Every kernel takes `WIDE: tl.constexpr`, set on the host when a
   tensor it addresses holds ≥ 2³¹ elements. Narrow launches compile to the baseline
   addressing, wide ones use int64 row bases. **All 13 cases within noise; `bloom_match_b16` is 3.5 % faster**
   (0.965, band 0.7 %; not investigated further). The large-size tests pin the wide path, and
   a mutation of `row_base`'s int64 cast turns all five red.
