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

## Phase 2: fp32 scalar pin, in-kernel −1 tail, boundary checks

Measured before the change, with `torch.profiler` (`prof_probe`): the `torch.full((B, N), -1)` prefill is
**190 µs of the 2055 µs `bloom_compact` op** at B=16, N=3M (384 MB written, 2.0 TB/s). It is 190 µs of
`clause_compact_b16` too, and about 12 µs at B=1.

- **−1 tail in the scatter kernel** (`torch.empty` output; each `(row, tile)` program writes −1
  over its own `BLOCK_N` slice of `[count_b, N)`). The runs cover `[0, count_b)`, so the writes are
  disjoint. Prediction: the op loses one launch and the survivors' second write. The tail itself
  is still written once. Saving ≈ pass_rate × 190 µs + one launch, so **−2 to −5 %** on
  `clause_compact_b16` / `bloom_compact_b16`. At B=1 it is **−1 to −3 %** (≈ 5 µs of 210–285 µs).
  The scatter kernel itself gets slower by the tail bytes it now writes.
- **fp32 pin of `global_scale` / `q_scale`** in the probe kernels: a no-op in eager, where both
  are fp32 already. **Same** timing, same bits.
- **Boundary checks** (non-contiguous item tables rejected, non-power-of-two `tl.arange`
  extents rejected): host-side only, **same** kernel time.

**Outcome** (`phase2.md`, against the Phase 1 commit, interleaved):
- `bloom_compact`: **−4.9 %** at B=16, **−3.0 %** at B=1. Inside the prediction.
- `clause_compact`: +0.7 % at B=16 and +1.5 % at B=1, both inside their noise bands (0.8 % and
  2.8 %). **The predicted saving did not happen.** Kernel split at B=16: fill 191 µs + scatter 80 µs
  before, scatter 274 µs after. With the benchmark's 1.8 % pass rate the tail is 98 % of the
  row, so the tail write costs what the fill did. What is saved is the fill launch plus the survivors'
  second write, and on `clause_compact` that is inside noise. The `bloom_compact` scatter (256
  lanes, 8 warps) went 248 + 190 → 333 µs; the `clause_compact` one (512 lanes, 2 warps) did not
  gain. The retune of `block_n` for the two-phase shape is TF-3, not done here.
- fp32 pin, boundary checks: same within noise, as predicted.

## Phase 3: TF-9 compact CSR probe layout + TF-1 transposed bloom

Measured first (k-means 1024, seed 0, d128): the padded probe width `n_probe · max_cluster` is
611,520 (goodreads, n_probe 24), against a compact width, the sum of the 24 largest clusters, of
**64,757**; each row holds about 18.7k real items. On arXiv it is 171,648 against **128,027**
(about 70k real items per row). A single 25,480-item cluster drives the goodreads padding.

Design: triton and torch register the official backend's CSR (`cluster_offsets`, `sort_perm`,
`inv_perm`, cluster-sorted `item_codes` and attrs). The scorer writes `[B, width]`, with the
compact width above; lane `p` finds its probe by counting row ends and reads the contiguous
sorted row `base[b, j] + p`. Ids are resolved after the top-k (`searchsorted` + `sort_perm`).
Bloom mode reads a transposed index `T [m_bits, ceil(N/64)]` over sorted positions, one bit per item
per set query bit (C·k_hash = 10 bits at the shipped settings, so 10/64 of a word per item),
instead of the 128 B row-wise signature.

Predictions, kernel-only at B=16 (b3's tier A, `torch.profiler` device µs):
- goodreads scorer: none 337 → **~40 µs**, bloom 521 → **~45 µs**, exact 398 → **~60 µs**. The
  time follows the real items (~300k × 128 B, with L2 reuse across the batch), not the padding.
  top-k 335 → **~60 µs** (width 64.8k instead of 611k).
- arXiv scorer: none 130 → **~100 µs**, bloom 229 → **~110 µs**, exact 261 → **~130 µs**.
  Only 1.34× less padding here, but contiguous code rows replace gathered ones.
- The gate, ours (scorer + our mask class) against official (scorer + mask), bloom: goodreads
  about 45 vs 38 µs (**~1.2×**), arXiv about 110 vs 87 µs (**~1.26×**). Both inside the 1.3×
  gate, with little room. If the lane-to-probe search (a loop over n_probe per tile) costs more
  than expected, arXiv misses.
- End to end, the SilverTorch forward gets faster on goodreads by roughly the saved scorer and
  top-k time (~0.5 ms of ~0.75–1.0 ms at B=16), and by less on arXiv.

**Outcome** (`h2h.md`, B=16, scorer µs, padded → compact; official fp16 scorer+mask for the gate):

| | predicted | measured |
|---|---|---|
| goodreads none / bloom / exact | ~40 / ~45 / ~60 | **26 / 34 / 53** |
| arXiv none / bloom / exact | ~100 / ~110 / ~130 | **99 / 105 / 197** (exact missed: the `C·A_max = 20` attrs read per item was not in the estimate) |
| top-k goodreads | ~60 | 92–99 |
| gate bloom goodreads / arXiv | ~1.2× / ~1.26× | **0.90× / 1.24×** |

The first compact kernel did not pass. It used a per-lane probe search, a tile-first grid, and a
host-built layout plus a torch epilogue: arXiv bloom came in at 1.49–1.78× and eager wall *regressed* on arXiv (57
launches against 38, about 230 µs of host overhead). What moved it, each measured on its own:
early-exit tail tiles (goodreads bloom 73 → 29 µs), a batch-first grid for L2 reuse across
the batch (arXiv bloom 126 → 107), cluster-aligned tiles (arXiv bloom 115 → 107 against the
per-lane search), and the probe table built in-kernel plus a one-launch id epilogue (launches
57 → 32–44). Writing ids from the scorer instead of the epilogue cost 13–20 µs on arXiv and was
dropped. The tile config (256×4) was re-swept and stays.

## Phase 4: hardware popcount, LiNR allocator hygiene, quantize OOM

- **Hardware popcount** (`libdevice.popc` on int64, i.e. `__nv_popcll`, two `POPC` in SASS) replaces
  the five-step SWAR in `common.popcount_int64`. The two are bit-identical (checked on random
  words plus 0, −1 and the sign bit). The torch twin stays SWAR, since this torch has no
  `bitwise_count`, and popcount is exact integer math either way. Prediction: the OPORP kernel is
  ALU-bound on SWAR at W=2 (12 int64 ops per word, while the 48 MB table read is ~30 µs). The
  full-scan kernel goes 585 → **~400 µs**, and the op (topk k=5000 dominates) **−8 to −12 %**.
  The indirect op is gather- and top-k-bound: **−2 to −5 %**.
- **quantize_int8_global_codes OOM** (sibling E2 finding, pubmed 10M × 768 fp32 = 28.6 GiB).
  `embs.abs()` and `embs / abs_max * 127` each materialize an N×D fp32 temporary. Fix: `abs_max` as
  `max(amax, -amin)` (two reductions, no temporary, the same value bit for bit), then the codes
  per row chunk into a preallocated int8 output. Prediction: peak transient drops from items +
  2 × items to items + one chunk. Build time changes by under ±10 % (the same bytes are read
  about 3× instead of 2×, but the writes of two temporaries go away).
- **`masked_topk` masking** (audit finding: no prefill analogue exists in the LiNR kernels, whose
  score buffers are `torch.empty` and poison-tested. The masked dense paths spend ~310 µs of a
  1782 µs `PostfilterKNN` forward at B=16, N=3M on `~valid` + a DtoD clone + the fill of
  out-of-place `masked_fill`). `torch.where(valid, scores, -inf)`: the same values, one pass.
  Prediction: **−150 to −180 µs** (~−9 %) on masked `PostfilterKNN` / `PostfilterKNNInt8`
  eager forwards. Inductor fuses the graph mode anyway.

**Outcome** (`phase4.md`, against the Phase 3 commit, interleaved, 3 rounds): `oporp_full_b16` **−2.1 %**
(predicted −8 to −12 %, a miss: the op is top-k(5000)-bound and the kernel's share was smaller than
assumed); `oporp_indirect_b16` **−12.5 %** (predicted −2 to −5 %, a miss the other way: the
bucketed width is 16.7M lanes for P = 3M, and every tail lane ran the SWAR); masked
`PostfilterKNN` **−5.7 %**, `PostfilterKNNInt8` **−2.2 %** (predicted ~−9 %);
`quantize_int8_global` at 3M × 128 **−13.4 %** with the transient bounded to one chunk
(`test_quantize.py`: 672 MiB on a 384 MiB table before, under half the table after); the controls
(`fmkt`, both compactions) within 0.5 %.

## Final: pre-change baseline (`ded8125`) against the branch head, interleaved

`final.md`, 13 cases whose API exists in both trees. The probe scorers' API changed, and their
before/after is `h2h.md`. **No case slower beyond its noise band.**
Faster: `oporp_indirect` −12.3 %, `quantize_int8_global` −13.7 %, masked `PostfilterKNN` −9.4 %,
`bloom_compact` −5.1 % / −3.3 % (B=16 / B=1), `bloom_match` −3.6 %, `PostfilterKNNInt8` −3.3 %,
`oporp_full` −2.4 %. `clause_compact_b1` +1.5 % sits inside an 8 % band and is `unstable` 3/3 on
both sides. The rest are within ±1 %.
