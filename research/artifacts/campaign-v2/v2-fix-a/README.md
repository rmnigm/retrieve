# V2 Fix A: grid-strided `fused_masked_knn_topk`

Result row: [validation](../../../validation.md) *V2 Fix A*, NOT CITABLE. Mechanism and the
profile it fixes: [v2-prof](../v2-prof/README.md).

## Change

`_fused_masked_knn_topk_kernel` launched `(cdiv(P, BLOCK_N), B)` programs; V2's compaction is
full-width, so at P = N = 797,084 that is 398,544 8-warp programs at bs 16, almost all of them
only storing `-inf`. Now the grid is `(G, B)`, `G = min(cdiv(P, BLOCK_N), cdiv(programs, B))`,
`programs` a field of the tile config (default 864, one A100 wave of 8-warp programs; picked by the
sweep below, the prediction was written at 1728).
A program loads its query once and strides over tiles `g, g + G, …`; a tile at or past
`counts[b]` stores `-inf` with no load, a tile below runs the old body (same `BLOCK_N`, same
`tl.sum` over D per lane), so the scores are expected bit-identical. Grid and buffer widths still
depend only on shapes: no host sync, capture unchanged.

## Prediction (written before measuring)

goodreads-synth d128, k 100, graph replay, A100.

| cell | fmkt now (V2-PROF) | fmkt predicted | V2 forward now | V2 predicted |
|---|---|---|---|---|
| clause bs 16, p 0.001 / 0.01 | 913 / 914 µs | 50–100 µs | 1.41 ms | 0.55–0.65 ms (new/old ≈ 0.4–0.45) |
| clause bs 16, p 1 | 2,034 µs | ±5 % (gather-bound) | 2.69 ms | within ±5 % |
| clause bs 1, p 0.001 | 59 µs | 15–30 µs | 0.254 ms | 0.21–0.23 ms |
| clause bs 1, p 1 | 125 µs | +0–10 % (fewer programs in flight) | 0.329 ms | within +5 % |

Bloom gains the same absolute amount (its fmkt is the same 913 µs). The largest risk is p 1: a
fixed G loses the tail-free scheduling of one-tile programs; `programs` is swept at op level
({864, 1728, 3456, 6912, 13824}) before the ABAB run, and the default changes only by
`tune-kernels`' rule (3 % geomean, 5 % worst-regime cap).

V3 stage 1 (`oporp_1bit_match_topk` indirect, P = N bucketed to 1,048,576, BLOCK_N 512): grid
`(2048, B)` = 32,768 4-warp programs at bs 16, 12× fewer than V2's and each doing 512 rows. Prediction:
the kernel is ≤ 100 µs at p ≤ 0.01 and the `[16, 1,048,576]` top-k (≈ 450 µs) dominates stage
1, so a grid fix there gains < 10 % of V3's forward and is **not** applied unless the profile
says otherwise.

## Method

[`fix_a.py`](fix_a.py), one process per phase on one A100 (GPU 0, `flock /scratch/gpu0.lock`). The
`campaign-v2` kernel files are loaded from git under `_old` op and kernel names, next to the new
ones, and a `triton_old` backend namespace routes a module's ops to them, so old and new run in the
same process on the same inputs. Phases: `gate` (op level on the same compaction, plus edge counts
and V2/V3 end to end), `v3prof` (stage 1's op alone), `sweep` (fmkt `programs`), `abab` (V2 and V3
captured by the harness's `graph_callable`, 7 interleaved old/new window pairs of 300 calls,
`sm_mhz` after each window, `unstable` = window spread > 5 %, new/old CI95 over the pair ratios).
Raw: Hub `artifacts/v2-fix-a` ([hub-index](../../hub-index.md)).

## Result

A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, 2026-10-08. `v3prof` and `sweep` ran at
`programs` 1728 for fmkt (the sweep sets it explicitly); `gate` and `abab` at the shipped 864.

**Gates.** `torch.equal` on ids and scores, old vs new, in all 112 comparisons: 96 op cells
(goodreads-synth 7 rates + goodreads `c0_genre`, clause + bloom, bs {1, 16}; fmkt at k {100, 1000},
OPORP indirect at k 5000; 15,360 row evaluations, 1,920 of them at count = N), 16 end-to-end V2/V3 cells
(k {100, 1000}), and 4 edge sets with counts {0, 1, tile ± 1, k − 1, k, 131072, N − 1, N} on the
public op, the bucketed `_impl` and a 3,000-wide `_impl` (OPORP: plus the full scan). Natural data
has no count-0 rows; the edge sets carry them. Every ABAB cell's graph output was also equal, and
every capture passed `graph_callable` (0 cudagraph skips, 1 `cudaGraphLaunch` per call).

**`programs` sweep (fmkt alone, graphed, µs incl. its top-k).**

| bs | p | old | 864 | 1728 | 3456 | 6912 | 13824 |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 204 | 159 | 160 | 162 | 165 | 169 |
| 1 | 0.01 | 205 | 161 | 161 | 161 | 165 | 171 |
| 1 | 1 | 275 | 278 | 279 | 278 | 278 | 275 |
| 16 | 0.001 | 1234 | 514 | 509 | 506 | 506 | 508 |
| 16 | 0.01 | 1237 | 524 | 518 | 514 | 514 | 516 |
| 16 | 1 | 2416 | 1392 | 1610 | 2054 | 2370 | 2452 |

864 is the best or within 1 % of the best in every row. 1728 was an unmeasured first guess, so the
`tune-kernels` keep-the-incumbent rule does not apply; `block_n` / `num_warps` were not re-tuned.

**V3 stage 1 alone (OPORP indirect, k 5000, µs).**

| bs | p | old (P→1,048,576) | new (P = N) | new, narrowed to next_pow2(max count) |
|---|---|---|---|---|
| 1 | 0.001 | 272 | 215 | 136 |
| 1 | 0.01 | 221 | 217 | 161 |
| 1 | 1 | 222 | 218 | – |
| 16 | 0.001 | 729 (kernel 87, top-k 586) | 510 (kernel 32, top-k 418) | 164 |
| 16 | 0.01 | 726 | 511 | 194 |
| 16 | 1 | 803 | 629 | – |

It is the same mechanism as V2's: a full-N grid, plus the public op's bucket, which widened the
buffer and the top-k to the next rung. At arxiv N = 2,988,996 that rung is 16,777,216 (5.6 × N);
at 10M it is 16,777,216 (1.7 × N); at goodreads N = 797,084 it is 1,048,576 (1.3 × N). At 3M and
bs 16 the old op wrote 1.07 GB of fp32 per call and ran top-k over 16.7M columns per row. That is
arithmetic: arxiv-synth is not staged on this pod and was not measured.

**Before/after, forward, graph replay (ms, k 100; kernel = fmkt for V2, OPORP for V3, µs).**

| arm | filter | bs | p | old | new | new/old [CI95] | kernel old → new | min sm_mhz | |
|---|---|---|---|---|---|---|---|---|---|
| V2 | clause | 1 | 0.001 | 0.254 | 0.209 | 0.825 [0.819, 0.830] | 59 → 11 | 1410 |  |
| V2 | clause | 1 | 0.01 | 0.254 | 0.211 | 0.830 [0.826, 0.834] | 59 → 11 | 1410 |  |
| V2 | clause | 1 | 1 | 0.329 | 0.333 | 1.013 [1.010, 1.015] | 124 → 128 | 1410 |  |
| V2 | clause | 16 | 0.001 | 1.409 | 0.688 | 0.483 [0.469, 0.496] | 913 → 136 | 1410 | † |
| V2 | clause | 16 | 0.01 | 1.413 | 0.698 | 0.494 [0.494, 0.495] | 914 → 141 | 1410 |  |
| V2 | clause | 16 | 1 | 2.686 | 1.636 | 0.609 [0.608, 0.609] | 2034 → 984 | 1410 |  |
| V2 | bloom | 1 | 0.001 | 0.287 | 0.243 | 0.825 [0.777, 0.876] | 59 → 11 | 1140 | † |
| V2 | bloom | 1 | 0.01 | 0.290 | 0.245 | 0.846 [0.844, 0.848] | 59 → 11 | 1410 |  |
| V2 | bloom | 1 | 1 | 0.370 | 0.375 | 1.013 [1.010, 1.016] | 129 → 134 | 1410 |  |
| V2 | bloom | 16 | 0.001 | 1.724 | 1.004 | 0.579 [0.570, 0.588] | 913 → 136 | 1410 |  |
| V2 | bloom | 16 | 0.01 | 1.737 | 1.023 | 0.589 [0.588, 0.590] | 914 → 142 | 1410 |  |
| V2 | bloom | 16 | 1 | 2.988 | 1.951 | 0.653 [0.653, 0.653] | 2029 → 994 | 1410 |  |
| V3 | clause | 1 | 0.001 | 0.316 | 0.315 | 0.980 [0.944, 1.017] | 8 → 5 | 1140 | † |
| V3 | clause | 1 | 0.01 | 0.320 | 0.319 | 0.996 [0.995, 0.998] | 8 → 6 | 1410 |  |
| V3 | clause | 1 | 1 | 0.328 | 0.328 | 1.002 [1.000, 1.003] | 12 → 12 | 1410 |  |
| V3 | clause | 16 | 0.001 | 0.962 | 0.738 | 0.767 [0.766, 0.768] | 86 → 32 | 1410 |  |
| V3 | clause | 16 | 0.01 | 0.965 | 0.748 | 0.774 [0.773, 0.776] | 87 → 35 | 1410 |  |
| V3 | clause | 16 | 1 | 1.162 | 0.961 | 0.827 [0.826, 0.828] | 164 → 136 | 1410 |  |
| V3 | bloom | 1 | 0.001 | 0.352 | 0.349 | 0.976 [0.936, 1.017] | 8 → 5 | 1140 | † |
| V3 | bloom | 1 | 0.01 | 0.357 | 0.356 | 0.998 [0.996, 1.001] | 8 → 6 | 1410 |  |
| V3 | bloom | 1 | 1 | 0.364 | 0.365 | 1.004 [1.003, 1.006] | 12 → 12 | 1410 |  |
| V3 | bloom | 16 | 0.001 | 1.289 | 1.067 | 0.829 [0.828, 0.830] | 86 → 31 | 1410 |  |
| V3 | bloom | 16 | 0.01 | 1.301 | 1.084 | 0.833 [0.833, 0.834] | 87 → 35 | 1410 |  |
| V3 | bloom | 16 | 1 | 1.468 | 1.277 | 0.869 [0.868, 0.870] | 154 → 134 | 1410 |  |

† an arm's window spread > 5 % (`unstable`); the three bs 1 p 0.001 ones saw the SM clock dip to
1140 MHz.

**Against the prediction.** V2 bs 16 p ≤ 0.01: predicted 0.55–0.65 ms, measured 0.69–0.70 clause
(fmkt 136–141 µs, not 50–100). V2 bs 16 p 1: predicted ±5 %, measured −39 %; the strided grid also
gathers faster (one wave of programs, query loaded once). bs 1 p ≤ 0.01: −17 %, as predicted. bs 1
p 1: +1.3 % (CI [1.010, 1.016]), inside the 3 % noise band. V3: predicted a < 10 % gain; measured
−17 % to −23 % at bs 16, because the unbucketed top-k shrank as well as the grid; bs 1 unchanged.

**Keep rule (brief).** Gates hold; bs 16, p ≤ 0.01 improves with every CI excluding 1; no
regression past noise at p 1 or bs 1 (worst +1.3 %). Kept.

**What remains.** V2's bs 16 floor is now the `[B, N]` top-k (≈ 355 µs) and the filter pass
(≈ 178 µs), both shared with V1, plus fmkt at ~140 µs (its 51 MB `-inf` store and the loop). V2/V1
at bs 16, p 0.001 is 0.69 / 0.94 ≈ 0.73 against V2-PROF's V1 (V1 was not re-timed here; was 1.50). V3's stage-1 top-k over N (418 µs at bs 16) is
the same floor. Removing it needs a data-dependent width: a user decision, not part of this step.
