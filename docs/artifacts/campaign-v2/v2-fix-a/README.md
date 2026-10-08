# V2 Fix A: grid-strided `fused_masked_knn_topk`

Result row: [validation](../../../validation.md) *V2 Fix A*, NOT CITABLE. Mechanism and the
profile it fixes: [v2-prof](../v2-prof/README.md).

## Change

`_fused_masked_knn_topk_kernel` launched `(cdiv(P, BLOCK_N), B)` programs; V2's compaction is
full-width, so at P = N = 797,084 that is 398,544 8-warp programs at bs 16, almost all of them
only storing `-inf`. Now the grid is `(G, B)`, `G = min(cdiv(P, BLOCK_N), cdiv(programs, B))`,
`programs` a field of the tile config (default 1728 = 108 SMs × 8 resident 8-warp programs × 2).
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
