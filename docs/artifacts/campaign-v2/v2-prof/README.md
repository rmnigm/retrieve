# V2 batch-16 floor (profile)

Why LiNR V2 is ≥ 1.5× V1 at bs 16 on goodreads-synth d128 at every pass rate (V-PILOT,
campaign-v2 `408b1188`: V2 clause ~1.41 ms floor, V1 0.94 ms; at bs 1 V2 is faster).
Result row: [validation](../../../validation.md) *V2 batch-16 floor (profile)*, NOT CITABLE.

## Prediction (written before measuring)

N = 797,084, D = 128, C = 7 (synth attrs, one live clause), k = 100, bs 16.

| step | V1 | V2 |
|---|---|---|
| filter | `clause_mask`: attrs 44.6 MB, `[B, N]` bool 12.8 MB | `clause_compact`: attrs 44.6 MB, 24,912 programs ×2 launches, survivors only |
| scoring | cuBLAS mm: 204 MB fp16 read, 51 MB fp32 write | `fused_masked_knn_topk` at **P = N**: 398,544 programs × 8 warps (~461 waves on 108 SMs), 51 MB of `-inf` writes, gathers = passing × 256 B |
| selection | `torch.topk` `[16, N]` fp32 | the same `torch.topk` `[16, N]` fp32 |

V2 moves fewer bytes than V1 (~150 vs ~430 MB), so its floor cannot be bandwidth. It should be the
launch shape of `fused_masked_knn_topk`: the compaction returns full-width `[B, N]` and the public op
does not bucket, so the grid and the score buffer are sized by N, not by the passing count, and
almost every program only stores `-inf`. Prediction: that kernel takes 0.5–0.8 ms at bs 16 at every p
and is the largest V2-only term; the `[16, N]` top-k is shared with V1 and is not the gap. At bs 1
the same grid is 29 waves, which is why V2 still wins there.

LiNR §3.1 ("KNN with Explicit Pre-Filtering") slices the passing items first, so its matmul and its
top-K run over the passing set only; §5.3.2 handles each query of a batch separately. The full-N
width is our choice (a static shape keeps the op graph-capturable without a host sync), not the
paper's design.

## Method

[`profile_v2.py`](profile_v2.py), one process on one A100: V1 (`linr_v1_filter_mask`) and V2
(`linr_v2`), Triton, clause and bloom, bs {1, 16}, p {0.001, 0.01, 1.0}, graph-captured with the
harness's `graph_callable`, timed in 5 interleaved V1/V2 window pairs of 300 calls, `sm_mhz`
sampled after each window; then `torch.profiler` over 20 graph replays (kernel name, launches per
call, µs per call, grid). Component split at bs 16 (clause): each op alone in a CUDA graph,
including the fused op on candidates narrowed to `next_pow2(max count)` columns, and a check that
the narrowed op returns `torch.equal` ids and scores.

## Result

Pending the GPU slot.
