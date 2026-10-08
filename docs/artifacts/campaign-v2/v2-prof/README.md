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

A100-SXM4-80GB GPU 0, staging `11f45b6` (library unchanged since `408b1188`), torch 2.10.0+cu128,
triton 3.6.0, 2026-10-08, one process, V1/V2 interleaved; raw `profile.json` and `run.log` on the
Hub (`artifacts/v2-prof`, [hub-index](../../hub-index.md)). The SM clock held 1410 MHz in every
bs 16 window; the bs 1 cells at p 0.001 saw 1140–1410 MHz. End-to-end medians equal V-PILOT's
to ±0.02 ms.

### Per-kernel time, graph replay (µs per call; k 100)

"compact/mask" is the filter: `clause_compact` + scan + `compact_scatter` (V2) or `clause_mask`
(V1), and the bloom twins. "mm" is cuBLAS (V1; at bs 1 it is a gemv, 257 µs, counted in
"other"). Launches are per forward, all inside one `cudaGraphLaunch`.

| filter | p | bs | arm | ms | launches | fmkt | topk | compact/mask | mm | other |
|---|---|---|---|---|---|---|---|---|---|---|
| clause | 0.001 | 1 | V1 | 0.449 | 26 | – | 136 | 32 | – | 266 |
| clause | 0.001 | 1 | V2 | 0.254 | 29 | 59 | 135 | 36 | – | 13 |
| clause | 1.0 | 1 | V2 | 0.329 | 29 | 125 | 139 | 37 | – | 13 |
| clause | 0.001 | 16 | V1 | 0.940 | 27 | – | 377 | 173 | 247 | 133 |
| clause | 0.001 | 16 | V2 | 1.408 | 29 | **913** | 354 | 178 | – | 10 |
| clause | 0.01 | 16 | V2 | 1.413 | 29 | **914** | 357 | 179 | – | 11 |
| clause | 1.0 | 16 | V1 | 0.943 | 27 | – | 380 | 174 | 247 | 133 |
| clause | 1.0 | 16 | V2 | 2.686 | 29 | 2034 | 374 | 259 | – | 11 |
| bloom | 0.001 | 16 | V1 | 1.163 | 31 | – | 376 | 383 | 247 | 141 |
| bloom | 0.001 | 16 | V2 | 1.724 | 33 | **913** | 354 | 495 | – | 19 |
| bloom | 1.0 | 16 | V2 | 2.988 | 33 | 2029 | 373 | 561 | – | 19 |

(All 24 cells: `profile.json`.) `fmkt` (`_fused_masked_knn_topk_kernel`) runs on grid
`[24909, B]` in every V2 cell: `cdiv(N, 32)` tiles per row, whatever the pass count.

### Component split, clause, bs 16 (each op alone, CUDA-graphed, µs)

| component | p 0.001 (max count 789) | p 0.01 (7,985) | p 1.0 (797,084) |
|---|---|---|---|
| `clause_mask` `[B, N]` bool | 181 | 181 | 181 |
| `clause_compact` (predicate + scan + scatter) | 185 | 185 | 270 |
| cuBLAS mm fp16 → fp32 `[B, N]` | 250 | 249 | 249 |
| `torch.topk` `[B, N]` fp32 | 379 | 379 | 379 |
| `torch.topk` `[B, next_pow2(max count)]` | 27 (1,024) | 65 (8,192) | 379 (N) |
| fused op as V2 runs it (P = N, incl. its top-k) | **1,235** | **1,239** | 2,425 |
| fused op on candidates narrowed to `next_pow2(max count)` | **44** | **96** | 2,426 |
| V2 forward (eager) | 1,418 | 1,422 | 2,714 |

The narrowed op returned ids and scores `torch.equal` to the P = N op at every p.

### Mechanism

1. **The floor is `fused_masked_knn_topk`'s grid, sized by N, not by the pass count.** The
   compaction returns full-width `[B, N]` (kernels.md § `clause_compact`), and the public op
   runs at `P = positive_indices.shape[1] = N` with no bucketing, so it launches
   `cdiv(N, 32) × B` = 398,544 programs of 8 warps at bs 16. At p 0.001, 789 lanes per row do
   work and the rest store `-inf`. The kernel costs 913 µs at p 0.001 and 914 at p 0.01. That is
   65 % of V2's forward, and it scales with B·N: 59 µs at bs 1, ×15.5 at bs 16. The kernel's real
   traffic at p 0.001 is 51 MB of `-inf` stores and 3 MB of gathers, about 35 µs at HBM
   bandwidth, so the kernel is bound by program count (≈ 460 waves of 8-warp programs), not by
   bytes. The same op on 1,024 columns takes 44 µs including its top-k.
2. **What else scales with N, not with B·passing:** the `torch.topk` over the `[B, N]` score
   buffer (354–374 µs; V1 pays the same 377 µs on its own `[B, N]`), and the compaction's
   predicate pass over the `[N, C, A]` attribute table (178 µs; V1's `clause_mask` costs the same
   173 µs). These are not the V1/V2 gap. Neither is the `[B, T]` scan (11 µs) or the scatter
   (21 µs). There is no per-row launch loop and no host sync: 29 launches per forward, one graph.
3. **V2 − V1 at bs 16, p 0.001 = +468 µs.** V2's fmkt is 913 µs against V1's mm 247 µs (+666 µs).
   V1 pays 133 µs of elementwise kernels (the mask `where`, casts) that V2 does not. The filters
   and the top-k differ by under 25 µs. The sum (≈ +510 µs) and the measured +468 µs differ
   because of kernel overlap and gaps.
4. **At p = 1, V2's fmkt is 2,034 µs**: it gathers all N rows once per query (16 × 204 MB),
   where V1's GEMM reads the table once for the batch. That part is inherent to V2's
   per-row gather and is LiNR's own L-4 (V1 wins at high pass rate). bs 1 has no such reuse to
   lose, which is why V2 beats V1 there at every p.

**Inherent or ours?** The B·N grid and the B·N top-k are ours. LiNR §3.1 slices the passing
items out first, so its matmul *and* its top-K run over the passing set only ("reducing the
computational burden during matrix multiplication and top-K selection"), and §5.3.2 runs each
query of a batch at its own width. We keep a static `[B, N]` width so the forward captures into
one CUDA graph with no host sync (kernels.md § Bucketing: "the compact family returns full-width
`[B, N]`"). The prediction above held in direction. It was low in size: fmkt is 913 µs, not
0.5–0.8 ms.

## Proposal (not applied)

**Fix A: a row-bounded, grid-strided `_fused_masked_knn_topk_kernel`.** This is the smallest
change that keeps one static shape and no sync. The grid becomes `(G, B)` with a fixed
`G ≈ 4 · n_SM / B` (at least 1), and each program loops `for t in range(pid, cdiv(P, BLOCK_N), G)`.
A tile with `t · BLOCK_N >= counts[b]` stores its `-inf` slab with no query, id or embedding load.
A tile below the count runs today's body unchanged. Each lane's dot is still one `tl.sum` over D
inside one program, so the scores are bit-identical. The `[B, P]` buffer is filled the same way,
so `torch.topk` returns the same ids, ties included. One file changes (the kernel plus its
grid in `_fmkt_prep`), and kernels.md § `fused_masked_knn_topk` changes with it.

- **Expected gain (estimate, not measured).** fmkt at p ≤ 0.01, bs 16: 913 → ~40–80 µs (the
  51 MB `-inf` store plus the passing gathers). V2 forward: 1.41 → ~0.55–0.60 ms
  (compaction 178 + top-k 355 + fmkt + 10), so **V2/V1 ≈ 0.6 at bs 16, p ≤ 0.01** against 1.50
  now. bs 1 gains ~50 µs. At p = 1 the gather dominates and nothing changes.
- **What stays:** the `[B, N]` top-k (355 µs) becomes V2's floor at every p, shared with V1.
  Removing it needs a data-dependent width. The narrowed op shows the size: 44 µs and V2 ≈ 0.25 ms
  at bs 16, p 0.001. But it needs `counts.max()` on the host (a sync, so no single-graph forward)
  or a caller-set width cap (exact only while `counts ≤ cap`). That is a design decision for the
  user, not a fix.
- **Gates:** (1) new V2 against current V2 `torch.equal` on ids **and** scores on goodreads-synth
  d128, clause and bloom, all 7 synth rates, bs {1, 16}, k {100, 1000}, plus one real-attrs
  sweep (goodreads `c0_genre`); (2) the existing parity files `test_fused_masked_knn_topk.py`,
  `test_compact_order.py` (consumers ignore the tail past counts), `test_accumulation.py`, and the
  library suite green on a pod GPU; (3) graph capture still `cudagraph_skips == 0` and one
  `cudaGraphLaunch` per call; (4) re-tune `block_n`/`num_warps` for the strided body (`tune-kernels
  fused-masked-knn-topk`) or keep `DEFAULT_CONFIG` if the tuner's best is within noise.
- **Rerun it implies:** a library change moves `code_version`, so every V2 Triton cell is
  re-timed: V-PILOT's goodreads-synth V2 cells (42 records), V-GR-FILTER's V2 cells, and the
  V2 arms of the not-yet-run legs (arxiv-/yfcc10m-synth, filter). Quality is unchanged by
  construction (gate 1), so the quality records stay reusable under the manifest's quality/perf
  split. V3's stage 2 uses the same op at width `candidate_pool`, not N, so it is less exposed.
  `OneBitKNN`'s sparse path (stage 1 on `[B, N]` candidates; `oporp_1bit_match_topk` sizes its grid as `cdiv(n_kernel, block_n)`) may carry the same full-N grid and
  should be profiled before the rerun is scoped. **Not measured here.**
- **Cheaper variant to try first in the same slot:** keep the grid and only add the early branch
  (skip the loads on tiles past the count). It is 3 lines, but it leaves the program count at
  B·N/32, so it probably recovers little. Measure both with a scratch patch before choosing.

Fix A was **not** built or measured here: the estimate is arithmetic from the component split.
A ~15 min GPU slot with a scratch patch in a throwaway worktree settles it.
