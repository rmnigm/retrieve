# Probe-scorer tile skipping (#16)

Plan item #16 of campaign-v2: in the fused probe scorers
(`codesigned_probe_score` bloom variant, `codesigned_probe_score_exact`), test the
tile's filter first and skip the int8 code load and the `tl.dot` when no lane of the
tile passes. Result row: [validation](../../../validation.md) *Probe-scorer tile
skipping*; mechanism: [kernels](../../../system/kernels.md#silvertorch-kernels).

## Prediction (written before measuring)

What the kernel already does: the code load is masked by `keep`, so a failing item's
128-byte int8 row is never fetched; HBM traffic for failing items is already zero. What
a skip can still save on an all-fail tile: the predicated load instructions, the
`tl.dot` (an IMMA with M = 1 padded to the MMA tile), the dequant epilogue, the query
load. It cannot save the filter test itself (the bloom loop over `C·k_hash` transposed
words, or the `C·A_max` attribute loads), and the tile still stores `-inf`.

Tile occupancy is what decides it. With `BLOCK_P = 256` (D = 128) and independent
per-item pass rate `p`, a tile is all-fail with probability `(1 - p)^256` (plus bloom
false positives, which only lower it):

| p | P(tile skipped), exact | expected effect at bs 16 |
|---|---|---|
| 0.001 | 0.77 | scorer 10-30 % faster (the dot and epilogue gone on 3/4 of the tiles) |
| 0.01 | 0.08 | within noise (≤ 3 %): 92 % of tiles still run the dot |
| 0.1 | ~0 | within noise |
| 1.0 | 0 | ≤ 2 % slower at worst (one `tl.max` reduction and a branch per tile) |

Bloom FPR at `m_bits = 1024, k_hash = 5`, one clause, is small, so bloom tracks exact.
So I expect the keep rule (p = 1.0 upper bound ≤ 1.03 **and** p ≤ 0.01 upper bound
< 1.0 on both shapes at bs 16) to **fail at p = 0.01**, and the change to be reverted,
with a real gain only at p = 0.001. Meta's bloom scorer being faster than its own
unfiltered scorer would then come from its finer skip granularity (a warp of 32 docs
per mask word), not from tile skipping as such.

## Method

`tile_skip.py` (one process per shape): one `SilverTorch` IVF index per shape (synthetic
fp32 Gaussian items, `n_iter = 5`), the bloom / exact buffers rebuilt per pass rate on
that same layout (one clause, `A_max = 1`; item value `1` with probability `p`, else `0`;
every query asks `1`). Arms, all on the same inputs:

- `before`: the scorer kernel as of `b96e1f2`, loaded from `git show` under a private op
  namespace;
- `after`: the working tree's kernel;
- `official`: Meta's `fused_kmean_ann` via `ops.official`, reference only (its op syncs the
  host, so it is timed by `torch.profiler` kernel time, not by graph replay).

`before` / `after` time the scorer kernel alone (no top-k): each arm is a CUDA graph of 64
launches over a pool of 8 query batches, replayed between CUDA events; windows run ABAB,
12 pairs per cell, the SM clock sampled after each window. Ratio = after / before per
pair; the 95 % CI is the t-interval of the mean log ratio over the pairs. A window is
`unstable` when the sampled clock moves by more than 50 MHz within the cell.

Shapes: arXiv-like (N 3.0 M, D 128, n_lists 1664, n_probe 24) and goodreads-like
(N 0.8 M, D 128, n_lists 1024, n_probe 24); bs ∈ {1, 16}; p ∈ {0.001, 0.01, 0.1, 1.0}.

```bash
cd evaluation   # for the official adapter's deps; RETRIEVE tree = this worktree
CUDA_VISIBLE_DEVICES=0 uv run python ../docs/artifacts/campaign-v2/tile-skip/tile_skip.py arxiv /scratch/cv2-lib/arxiv.json
CUDA_VISIBLE_DEVICES=0 uv run python ../docs/artifacts/campaign-v2/tile-skip/tile_skip.py goodreads /scratch/cv2-lib/goodreads.json
```

## Result: reverted

A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, 2026-10-08, `before` = `b96e1f2`,
`after` = that tree plus [`tile_skip_kernel.diff`](tile_skip_kernel.diff) (the kernel
change, kept here because it was reverted). Synthetic layouts: goodreads-like width
21,511, arXiv-like width 47,735. The scores of `before` and `after` were `torch.equal`
on every cell (the full `[B, width]` buffer), and the parity files
(`test_codesigned_probe_score*.py`, `test_official.py`, 116 tests) passed on the changed
kernel. The SM clock held at 1140 or 1410 MHz within almost every cell; the two
`unstable` cells are bs 1 cells.

| shape | mode | bs | p | before µs | after µs | after / before [95 % CI] | official scorer µs | official mask µs | sm_mhz | unstable |
|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | bloom | 1 | 0.001 | 5.9 | 6.2 | 1.064 [1.017, 1.112] | 17.5 | 10.5 | 1410–1410 | False |
| goodreads | bloom | 16 | 0.001 | 23.2 | 16.5 | 0.710 [0.704, 0.717] | 24.2 | 12.4 | 1140–1140 | False |
| goodreads | exact | 1 | 0.001 | 6.1 | 6.0 | 1.006 [0.948, 1.068] | 17.8 | 5.6 | 1140–1140 | False |
| goodreads | exact | 16 | 0.001 | 20.9 | 14.4 | 0.692 [0.684, 0.701] | 25.8 | 33.8 | 1140–1140 | False |
| goodreads | bloom | 1 | 0.01 | 6.4 | 6.6 | 1.007 [0.976, 1.038] | 14.7 | 8.5 | 1410–1410 | False |
| goodreads | bloom | 16 | 0.01 | 19.1 | 18.3 | 0.958 [0.943, 0.973] | 20.4 | 9.7 | 1410–1410 | False |
| goodreads | exact | 1 | 0.01 | 6.1 | 6.1 | 1.010 [0.955, 1.069] | 18.8 | 5.8 | 1140–1140 | False |
| goodreads | exact | 16 | 0.01 | 21.1 | 20.0 | 0.951 [0.937, 0.966] | 27.5 | 33.8 | 1140–1140 | False |
| goodreads | bloom | 1 | 0.1 | 6.6 | 6.9 | 1.038 [1.001, 1.076] | 15.3 | 8.7 | 1140–1380 | True |
| goodreads | bloom | 16 | 0.1 | 20.5 | 20.8 | 1.020 [1.010, 1.031] | 25.1 | 11.4 | 1380–1380 | False |
| goodreads | exact | 1 | 0.1 | 6.1 | 6.2 | 1.017 [0.990, 1.045] | 19.5 | 5.7 | 1140–1140 | False |
| goodreads | exact | 16 | 0.1 | 21.9 | 22.6 | 1.022 [0.997, 1.048] | 31.6 | 33.8 | 1140–1140 | False |
| goodreads | bloom | 1 | 1.0 | 6.7 | 7.1 | 1.042 [1.009, 1.076] | 17.4 | 8.6 | 1140–1410 | True |
| goodreads | bloom | 16 | 1.0 | 28.4 | 28.9 | 1.020 [1.006, 1.035] | 51.2 | 13.0 | 1410–1410 | False |
| goodreads | exact | 1 | 1.0 | 7.1 | 6.9 | 0.991 [0.954, 1.029] | 22.6 | 6.3 | 1140–1140 | False |
| goodreads | exact | 16 | 1.0 | 28.5 | 29.6 | 1.042 [1.028, 1.056] | 55.3 | 33.9 | 1140–1140 | False |
| arxiv | bloom | 1 | 0.001 | 6.2 | 6.4 | 1.027 [1.009, 1.046] | 14.8 | 9.1 | 1410–1410 | False |
| arxiv | bloom | 16 | 0.001 | 31.0 | 21.4 | 0.688 [0.682, 0.694] | 25.6 | 14.2 | 1410–1410 | False |
| arxiv | exact | 1 | 0.001 | 6.9 | 6.6 | 0.940 [0.889, 0.994] | 18.8 | 19.6 | 1140–1140 | False |
| arxiv | exact | 16 | 0.001 | 35.3 | 24.3 | 0.689 [0.682, 0.696] | 27.0 | 116.6 | 1140–1140 | False |
| arxiv | bloom | 1 | 0.01 | 6.7 | 7.0 | 1.040 [1.006, 1.074] | 14.9 | 9.0 | 1410–1410 | False |
| arxiv | bloom | 16 | 0.01 | 32.7 | 31.9 | 0.975 [0.967, 0.983] | 23.4 | 11.8 | 1410–1410 | False |
| arxiv | exact | 1 | 0.01 | 6.8 | 6.9 | 1.013 [0.979, 1.049] | 19.7 | 19.5 | 1140–1140 | False |
| arxiv | exact | 16 | 0.01 | 35.6 | 35.6 | 1.000 [0.992, 1.008] | 29.9 | 116.6 | 1140–1140 | False |
| arxiv | bloom | 1 | 0.1 | 6.9 | 7.1 | 1.039 [1.002, 1.078] | 15.3 | 9.0 | 1410–1410 | False |
| arxiv | bloom | 16 | 0.1 | 36.5 | 36.8 | 1.009 [1.004, 1.014] | 29.7 | 12.0 | 1410–1410 | False |
| arxiv | exact | 1 | 0.1 | 6.7 | 6.9 | 1.028 [1.004, 1.054] | 20.3 | 19.7 | 1140–1140 | False |
| arxiv | exact | 16 | 0.1 | 36.9 | 39.0 | 1.058 [1.052, 1.064] | 36.3 | 116.6 | 1140–1140 | False |
| arxiv | bloom | 1 | 1.0 | 9.6 | 9.5 | 1.015 [0.972, 1.061] | 19.5 | 10.5 | 1410–1410 | False |
| arxiv | bloom | 16 | 1.0 | 51.2 | 52.4 | 1.022 [1.013, 1.030] | 72.1 | 11.9 | 1410–1410 | False |
| arxiv | exact | 1 | 1.0 | 9.1 | 9.2 | 1.041 [0.985, 1.099] | 25.2 | 19.8 | 1140–1140 | False |
| arxiv | exact | 16 | 1.0 | 51.8 | 52.8 | 1.019 [1.015, 1.024] | 77.0 | 116.6 | 1140–1140 | False |

The keep rule required, on both shapes at bs 16, a p = 1.0 upper bound ≤ 1.03 and a
p ≤ 0.01 upper bound < 1.0. It **fails**: goodreads p = 1.0 reaches 1.035 (bloom) and
1.056 (exact), and arXiv exact at p = 0.01 is 1.000 [0.992, 1.008]. The kernel was
reverted. Against the prediction:

- p = 0.001: **≈ 30 % faster** on both shapes and both modes (0.69–0.71), at the top of
  the predicted 10–30 % range;
- p = 0.01: within a few percent (0.95–1.00), as predicted;
- p ≥ 0.1: **slower by 1–6 %**, more than the predicted ≤ 2 %. arXiv exact at p = 0.1
  is +5.8 %;
- bs 1: no gain at any p.

Both kernels already mask the code load with `keep`, so the HBM bytes of failing items
were saved before this change: `before` at p = 0.001 is already 18–39 % faster than at
p = 1.0 (bs 16). Meta's scorer gains far more from selectivity. On the arXiv-like
layout at bs 16 it drops from 72–77 µs at p = 1.0 to 23–27 µs at p ≤ 0.01, and there
it beats ours (31–36 µs), while ours is faster at p ≥ 0.1. The one constant behind
that gain is p = 0.01, where the scorer still runs about 3× faster. That fits a skip
unit much finer than our 256-lane tile, such as a 32-document mask word, all-fail
with probability `0.99³² ≈ 0.72`, against 0.08 for a 256-lane tile. **Not
validated**: we did not inspect Meta's kernel for it. A finer-grained skip (a
narrower tile for the filtered scorers, or skipping per sub-tile) would be the next
experiment. It needs a retune, which this step excluded. The h2h's arXiv bloom gap
(`../../kernel-opt/h2h.md`: ours 105 vs official 84.5 µs scorer+mask on the real
layout) is consistent with this, but was not re-measured here.

Raw JSONs (`goodreads.json`, `arxiv.json`, and the A/A run `aa_goodreads.json` taken
before the change, all ratios 0.98–1.03) were written to `/scratch/cv2-lib/` on the pod.
