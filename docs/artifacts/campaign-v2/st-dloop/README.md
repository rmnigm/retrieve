# ST-DLOOP — the SilverTorch probe scorers against Meta's kernels at wide embeddings

Roadmap step ST-DLOOP (user, 2026-10-09; widened by the controller the same day to add a filter skip).
At campaign-v2.1, D3 PubMed `bloomwidth-timed` (d768, bloom `c0_mesh`, bs 16) had Triton at 2.84 ms
eager against official at 1.99 ms, the only dataset where Triton lost (surprise gate). Current state:
[validation](../../../validation.md) row *ST-DLOOP*; mechanism: [kernels](../../../system/kernels.md#silvertorch-kernels).
A100-SXM4-80GB (pod b, GPU 0), torch 2.10.0+cu128, triton 3.6.0. SM clocks cannot be locked; they are sampled per
window. Raw outputs are on the Hub under `artifacts/st-dloop/` ([hub-index](../../hub-index.md)).

## 1. Architecture: Meta's scorer against ours

Meta: `meta-recsys/silvertorch` @ 21aa35e, `fused_kmean_ann_cuda.cu` and `simple_index_mm.cuh`, as our adapter
calls them (`fused_kmean_ann` and `..._with_partial_masks`, `query_on_shared_mem = false`).
Ours: `ops/triton/codesigned_probe_score*.py` at campaign-v2.1.

| phase | Meta (official) | ours (Triton) |
|---|---|---|
| query quantize | ours (the adapter calls `quantize_int8`) | `quantize_int8`, eager torch ops before the launch |
| centroid / probe selection | ours (`q @ centroids^T`, `topk`), the same in both arms | the same |
| work decomposition | host: `round_cluster_to_warp`, two cumsums, `repeat_interleave`, two payload kernels → one 16-byte `WarpPayload` per 32 docs of a probed cluster, plus per-doc payloads for each cluster's tail; one host sync on `total_needed_warps` | none on the host: each program builds its row's tile table in-kernel (`probe_tile`, cluster-aligned tiles of `BLOCK_P`) |
| scoring unit | **one document per thread**, a warp = 32 consecutive documents, `kBlockSize` 256, grid capped at 192 × SMs | **one tile of `BLOCK_P` documents per program**, the tile's int8 codes into one `tl.dot` (IMMA, M = 1 padded to 16) |
| loop over D | `DIM` a template (16 … 768 instantiated); `#pragma unroll 1` over 256-byte chunks, each 16 `int4` loads + 64 `dp4a` into one int32 register; D = 768 = 3 chunks, no padding | v2.1: **none**, the whole `[BLOCK_P, D_PAD]` tile in one dot, D = 768 padded to 1024 (a quarter zero lanes), a `64 × 4` tile to avoid spilling |
| accumulator | int32 register per thread; fp16 out (`/ divisor`) or int32 | int32 from the dot; fp32 out (`· q_scale · global_scale`) |
| filter skip unit | **one document**: a thread whose mask bit is 0 returns before any load or dp4a and writes nothing (`indices` pre-filled with -1) | v2.1: the code *load* of a failing lane is masked, but the tile's dot always runs |
| phase 2 (bloom) | `bloom_index_search_batch_return_partial_response` over the probed clusters → 32-bit masks per warp payload | fused: the transposed bloom test per lane (`C·k_hash` 8-byte words, one per 64 items) |
| top-k | ours, `masked_topk` over `[B, width]` (the same cost in both arms) | `torch.topk` over `[B, width]`, then `probe_ids_kernel` |
| launches per call (eager, PubMed bloom) | 101 (with the adapter's epilogue), host-synchronous | 53 |

How each scales: Meta's scorer costs O(passing docs × D) plus one thread per probed document, and its host prep is O(B ·
n_probe). Ours (v2.1) cost O(probed docs × D_PAD) in the dot whatever the pass rate, and only the code bytes fell with p.

**Profile, v2.1 tree, the real PubMed d768 cell** (n_lists 1024, n_probe 24, bs 16, k 100, seed 0; `real_cell.py
profile`; sm 1410 on every window):

| cell | Triton scorer | official scorer | Triton wall | official wall (int32 / fp16) |
|---|---|---|---|---|
| bloom `c0_mesh` (p 0.0002) | 2,391 µs | 72 µs | 2.94 ms | 2.23 / 2.18 ms |
| none | 2,682 µs | 4,560 / 4,181 µs | 3.21 ms | 6.09 / 5.67 ms |

**Reading.** Unfiltered, our scorer is already 1.7× faster than Meta's at d768. The bloom gap came from skip granularity,
not from D: at a 0.0002 pass rate Meta's scorer drops 60× (it touches only passing documents), while ours drops 11 %.
The D padding was a smaller second cost (a quarter of the dot on zero lanes, and the narrow tile it forced).

## 2. The change

- **D loop** (`common.probe_dots`). Past `D_PAD` 256: `BLOCK_D` chunks into one int32 accumulator (`tl.dot(acc=)`,
  `tl.range` pipelined by `num_stages`). At `D_PAD ≤ 256`: the v2.1 single dot, in v2.1's load order.
- **Tile skip** (`SKIP`, filtered scorers, `D_PAD > 256` only): after the filter, `tl.max(keep) == 0` → store `-inf`,
  no code loads, no dot. A constexpr from the width's `CONFIGS` entry, so there is no host sync and graph capture is
  unchanged. Off at ≤ 256: the forced-skip grid below reproduces #16 there.
- **Tiles per width, then per program count.** `CONFIGS[1024]` is ordered `256 × 4 / BLOCK_D 128`, `128 × 4 / 256`,
  `64 × 4 / 256` (2 stages, skip), and `_host.tile_for_width(configs, D, B, width)` takes the first tile giving at
  least `MIN_PROGRAMS = 1024` programs. Exact: one tile, `128 × 4 / 128`.
- `tune-kernels` sweeps `(block_p, num_warps, num_stages, block_d)` with the skip on past `D_PAD` 256 (`_CPS_WIDE_GRID`).

## 3. Gates

Every gate below ran on the final tree (`phase.sh` + `phase2.sh` into one output directory, then `real_gate.py`
one mode per process). Before = the campaign-v2.1 library (`92af9b6`; its library source equals staging `45266ab`'s).

**Bit-exact (CLAUDE.md rule 3), all green:**
- Synthetic (`dloop_gate.py exact`): **312 / 312** cells `torch.equal` on ids and scores at D 128, 192 and 768.
  Each width covers none / bloom / exact × p {0, 0.001, 0.01, 0.1, 1.0}, plus an all-inactive query (p = 0: every slot
  `-inf`, every d768 tile skipped) × bs {1, 16} × k {100, 1000} × n_probe {24, 1024}. N 2 M Gaussian, n_lists 1024.
- Real PubMed d768 (`real_gate.py`): **32 / 32**: none, bloom `c0_mesh`, exact `c0_mesh` and `all5` × bs × k × n_probe
  (as above), 4 query batches each.
- End to end: the `ids_sha256` of every Triton perf entry in the interleaved bench (below) is identical before / after
  (16 / 16 cell × mode pairs, eager and graph).
- Tile sweeps: every config of the 164 swept on the real cell was `torch.equal` to the shipped one.

**`D_PAD ≤ 256` is the v2.1 code:** SASS identical at D = 128 and 192, 7 of 7 cubins (`pubmed-fixes/sass_identity.py`).
Routing the dot through a helper first changed every scorer cubin, for two reasons. (1) The `q_scale` scalar load moved
after the code-tile load, which reordered scheduling and register allocation (the instruction multiset was unchanged).
(2) At D = 192, `q_scales_ptr + bid` was computed at the call site, before the q_codes load (TTIR). `probe_dots` now
loads the scale between the two tiles and computes its address itself. So no d128 / d192 record goes stale.

**Library suite** on pod b's GPU: 781 passed (`pytest tests/`, the full library suite, parity files included).
**Graph capture:** the `reduce-overhead` replay tests in the suite pass unchanged. The bench's graph mode ran every
Triton cell, with ids equal to eager's (the same `ids_sha256` as v2.1's).

**Keep rule, interleaved, 95 % CI: holds at every width × pass rate × bs.**

*The real D3 cell*: PubMed d768 (n_lists 1024, n_probe 24, k 100, bloom 1024 / 5, seed 0). `bench run --interleave`
puts Triton with official in each process. Two rounds, before r1, after r1, before r2, after r2. Each arm's 6 window
medians, Welch interval on logs (`summarize_bench.py`):

| kind | sweep | bs | mode | Triton before ms | Triton after ms | after / before [95 % CI] | official ms | official / after [95 % CI] | ids before = after | sm_mhz |
|---|---|---|---|---|---|---|---|---|---|---|
| bloom | c0_mesh | 1 | eager | 0.722 | 0.723 | 1.003 [0.995, 1.011] | 1.388 | 1.92 [1.91, 1.93] | yes | 1410-1410 |
| bloom | c0_mesh | 1 | graph | 0.325 | 0.209 | 0.644 [0.640, 0.647] | – | – | yes | 1410-1410 |
| bloom | c0_mesh | 16 | eager | 2.825 | 0.901 | 0.319 [0.319, 0.319] | 2.164 | 2.40 [2.39, 2.41] | yes | 1410-1410 |
| bloom | c0_mesh | 16 | graph | 2.721 | 0.793 | 0.291 [0.291, 0.292] | – | – | yes | 1410-1410 |
| clause | all5 | 1 | eager | 0.507 | 0.515 | 1.011 [0.996, 1.026] | – | – | yes | 1410-1410 |
| clause | all5 | 1 | graph | 0.326 | 0.230 | 0.707 [0.706, 0.707] | – | – | yes | 1410-1410 |
| clause | all5 | 16 | eager | 2.800 | 1.179 | 0.421 [0.421, 0.422] | – | – | yes | 1410-1410 |
| clause | all5 | 16 | graph | 2.730 | 1.108 | 0.406 [0.406, 0.406] | – | – | yes | 1410-1410 |
| clause | c0_mesh | 1 | eager | 0.511 | 0.516 | 1.012 [0.999, 1.025] | – | – | yes | 1410-1410 |
| clause | c0_mesh | 1 | graph | 0.326 | 0.233 | 0.716 [0.715, 0.717] | – | – | yes | 1410-1410 |
| clause | c0_mesh | 16 | eager | 2.787 | 1.183 | 0.424 [0.424, 0.425] | – | – | yes | 1410-1410 |
| clause | c0_mesh | 16 | graph | 2.716 | 1.112 | 0.409 [0.408, 0.410] | – | – | yes | 1410-1410 |
| none | full_scan | 1 | eager | 0.406 | 0.407 | 1.002 [0.997, 1.006] | 0.769 | 1.89 [1.89, 1.89] | yes | 1410-1410 |
| none | full_scan | 1 | graph | 0.347 | 0.345 | 0.993 [0.991, 0.994] | – | – | yes | 1410-1410 |
| none | full_scan | 16 | eager | 3.135 | 3.057 | 0.975 [0.975, 0.976] | 5.530 | 1.81 [1.81, 1.81] | yes | 1410-1410 |
| none | full_scan | 16 | graph | 3.068 | 2.995 | 0.976 [0.975, 0.977] | – | – | yes | 1410-1410 |

Official's bloom bs 1 windows sampled 1245 MHz (`unstable`, all four rounds). Every Triton window and every other
official window held 1410. Before = the surprise cell: Triton eager 2.825 ms against official 2.164 ms (official /
Triton 0.77). After: **0.901 ms, official / Triton 2.40**. Eager bs 1 is host-bound (wall 0.4-0.7 ms against 0.2-0.35 ms
of GPU in graph mode); its CIs include 1.0. A first round alone had shown +3 to +6 % there, which did not repeat.

*Synthetic grid* (`dloop_gate.py time`, #16's method: kernel-only CUDA graphs of 64 launches over 8 query batches, ABAB
windows, 12 pairs; index as above, probe width 64,275 / 51,738 / 50,947 at D 768 / 192 / 128; one clause, every
query asks 1, p = the item pass rate):

**shipped** — scorer kernel µs (kernel-only CUDA graph), after / before [95 % CI], Meta's scorer + its phase-2 mask µs (reference):

| D | mode | p | bs | before | after | after / before [95 % CI] | official scorer + mask | ids+scores equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|---|
| 768 | bloom | 0.001 | 1 | 7.2 | 5.4 | 0.748 [0.745, 0.751] | 11.3 + 7.9 | yes | 1410-1410 |
| 768 | bloom | 0.001 | 16 | 58.0 | 17.7 | 0.308 [0.297, 0.320] | 30.7 + 14.2 | yes | 1140-1410 (unstable) |
| 768 | exact | 0.001 | 1 | 12.5 | 5.9 | 0.472 [0.469, 0.474] | 15.1 + 16.0 | yes | 1140-1140 |
| 768 | exact | 0.001 | 16 | 62.5 | 30.5 | 0.487 [0.487, 0.488] | 32.2 + 79.0 | yes | 1140-1140 |
| 768 | bloom | 0.01 | 1 | 7.6 | 6.9 | 0.906 [0.899, 0.913] | 14.3 + 8.3 | yes | 1410-1410 |
| 768 | bloom | 0.01 | 16 | 58.2 | 18.8 | 0.322 [0.322, 0.323] | 28.5 + 11.8 | yes | 1410-1410 |
| 768 | exact | 0.01 | 1 | 12.4 | 9.1 | 0.734 [0.729, 0.739] | 21.9 + 16.1 | yes | 1140-1410 (unstable) |
| 768 | exact | 0.01 | 16 | 62.6 | 31.3 | 0.501 [0.501, 0.502] | 35.7 + 79.0 | yes | 1140-1140 |
| 768 | bloom | 0.1 | 1 | 7.9 | 7.4 | 0.946 [0.939, 0.953] | 16.9 + 8.0 | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.1 | 16 | 58.8 | 24.4 | 0.415 [0.414, 0.415] | 32.6 + 11.9 | yes | 1410-1410 |
| 768 | exact | 0.1 | 1 | 12.5 | 10.3 | 0.830 [0.825, 0.835] | 29.0 + 16.0 | yes | 1140-1410 (unstable) |
| 768 | exact | 0.1 | 16 | 62.8 | 35.3 | 0.563 [0.562, 0.563] | 40.1 + 79.0 | yes | 1140-1140 |
| 768 | bloom | 1.0 | 1 | 8.2 | 7.8 | 0.937 [0.900, 0.975] | 22.8 + 8.3 | yes | 1140-1410 (unstable) |
| 768 | bloom | 1.0 | 16 | 59.8 | 29.4 | 0.491 [0.491, 0.492] | 43.6 + 12.2 | yes | 1410-1410 |
| 768 | exact | 1.0 | 1 | 13.5 | 10.9 | 0.806 [0.801, 0.811] | 35.9 + 16.2 | yes | 1140-1410 (unstable) |
| 768 | exact | 1.0 | 16 | 63.7 | 38.5 | 0.605 [0.603, 0.607] | 50.4 + 79.0 | yes | 1140-1140 |
| 768 | none | – | 1 | 8.5 | 7.4 | 0.875 [0.862, 0.887] | 26.8 + 0.0 | yes | 1140-1140 |
| 768 | none | – | 16 | 70.6 | 34.5 | 0.489 [0.488, 0.489] | 48.2 + 0.0 | yes | 1140-1140 |
| 192 | bloom | 0.001 | 1 | 7.6 | 7.6 | 1.004 [1.000, 1.009] | 15.2 + 9.1 | yes | 1410-1410 |
| 192 | bloom | 0.001 | 16 | 54.7 | 54.7 | 1.001 [1.000, 1.002] | 28.3 + 14.3 | yes | 1140-1410 (unstable) |
| 192 | exact | 0.001 | 1 | 8.2 | 8.3 | 1.004 [0.999, 1.010] | 20.0 + 16.0 | yes | 1140-1140 |
| 192 | exact | 0.001 | 16 | 62.2 | 62.2 | 1.000 [0.999, 1.001] | 29.7 + 78.9 | yes | 1140-1140 |
| 192 | bloom | 0.01 | 1 | 7.8 | 7.8 | 0.989 [0.955, 1.025] | 15.7 + 9.1 | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.01 | 16 | 57.4 | 57.3 | 0.999 [0.998, 1.000] | 28.4 + 11.8 | yes | 1410-1410 |
| 192 | exact | 0.01 | 1 | 8.3 | 8.3 | 1.023 [0.980, 1.068] | 21.6 + 16.0 | yes | 1140-1140 |
| 192 | exact | 0.01 | 16 | 62.7 | 62.8 | 1.001 [0.999, 1.002] | 35.1 + 78.9 | yes | 1140-1140 |
| 192 | bloom | 0.1 | 1 | 8.0 | 8.0 | 1.004 [0.996, 1.011] | 16.7 + 9.2 | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.1 | 16 | 63.5 | 63.5 | 1.000 [0.999, 1.001] | 41.4 + 12.0 | yes | 1410-1410 |
| 192 | exact | 0.1 | 1 | 8.3 | 8.3 | 1.003 [0.998, 1.009] | 22.5 + 16.2 | yes | 1140-1140 |
| 192 | exact | 0.1 | 16 | 65.3 | 65.3 | 1.001 [0.999, 1.003] | 47.4 + 78.9 | yes | 1140-1140 |
| 192 | bloom | 1.0 | 1 | 13.2 | 13.2 | 1.001 [0.993, 1.008] | 22.8 + 10.9 | yes | 1410-1410 |
| 192 | bloom | 1.0 | 16 | 84.7 | 84.9 | 1.001 [1.000, 1.002] | 109.1 + 12.0 | yes | 1410-1410 |
| 192 | exact | 1.0 | 1 | 12.5 | 12.5 | 1.004 [0.997, 1.011] | 29.5 + 16.5 | yes | 1140-1410 (unstable) |
| 192 | exact | 1.0 | 16 | 83.8 | 83.7 | 0.999 [0.999, 1.000] | 111.3 + 78.9 | yes | 1140-1140 |
| 192 | none | – | 1 | 12.3 | 12.3 | 1.004 [1.000, 1.008] | 23.9 + 0.0 | yes | 1140-1140 |
| 192 | none | – | 16 | 80.9 | 80.9 | 0.999 [0.998, 1.001] | 110.3 + 0.0 | yes | 1140-1140 |
| 128 | bloom | 0.001 | 1 | 5.9 | 5.9 | 1.003 [0.997, 1.009] | 14.5 + 9.1 | yes | 1410-1410 |
| 128 | bloom | 0.001 | 16 | 36.9 | 41.2 | 1.017 [0.979, 1.057] | 28.1 + 14.1 | yes | 1140-1410 (unstable) |
| 128 | exact | 0.001 | 1 | 6.5 | 6.5 | 1.002 [0.992, 1.012] | 18.3 + 16.0 | yes | 1140-1140 |
| 128 | exact | 0.001 | 16 | 41.9 | 41.9 | 1.000 [0.999, 1.001] | 29.2 + 79.1 | yes | 1140-1140 |
| 128 | bloom | 0.01 | 1 | 6.3 | 6.3 | 1.005 [0.994, 1.015] | 15.2 + 9.1 | yes | 1410-1410 |
| 128 | bloom | 0.01 | 16 | 38.5 | 38.6 | 1.005 [1.002, 1.008] | 26.5 + 11.7 | yes | 1410-1410 |
| 128 | exact | 0.01 | 1 | 6.0 | 6.5 | 1.020 [0.975, 1.066] | 19.3 + 16.1 | yes | 1140-1410 (unstable) |
| 128 | exact | 0.01 | 16 | 42.4 | 42.4 | 0.999 [0.996, 1.001] | 33.4 + 79.2 | yes | 1140-1140 |
| 128 | bloom | 0.1 | 1 | 6.5 | 6.6 | 1.008 [0.993, 1.024] | 15.7 + 9.2 | yes | 1410-1410 |
| 128 | bloom | 0.1 | 16 | 43.4 | 43.5 | 1.001 [1.000, 1.002] | 35.7 + 12.0 | yes | 1410-1410 |
| 128 | exact | 0.1 | 1 | 6.5 | 6.5 | 1.006 [0.994, 1.017] | 20.3 + 16.3 | yes | 1140-1410 (unstable) |
| 128 | exact | 0.1 | 16 | 44.4 | 44.4 | 1.001 [0.999, 1.002] | 42.8 + 79.1 | yes | 1140-1140 |
| 128 | bloom | 1.0 | 1 | 10.8 | 10.9 | 1.006 [0.995, 1.017] | 21.6 + 10.9 | yes | 1410-1410 |
| 128 | bloom | 1.0 | 16 | 64.5 | 64.7 | 1.003 [1.002, 1.005] | 90.3 + 12.1 | yes | 1410-1410 |
| 128 | exact | 1.0 | 1 | 10.3 | 10.3 | 1.000 [0.990, 1.009] | 27.7 + 16.5 | yes | 1140-1410 (unstable) |
| 128 | exact | 1.0 | 16 | 66.0 | 65.8 | 0.998 [0.997, 1.000] | 96.3 + 79.2 | yes | 1140-1140 |
| 128 | none | – | 1 | 9.5 | 9.4 | 0.998 [0.991, 1.005] | 22.7 + 0.0 | yes | 1140-1140 |
| 128 | none | – | 16 | 60.9 | 61.0 | 1.001 [1.000, 1.002] | 95.6 + 0.0 | yes | 1140-1140 |

**forced skip (experiment, not shipped)** — scorer kernel µs (kernel-only CUDA graph), after / before [95 % CI], Meta's scorer + its phase-2 mask µs (reference):

| D | mode | p | bs | before | after | after / before [95 % CI] | official scorer + mask | ids+scores equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|---|
| 192 | bloom | 0.001 | 1 | 7.6 | 8.0 | 1.044 [1.039, 1.049] | 15.2 + 9.1 | yes | 1410-1410 |
| 192 | bloom | 0.001 | 16 | 54.7 | 31.9 | 0.583 [0.583, 0.584] | 23.0 + 11.8 | yes | 1410-1410 |
| 192 | exact | 0.001 | 1 | 8.3 | 8.7 | 1.062 [1.055, 1.070] | 20.0 + 16.1 | yes | 1140-1410 (unstable) |
| 192 | exact | 0.001 | 16 | 62.2 | 34.4 | 0.553 [0.552, 0.554] | 29.7 + 79.0 | yes | 1140-1140 |
| 192 | bloom | 0.01 | 1 | 7.8 | 8.3 | 1.062 [1.049, 1.075] | 15.7 + 9.1 | yes | 1410-1410 |
| 192 | bloom | 0.01 | 16 | 57.3 | 55.8 | 0.974 [0.973, 0.975] | 28.4 + 11.9 | yes | 1410-1410 |
| 192 | exact | 0.01 | 1 | 8.3 | 8.9 | 1.081 [1.074, 1.089] | 21.5 + 16.1 | yes | 1140-1410 (unstable) |
| 192 | exact | 0.01 | 16 | 62.7 | 60.9 | 0.971 [0.970, 0.972] | 35.1 + 79.0 | yes | 1140-1140 |
| 192 | bloom | 0.1 | 1 | 8.1 | 8.5 | 1.051 [1.041, 1.061] | 16.7 + 9.3 | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.1 | 16 | 63.5 | 65.1 | 1.026 [1.025, 1.027] | 41.4 + 12.0 | yes | 1410-1410 |
| 192 | exact | 0.1 | 1 | 8.4 | 9.0 | 1.098 [1.051, 1.148] | 22.6 + 16.1 | yes | 1140-1140 |
| 192 | exact | 0.1 | 16 | 65.3 | 67.4 | 1.032 [1.031, 1.034] | 47.5 + 79.0 | yes | 1140-1140 |
| 192 | bloom | 1.0 | 1 | 13.2 | 13.5 | 1.023 [1.013, 1.033] | 22.8 + 10.9 | yes | 1410-1410 |
| 192 | bloom | 1.0 | 16 | 84.7 | 86.8 | 1.025 [1.024, 1.026] | 109.3 + 12.1 | yes | 1410-1410 |
| 192 | exact | 1.0 | 1 | 12.5 | 13.3 | 1.069 [1.066, 1.072] | 29.5 + 16.6 | yes | 1140-1410 (unstable) |
| 192 | exact | 1.0 | 16 | 83.8 | 87.5 | 1.044 [1.043, 1.045] | 111.2 + 79.0 | yes | 1140-1140 |
| 192 | none | – | 1 | 12.3 | 12.4 | 1.007 [0.999, 1.015] | 24.1 + 0.0 | yes | 1140-1140 |
| 192 | none | – | 16 | 80.8 | 80.8 | 1.000 [0.999, 1.001] | 110.3 + 0.0 | yes | 1140-1140 |
| 128 | bloom | 0.001 | 1 | 5.9 | 6.2 | 1.045 [1.038, 1.052] | 14.5 + 9.1 | yes | 1410-1410 |
| 128 | bloom | 0.001 | 16 | 36.9 | 24.3 | 0.660 [0.658, 0.662] | 22.7 + 11.5 | yes | 1410-1410 |
| 128 | exact | 0.001 | 1 | 5.3 | 5.2 | 0.979 [0.969, 0.988] | 18.3 + 16.0 | yes | 1140-1410 (unstable) |
| 128 | exact | 0.001 | 16 | 41.7 | 27.4 | 0.656 [0.656, 0.657] | 29.1 + 79.0 | yes | 1140-1140 |
| 128 | bloom | 0.01 | 1 | 6.3 | 6.6 | 1.053 [1.035, 1.070] | 15.2 + 9.1 | yes | 1410-1410 |
| 128 | bloom | 0.01 | 16 | 38.4 | 38.2 | 0.995 [0.994, 0.996] | 26.5 + 11.7 | yes | 1410-1410 |
| 128 | exact | 0.01 | 1 | 6.5 | 6.5 | 1.019 [0.981, 1.059] | 19.3 + 16.0 | yes | 1140-1410 (unstable) |
| 128 | exact | 0.01 | 16 | 42.3 | 43.1 | 1.019 [1.018, 1.020] | 33.4 + 79.0 | yes | 1140-1140 |
| 128 | bloom | 0.1 | 1 | 6.5 | 6.8 | 1.036 [1.023, 1.048] | 15.7 + 9.2 | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.1 | 16 | 43.5 | 44.7 | 1.031 [1.026, 1.036] | 35.8 + 12.0 | yes | 1410-1410 |
| 128 | exact | 0.1 | 1 | 6.5 | 6.5 | 1.009 [1.000, 1.019] | 20.3 + 16.3 | yes | 1140-1140 |
| 128 | exact | 0.1 | 16 | 44.3 | 47.8 | 1.080 [1.078, 1.081] | 42.6 + 79.0 | yes | 1140-1140 |
| 128 | bloom | 1.0 | 1 | 10.9 | 11.1 | 1.019 [1.009, 1.030] | 21.7 + 10.9 | yes | 1410-1410 |
| 128 | bloom | 1.0 | 16 | 64.7 | 65.8 | 1.017 [1.015, 1.018] | 90.5 + 12.2 | yes | 1410-1410 |
| 128 | exact | 1.0 | 1 | 10.3 | 10.3 | 1.014 [0.985, 1.045] | 27.6 + 16.6 | yes | 1140-1410 (unstable) |
| 128 | exact | 1.0 | 16 | 65.9 | 67.2 | 1.020 [1.019, 1.022] | 96.0 + 79.0 | yes | 1140-1140 |
| 128 | none | – | 1 | 9.4 | 9.4 | 0.999 [0.986, 1.011] | 22.5 + 0.0 | yes | 1140-1140 |
| 128 | none | – | 16 | 60.9 | 61.0 | 1.002 [1.000, 1.003] | 95.6 + 0.0 | yes | 1140-1140 |


At d768 every cell is faster (worst upper bound 0.975). At D 128 / 192 (the v2.1 code) every CI includes 1.0. The
forced skip at D 128 / 192 reproduces #16: −34 to −45 % at p 0.001 bs 16, and +2 to +8 % at p ≥ 0.1. It stays off there.

**The bs 1 regression, found by this grid and fixed.** With the 256-item tile alone, d768 bs 1 ran 1.68× (none) and
1.27-1.51× (bloom, p ≥ 0.01) v2.1's time. Mechanism (`bs_sweep.py`, p 1.0): the ratio follows the program count. The
256-item tile launched 276 programs at bs 1 on the 64 K-wide layout, against v2.1's 1,029 with a 64-item tile. On 108 SMs a
sub-10 µs kernel then cannot hide latency (128 items: 527 programs, 1.24×; 64 items, `BLOCK_D` 256: 1,029 programs,
0.86×). From about 1,000 programs up every tile beat v2.1 (bs 4: 0.65-0.87×). Hence the program-count fallback.
Real PubMed's probe width is wide enough that it keeps the 256-item tile at every bs.

**Profile after** (`real_cell.py profile`, the real cell, eager, 50 calls; sm 1410):

| cell | arm | wall µs | kernel µs | scorer µs | top-k µs | launches |
|---|---|---|---|---|---|---|
| PubMed d768 bloom `c0_mesh` | Triton | 939 | 871 | **411** (v2.1: 2,391) | 355 | 53 |
| | official int32 / fp16 | 2,208 / 2,152 | 1,310 / 1,246 | 82 / 81 | 356 | 101 |
| PubMed d768 none | Triton | 3,119 | 3,018 | 2,592 (v2.1: 2,682) | 354 | 39 |
| | official int32 / fp16 | 6,080 / 5,630 | 5,773 / 5,329 | 4,576 / 4,196 | 352 | 80 |
| arXiv d128 bloom `c0_maincat` (v2.1 code) | Triton | 699 | 322 | 101 | 128 | 52 |
| | official int32 / fp16 | 1,636 / 1,662 | 661 / 652 | 86 / 86 | 155 | 100 |
| arXiv d128 none | Triton | 535 | 286 | 94 | 129 | 38 |
| | official int32 / fp16 | 1,023 / 1,045 | 607 / 612 | 119 / 120 | 150 | 79 |

Prediction (written before measuring, `.chains/st-dloop/` plan note) against result: the bloom scorer was predicted at
−60 to −75 % from the skip and measured at −83 % against v2.1 (D loop, tile and skip together). The whole cell was
predicted at 0.6-0.8 ms and measured at 0.90 ms: the top-k tail is 0.36 ms, more than I allowed for. At high p, ±1 % was
predicted and the d768 grid is 0.49-0.94×, faster: the D loop's wider tile helps at every p, which the prediction left out.


## 4. Improvements found, not applied (for the controller)

Measured or estimated; none applied here. Ordered by expected gain on the campaign's cells.

1. **The id epilogue's `[next_pow2(k), next_pow2(n_probe)]` tile** (`common.probe_ids_kernel`). At k 1000 × n_probe
   1024 it is 2^20 elements, and ptxas ran **about 11 min** on it (pod b, 01:17-01:28). The cubin it produced has
   **32 registers and a 122,376-byte stack per thread**. The driver sizes its local-memory pool for every resident
   thread (122 KB × 2,048 × 108 SMs ≈ 25 GB) and keeps it for the context: measured 25.5 GB in use outside PyTorch's
   allocator in every process that launched it, which pushed the real bloom gate out of memory until the fp32 item table
   was freed. Past k · n_probe = 2^20 the kernel does not compile at all. A loop over k in `[BLOCK_K, NPP]` tiles removes
   all three. **Taken as ST-IDS** (the controller's v2.2 bundle, item 2).
2. **The top-k tail** at low pass rates. After the skip it is 355 µs of the PubMed bloom cell's 871 µs of kernel time
   (41 %): `torch.topk` (radix, 4 × 3 kernels + a gather) over `[16, width]` slots, nearly all `-inf` at p 0.0002.
   Official pays the same 356 µs (`masked_topk`). A per-tile partial top-k in the scorer (keep each tile's best k, then
   a small top-k), or compacting finite slots first, would leave only work proportional to passing items. Estimate:
   up to −0.3 ms at the D3 cell (0.90 → about 0.6 ms); at p ≈ 1 the gain is small and the change costs registers, so it
   needs its own keep rule.
3. **A tile skip at `D_PAD ≤ 256` that engages only at low pass rate.** Prior art: #16
   ([tile-skip](../tile-skip/README.md), validation row *Probe-scorer tile skipping (#16)*): bit-exact, about −30 % at
   p 0.001, +1 to 6 % at p ≥ 0.1, reverted. This step's forced-skip grid at D 128 / 192 reproduces it: 0.66 / 0.58 at
   p 0.001 bs 16, and 1.02-1.08 at p ≥ 0.1. The cost at high p is the per-tile cross-warp `tl.max` plus the branch,
   paid on every tile. Engaging it only at low p needs a pass-rate estimate without a host sync. Options:
   - (a) a per-row estimate on the device from per-(clause, value) item counts kept at `register_index` (an AND's pass
     rate ≤ its rarest clause's), written by a small pre-kernel (or by the bloom bit-positions kernel) into a `[B]` flag.
     The scorer branches on one scalar per program, so high-p rows pay a scalar load, not a reduction. Risk: one more
     launch (≈ 5 µs eager, nothing in a graph); bloom false positives raise p a little; the estimate is a bound, so
     it engages only when safe.
   - (b) the host picks a constexpr from a caller hint (the sweep's selectivity): no device cost, but it is a new API
     surface and wrong hints cost either way.
   - (c) the per-tile vote alone (#16) fails the keep rule at D ≤ 256.
   Expected: −30 to −35 % of the d128 / d192 scorer at p ≤ 0.001 bs 16, about 0 elsewhere. Retiming every d128 / d192
   SilverTorch cell would be the price (the SASS changes).
4. **Separate tiles for none and bloom at wide D.** One `CONFIGS` entry serves both. The chosen `256 × 4 / BLOCK_D 128`
   is +1.5 % on none and **+12 %** on bloom (`_impl` 0.833 against 0.744 ms for `256 × 4 / BLOCK_D 64`, real cell,
   bs 16) against each mode's best. A per-`HAS_QB` entry would gain about 0.09 ms at the D3 cell. Low risk: a tile
   change, bit-exact by construction.
5. **Eager launch count and host cost.** Triton's eager bs 1 is host-bound: PubMed bloom 0.72 ms wall against 0.21 ms
   in graph mode; arXiv d128 bloom 0.70 against 0.32 ms of kernels. 52-53 launches a call: phase 1 (sgemm + topk), the
   quantize (several torch ops), the bloom bit positions (torch ops), the top-k (≈ 13), the id epilogue. Fusing the
   quantize and the bit positions into the scorer's prologue (each program needs one row) would cut launches by about
   10-15. Estimate: −0.05 to −0.15 ms eager bs 1. Graph mode would not change. Official is at 79-101 launches and
   host-synchronous, so this only widens a lead.
6. **D = 192 through the D loop** (YFCC). 192 pads to 256: a quarter of the dot on zero lanes, as 768 did. Three 64-wide
   chunks avoid that. **Not measured.** The ≤ 256 path is kept SASS-identical on purpose (the brief), and changing it
   would invalidate the d192 records. If tried: bit-exact by construction, gain likely below the d768 case (the d192 tile
   does not spill).
7. **Exact mode reads `C · A_max` int64 attrs per item** (160 B at C 5, A_max 4). At low p the code loads are skipped
   but the attribute loads are not, so they set the exact scorer's floor: PubMed exact bs 16 is 1.18 ms against bloom's
   0.90 ms. Narrow attrs (int32, or int16 where the vocabulary allows) would halve or quarter those bytes. Estimate: up
   to −0.15 ms at the cell. It needs an index-layout change (`item_clause_attrs` dtype) on every backend.
8. **A reference cycle in `SilverTorch`.** `_forward_impl` holds a bound method of the module, so a dropped module is
   freed only by the cycle collector (seen while building several 10 M indexes in one process). Harmless in the
   harness, which builds one module a cell. A `forward` that branches on `self.backend` would remove it.

## Scripts

| script | what |
|---|---|
| [`make_v21_pkg.sh`](make_v21_pkg.sh) | the campaign-v2.1 library as package `retrieve_v21` (op namespace `retrieve_v21::`), for before/after in one process |
| [`real_cell.py`](real_cell.py) | the real D3 cell: `profile` (torch.profiler kernel split, both backends) and `sweep` (tiles, each `torch.equal` to the shipped one) |
| [`dloop_gate.py`](dloop_gate.py) | synthetic pass rates: `exact` (the bit-exact gate) and `time` (the keep-rule grid against official, #16's ABAB method) |
| [`bs_sweep.py`](bs_sweep.py) | tile × batch size at d768, kernel-only against v2.1 (the program-count mechanism) |
| [`real_gate.py`](real_gate.py) | the bit-exact gate on PubMed's real sweeps |
| [`summarize_bench.py`](summarize_bench.py), [`suites.yaml`](suites.yaml) | the interleaved `bench run` before/after table |
| [`phase.sh`](phase.sh), [`phase2.sh`](phase2.sh) | the GPU phases as run |
