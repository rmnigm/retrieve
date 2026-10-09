# V2-HIGHP — LiNR V2's `fused_masked_knn_topk` at wide embeddings

Roadmap step V2-HIGHP (campaign-v2.3 bundle). At campaign-v2.1, Fix A's grid-strided body made V2 1.43-1.44× slower than
campaign-v2's straight-line body at PubMed 10 M d768 and p 0.9993, at any `programs` (pod 1, `artifacts/v2-crosstree`).
At low p, Fix A was about 5× faster (pod b, `.chains/v-pubmed/2026-10-09-012000000-URGENT-v2-high-p-regression.md`).
Current state: [validation](../../../validation.md) row *V2-HIGHP*; mechanism:
[kernels](../../../system/kernels.md#fused_masked_knn_topk--prefilterknn-sparse-path) ("Wide D").
A100-SXM4-80GB (pod b, GPU 0), torch 2.10.0+cu128, triton 3.6.0. Packages for before / old: `../st-ids/make_pkg.sh`
(`retrieve_v22` = campaign-v2.2, `retrieve_v20` = campaign-v2). Raw outputs: Hub `artifacts/v2-highp/` ([hub-index](../../hub-index.md)).

## Finding the mechanism ([`fmkt_variants.py`](fmkt_variants.py))

Every variant was measured kernel-only (10 M d768, p 0.0002-1, bs 1 / 16, ABAB) and was `torch.equal` to v2.2's scores.
The predictions are in `.chains/st-dloop/` (two plan notes; the stop-check corrected the first).

| variant | p ≤ 0.001 | p 1 | reading |
|---|---|---|---|
| A: v2.2 body, `num_stages` 1 | 1.00 | 1.00 | not software pipelining |
| C: campaign-v2 straight-line, verbatim | 26-30× | 0.66-0.69 | the high-p speed, and 5 M near-empty CTAs at low p |
| B: straight-line + early exit past `count` | 1.46-1.61 | 0.66-0.69 | the exit helps, but the CTA count remains |
| D: strided, `q` per tile | 0.93 | 0.79 | holding `q` costs registers |
| E: v2.2 body at `maxnreg` 64 | 1.01-1.02 | 0.60-0.70 | **registers / occupancy**: 98-112 → 64 regs, 2 → 4 CTAs an SM |
| Gq64: split loops, `q` per tile, `maxnreg` 64 | 0.70-0.73 | 0.61-0.70 | the split recovers low p |
| H: split loops, 1,024-lane fill, `block_n` 8, 4 warps, no cap, `programs` 864 | 0.16-0.22 | 0.47-0.72 | but **1.21** on a skewed batch (below) |
| **H at `programs` 3456** (shipped) | **0.21-0.24** | **0.68-0.70** | skewed batch **0.45** |

**Skewed batches.** The first real-PubMed after-round failed the keep rule on `all5` bs 16 (1.026). Its per-query pass
counts are skewed (oracle: median 142, p90 1.2 M, max 6.9 M per row), so a bs-16 batch's work sits in one or two rows. At
`programs` 864 such a row gets 54 programs, and H's small tile then has too little per-program throughput. Reproduced
synthetically (2 rows at p 0.12, 14 at 0.00002): H 1.21 at 864 programs; 0.66 / 0.53 / 0.45 / 0.39 at 1,728 / 2,592 /
3,456 / 13,824. The uniform p 1 bs 16 cell goes 0.47 → 0.59 / 0.70 / 0.69 / 0.73 over the same steps. 3,456 was chosen for
skewed real batches.

No body spills. `maxnreg` was rejected because inductor drops it: the higher-order op carries `maxnreg: 64`, but the
generated launcher keeps only `num_warps` / `num_stages`, so the compiled kernel ran at 112 registers. In the first
real-PubMed pair, graph mode was 0.86 against eager's 0.71. H uses only options inductor honors. Its wide `-inf` fill is
most of the low-p gain: v2.2's fill walks 32-lane tiles.

## The change

`SPLIT = next_pow2(D) > 256` (constexpr). True: the strided grid split into a scoring loop over the row's counted tiles
(`q` reloaded per tile) and a `-inf` fill over `[cdiv(count, BLOCK_N) · BLOCK_N, P)` in `FILL_N` = 1,024-lane chunks.
`config_for_width` picks `WIDE_CONFIG` (`block_n` 8, `num_warps` 4, `programs` 3456). False: v2.2's body, unchanged.

## Gates (final tree)

- **Bit-exact** ([`fmkt_gate.py`](fmkt_gate.py) `exact`): **64 / 64** cells, ids and scores `torch.equal` to v2.2 through the
  public op and the bucketed `_impl`, at 0.8 M d128, 3 M d128 and 10 M d768 × p {0, 0.001, 0.01, 0.1, 1} × bs {1, 16} × k
  {100, 1000}, plus the skewed cell at d768. Per-row Bernoulli(p) candidates in `clause_compact`'s `[B, N]` shape, garbage
  past `counts`.
- **D 128 / 192 SASS-identical** to v2.2 (4 / 4 cubins, [`sass_fmkt.py`](sass_fmkt.py)).
- **Library suite** 788 passed on pod b on a fresh inductor cache, including a new gate test: LiNR V2 at D 768 under
  `reduce-overhead`, captured with 0 cudagraph skips and replayed equal to eager.
- **Keep rule, kernel-only** ([`fmkt_gate.py`](fmkt_gate.py) `time`, windows before / after / campaign-v2 / after, 10 pairs,
  95 % t-interval):

| N | D | p | bs | v2.2 ms | after ms | campaign-v2 ms | after / v2.2 [95 % CI] | after / campaign-v2 [95 % CI] | scores equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 M | 128 | 0.001 | 1 | 0.014 | 0.014 | 0.069 | 0.959 [0.946, 0.973] | 0.200 [0.199, 0.201] | yes | 1140-1140 |
| 0.8 M | 128 | 0.001 | 16 | 0.170 | 0.170 | 1.062 | 0.997 [0.995, 0.999] | 0.160 [0.160, 0.160] | yes | 1140-1140 |
| 0.8 M | 128 | 0.01 | 1 | 0.015 | 0.014 | 0.069 | 0.965 [0.954, 0.976] | 0.209 [0.208, 0.210] | yes | 1140-1140 |
| 0.8 M | 128 | 0.01 | 16 | 0.192 | 0.191 | 1.063 | 0.998 [0.997, 0.999] | 0.181 [0.180, 0.181] | yes | 1140-1200 (unstable) |
| 0.8 M | 128 | 0.1 | 1 | 0.020 | 0.020 | 0.068 | 0.976 [0.970, 0.983] | 0.294 [0.291, 0.297] | yes | 1200-1200 |
| 0.8 M | 128 | 0.1 | 16 | 0.259 | 0.258 | 0.994 | 0.993 [0.979, 1.006] | 0.257 [0.254, 0.259] | yes | 1200-1380 (unstable) |
| 0.8 M | 128 | 1.0 | 1 | 0.137 | 0.139 | 0.132 | 1.007 [1.004, 1.011] | 1.054 [1.050, 1.057] | yes | 1380-1380 |
| 0.8 M | 128 | 1.0 | 16 | 0.988 | 0.986 | 2.045 | 1.001 [0.997, 1.005] | 0.484 [0.480, 0.488] | yes | 1380-1410 |
| 3 M | 128 | 0.001 | 1 | 0.043 | 0.042 | 0.251 | 0.989 [0.984, 0.993] | 0.168 [0.167, 0.168] | yes | 1140-1140 |
| 3 M | 128 | 0.001 | 16 | 0.505 | 0.505 | 3.207 | 1.000 [0.997, 1.002] | 0.156 [0.154, 0.158] | yes | 1215-1410 (unstable) |
| 3 M | 128 | 0.01 | 1 | 0.037 | 0.037 | 0.204 | 0.988 [0.982, 0.994] | 0.179 [0.179, 0.180] | yes | 1410-1410 |
| 3 M | 128 | 0.01 | 16 | 0.571 | 0.570 | 3.269 | 0.999 [0.998, 1.000] | 0.174 [0.174, 0.175] | yes | 1410-1410 |
| 3 M | 128 | 0.1 | 1 | 0.084 | 0.084 | 0.235 | 1.011 [1.005, 1.017] | 0.355 [0.355, 0.356] | yes | 1410-1410 |
| 3 M | 128 | 0.1 | 16 | 1.011 | 1.010 | 3.717 | 0.999 [0.998, 1.000] | 0.272 [0.271, 0.272] | yes | 1410-1410 |
| 3 M | 128 | 1.0 | 1 | 0.506 | 0.508 | 0.483 | 1.004 [1.002, 1.006] | 1.051 [1.050, 1.052] | yes | 1410-1410 |
| 3 M | 128 | 1.0 | 16 | 3.834 | 3.835 | 7.658 | 1.000 [0.997, 1.003] | 0.500 [0.498, 0.502] | yes | 1410-1410 |
| 10 M | 768 | 0.001 | 1 | 0.176 | 0.037 | 4.566 | 0.210 [0.208, 0.212] | 0.008 [0.008, 0.008] | yes | 1410-1410 |
| 10 M | 768 | 0.001 | 16 | 2.618 | 0.618 | 72.966 | 0.236 [0.236, 0.236] | 0.008 [0.008, 0.009] | yes | 1410-1410 |
| 10 M | 768 | 0.01 | 1 | 0.311 | 0.125 | 4.615 | 0.403 [0.402, 0.404] | 0.027 [0.027, 0.027] | yes | 1410-1410 |
| 10 M | 768 | 0.01 | 16 | 4.502 | 1.910 | 73.794 | 0.424 [0.424, 0.424] | 0.026 [0.026, 0.026] | yes | 1410-1410 |
| 10 M | 768 | 0.1 | 1 | 1.496 | 0.967 | 5.012 | 0.646 [0.646, 0.647] | 0.193 [0.193, 0.193] | yes | 1410-1410 |
| 10 M | 768 | 0.1 | 16 | 23.597 | 15.556 | 79.874 | 0.659 [0.659, 0.659] | 0.195 [0.195, 0.195] | yes | 1410-1410 |
| 10 M | 768 | 1.0 | 1 | 13.396 | 9.391 | 8.854 | 0.701 [0.701, 0.701] | 1.061 [1.060, 1.061] | yes | 1410-1410 |
| 10 M | 768 | 1.0 | 16 | 206.313 | 141.363 | 141.549 | 0.686 [0.684, 0.688] | 0.999 [0.997, 1.002] | yes | 1410-1410 |
| 10 M | 768 | 2e-5, 2 rows at 0.12 | 1 | 1.758 | 1.156 | 5.108 | 0.657 [0.657, 0.658] | 0.226 [0.226, 0.227] | yes | 1410-1410 |
| 10 M | 768 | 2e-5, 2 rows at 0.12 | 16 | 6.929 | 3.095 | 74.036 | 0.447 [0.446, 0.448] | 0.042 [0.042, 0.042] | yes | 1410-1410 |


At d768 every cell is faster than v2.2 (0.21-0.70, the skewed cell 0.45-0.66). Against campaign-v2's straight-line body it is
faster everywhere but p 1 bs 1 (1.06) and equal at p 1 bs 16 (1.00). At d128 the code is SASS-identical and the ratios scatter in both directions (0.963-1.013): a few bs 1 CIs exclude 1
either way, which is the method's order effect on identical code.

- **End to end, real PubMed d768** (V2 Triton clause; [`suites.yaml`](suites.yaml), [`summarize_bench.py`](summarize_bench.py)):
  one before process (campaign-v2.2) and one after process, ids `ids_sha256` equal:

| sweep | pass rate | bs | mode | V2 before ms | V2 after ms | after / before [95 % CI] | ids equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|
| c3_journal_reverse | 0.9993 | 1 | eager | 15.011 | 11.009 | 0.733 [0.733, 0.734] | yes | 1410-1410 |
| c3_journal_reverse | 0.9993 | 1 | graph | 14.961 | 10.963 | 0.733 [0.733, 0.733] | yes | 1410-1410 |
| c3_journal_reverse | 0.9993 | 16 | eager | 225.555 | 167.196 | 0.741 [0.741, 0.741] | yes | 1410-1410 |
| c3_journal_reverse | 0.9993 | 16 | graph | 225.499 | 167.148 | 0.741 [0.741, 0.742] | yes | 1410-1410 |
| all5 | 0.0181 | 1 | eager | 1.620 | 1.479 | 0.913 [0.913, 0.913] | yes | 1410-1410 |
| all5 | 0.0181 | 1 | graph | 1.572 | 1.431 | 0.910 [0.909, 0.911] | yes | 1410-1410 |
| all5 | 0.0181 | 16 | eager | 21.938 | 16.698 | 0.761 [0.759, 0.763] | yes | 1410-1410 |
| all5 | 0.0181 | 16 | graph | 21.866 | 16.634 | 0.761 [0.758, 0.763] | yes | 1410-1410 |
| c0_mesh | 0.0002 | 1 | eager | 1.620 | 1.478 | 0.913 [0.912, 0.913] | yes | 1410-1410 |
| c0_mesh | 0.0002 | 1 | graph | 1.573 | 1.431 | 0.910 [0.910, 0.910] | yes | 1410-1410 |
| c0_mesh | 0.0002 | 16 | eager | 15.468 | 13.506 | 0.873 [0.872, 0.875] | yes | 1410-1410 |
| c0_mesh | 0.0002 | 16 | graph | 15.408 | 13.464 | 0.874 [0.873, 0.875] | yes | 1410-1410 |

The before process is campaign-v2.2's (its inductor cache only ever held v2.2). The after process ran under the
harness's per-`code_version` inductor cache (`files:dfd6917b`, the uncommitted final tree). Earlier after-rounds were run
with a fixed `TORCHINDUCTOR_CACHE_DIR`, and their graph mode replayed an older variant's compiled graph (testing.md §
Running). They are void, and the phase script no longer sets it ([`recheck.sh`](recheck.sh) is the rerun). One process
pair, not interleaved within a process: the kernel-level grid above is the interleaved gate.

## What goes stale on adoption

**V2 Triton perf at `D_PAD` > 256** (PubMed d768). d128 / d192 V2 records stay valid (identical code). Quality is unchanged
(bit-exact).

## Improvement found, not applied

**The wide `-inf` fill at `D_PAD ≤ 256`.** v2.2's strided body fills `-inf` tile by tile (32 lanes). At d768 the 1,024-lane
fill alone took the low-p kernel from about 0.70 to 0.16-0.22 of v2.2 (Gq64 → H, the same scoring). At d128 the low-p
kernel is probably fill-bound as well. Expected: a large cut of V2's low-p kernel time at d128. Price: re-timing the d128 /
d192 V2 records (the SASS changes).

## Scripts

| script | what |
|---|---|
| [`fmkt_variants.py`](fmkt_variants.py) | the variant sweep (A-H, register caps, tile shapes) against v2.2, kernel-only |
| [`fmkt_gate.py`](fmkt_gate.py) | `exact` (bit-exact against v2.2) and `time` (the keep-rule grid against v2.2 and campaign-v2) |
| [`sass_fmkt.py`](sass_fmkt.py) | SASS hashes of the kernel at D 128 / 192, per tree |
| [`summarize_bench.py`](summarize_bench.py), [`suites.yaml`](suites.yaml) | the real PubMed V2 before / after table |
| [`phase.sh`](phase.sh), [`recheck.sh`](recheck.sh) | the GPU phase, and the fresh-cache rerun of the suites and the after-round |
