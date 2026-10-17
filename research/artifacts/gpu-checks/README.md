# gpu-checks: three GPU checks of the papers' claims

Three checks for the reproducibility paper (brief `.chains/gpu-checks/2026-10-17-050000000-brief-gpu-checks.md`), run
on one A100-SXM4-80GB (driver 570.172.08, torch 2.10.0+cu128, triton 3.6.0, CUDA 12.8) at `staging` `61045809` on
branch `dev/gpu-checks`, library `code_version` `fed664d6`, clean tree. They are session scripts, not harness suites,
and have no gate: **nothing here is citable** ([validation](../../validation.md#gpu-checks-not-citable)). Raw records
and logs are on the Hub under `artifacts/gpu-checks/<check>/` ([hub-index](../hub-index.md)).

| file | what |
|---|---|
| [`common.py`](common.py) | dataset loading through the harness (`bench.inputs.load_inputs`), the draw-0 synthetic flags (`eval-data synth-filter`'s rule and seed), the JSONL writer |
| [`check1_native.py`](check1_native.py) | check 1: LiNR's native-PyTorch baseline against our Triton V1 / V2 |
| [`check2_v3bits.py`](check2_v3bits.py) | check 2: LiNR V3 at 512 bits against 128 |
| [`check3_bloom_not.py`](check3_bloom_not.py) | check 3: Meta's bloom with NOT on the four kept reverse-clause filters |
| [`summarize.py`](summarize.py) | the tables below from the records |
| [`upload.py`](upload.py) | one commit per check to `pinkmeme/eval-results` with a `MANIFEST.json` (sha256 per file), read back and checked; the records are not harness-schema, so `bench upload` does not apply |

Run from `evaluation/` (`python ../research/artifacts/gpu-checks/check1_native.py`), with `evaluation/data` pointing at the
staged datasets.

**Datasets.** Goodreads 0.8 M d128 (SASRec `sasrec-ssm-logq-d128`), arXiv 3 M d128 and PubMed 10 M d768, from the Hub.
LAION is not on the Hub (it waits for the user's release decision), and rebuilding 10 M from Re-LAION means a download plus
about 20 min of encoding, so PubMed 10 M is the large scale point in every check. Each check uses the configs' first 10,000
queries (`users_limit`).

**Timing protocol.** The harness's: the fixed-seed query pool (`inputs.query_pool`, seed 0), the query side prepared
outside the timed call (`run.prepare_pool`), graph = `measure.graph_callable` (one `cudaGraphLaunch`, zero cudagraph skips),
and `measure.latency_group`'s three windows with the sampled `sm_mhz` and the `unstable` flag (spread > 5 %). The arms of a
variant are timed round-robin in one group.

## Check 1: LiNR's native PyTorch, as LiNR §6 states it

LiNR §6: "Native boolean masking and indexing cause a 100X latency increase". Our own native reference is sync-free, so it
does not test the claim. The script builds the literal forms in eager PyTorch:

- `native_index`: the clause mask by broadcast compare (LiNR §3.1 semantics, `-1` inactive), then per query (LiNR §5.3
  splits a batch per query) `mask[b].nonzero()` (host sync, data-dependent shape), `items[idx]` (a copy), an fp16 matmul on
  the gathered rows, `topk`, and the ids mapped back. A batch therefore costs B host syncs.
- `native_mask`: the same mask, a dense fp16 matmul, failing scores set to `-inf`, then `topk`. This is V1's similarity
  masking in native ops, and it is sync-free.

These are timed against `LiNRV1` / `LiNRV2` (triton backend) in graph replay, as in the final run, and in eager. Each cell
is one synthetic clause (draw 0, `u_i < p`) at p ∈ {0.001, 0.01, 0.1, 0.11, 0.5}, where 0.11 is LiNR's high-pass-rate set
(1.7 M of 15.5 M), with bs 1 and 16 and k 100. All arms read the same fp16 table, the LiNR modules' own buffer.
**Deviation:** the native arms use 3 warm-up calls and windows of at least 10 calls (`target_s` 1 s; the harness's minimum
is 1,000), because a native call reaches 0.4 s.

Each native arm returns the same top-100 as V2: overlap 0.988–1.000 on 4 probe batches. The rest is the native arms' fp16
score rounding; V1 and V2 agree at 1.000.

Medians in ms. "/ best graph" = the native median over the faster of V1 and V2 in graph replay. "unstable" names the arms
whose window spread exceeded 5 % (3 of 30 cells). Most windows ran at 1380–1410 MHz. Some dropped to 1140–1215 MHz:
goodreads 0.11 bs 16 (V1 and V2 eager), pubmed 0.001 bs 1 (V2 graph and eager) and pubmed 0.001 bs 16 (V1 eager, native
index). The ratios in those cells carry that throttle.

| dataset | pass | bs | V1 graph | V2 graph | V1 eager | V2 eager | native mask | native index | mask / best graph | index / best graph | index / best eager | overlap | unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | 0.001 | 1 | 0.294 | 0.171 | 0.385 | 0.547 | 0.516 | 0.289 | 3.0 | 1.7 | 0.8 | 0.9975 | no |
| goodreads | 0.001 | 16 | 0.467 | 0.490 | 0.552 | 0.549 | 0.995 | 2.382 | 2.1 | 5.1 | 4.3 | 0.9997 | no |
| goodreads | 0.01 | 1 | 0.293 | 0.172 | 0.386 | 0.549 | 0.515 | 0.297 | 3.0 | 1.7 | 0.8 | 1.0000 | no |
| goodreads | 0.01 | 16 | 0.468 | 0.502 | 0.555 | 0.561 | 0.995 | 2.757 | 2.1 | 5.9 | 5.0 | 0.9997 | no |
| goodreads | 0.1 | 1 | 0.291 | 0.182 | 0.388 | 0.556 | 0.514 | 0.395 | 2.8 | 2.2 | 1.0 | 1.0000 | no |
| goodreads | 0.1 | 16 | 0.470 | 0.587 | 0.555 | 0.645 | 0.995 | 4.086 | 2.1 | 8.7 | 7.4 | 0.9988 | no |
| goodreads | 0.11 | 1 | 0.291 | 0.185 | 0.390 | 0.558 | 0.517 | 0.399 | 2.8 | 2.2 | 1.0 | 0.9975 | no |
| goodreads | 0.11 | 16 | 0.471 | 0.595 | 0.581 | 0.771 | 1.000 | 4.432 | 2.1 | 9.4 | 7.6 | 0.9986 | triton_v1/eager, triton_v2/eager |
| goodreads | 0.5 | 1 | 0.291 | 0.239 | 0.385 | 0.551 | 0.515 | 0.690 | 2.2 | 2.9 | 1.8 | 1.0000 | no |
| goodreads | 0.5 | 16 | 0.471 | 0.938 | 0.555 | 0.997 | 0.998 | 10.544 | 2.1 | 22.4 | 19.0 | 0.9995 | no |
| arxiv | 0.001 | 1 | 0.735 | 0.227 | 0.765 | 0.544 | 2.169 | 0.295 | 9.5 | 1.3 | 0.5 | 0.9950 | no |
| arxiv | 0.001 | 16 | 1.452 | 1.554 | 1.544 | 1.615 | 3.380 | 3.088 | 2.3 | 2.1 | 2.0 | 0.9964 | no |
| arxiv | 0.01 | 1 | 0.733 | 0.235 | 0.763 | 0.549 | 2.167 | 0.392 | 9.2 | 1.7 | 0.7 | 0.9950 | no |
| arxiv | 0.01 | 16 | 1.453 | 1.591 | 1.549 | 1.649 | 3.379 | 4.483 | 2.3 | 3.1 | 2.9 | 0.9934 | no |
| arxiv | 0.1 | 1 | 0.735 | 0.280 | 0.766 | 0.552 | 2.170 | 0.602 | 7.7 | 2.1 | 1.1 | 0.9925 | no |
| arxiv | 0.1 | 16 | 1.453 | 1.857 | 1.547 | 1.920 | 3.385 | 9.217 | 2.3 | 6.3 | 6.0 | 0.9920 | no |
| arxiv | 0.11 | 1 | 0.735 | 0.287 | 0.765 | 0.554 | 2.167 | 0.637 | 7.6 | 2.2 | 1.2 | 0.9950 | no |
| arxiv | 0.11 | 16 | 1.454 | 1.888 | 1.548 | 1.948 | 3.384 | 9.818 | 2.3 | 6.8 | 6.3 | 0.9931 | no |
| arxiv | 0.5 | 1 | 0.735 | 0.480 | 0.768 | 0.559 | 2.170 | 1.917 | 4.5 | 4.0 | 3.4 | 0.9925 | no |
| arxiv | 0.5 | 16 | 1.453 | 3.305 | 1.547 | 3.376 | 3.384 | 30.187 | 2.3 | 20.8 | 19.5 | 0.9919 | no |
| pubmed | 0.001 | 1 | 9.240 | 0.567 | 9.321 | 0.737 | 9.558 | 0.445 | 16.9 | 0.8 | 0.6 | 0.9975 | triton_v2/graph, native_index/eager |
| pubmed | 0.001 | 16 | 11.584 | 4.666 | 11.719 | 4.723 | 18.762 | 7.809 | 4.0 | 1.7 | 1.7 | 0.9980 | native_index/eager |
| pubmed | 0.01 | 1 | 9.239 | 0.593 | 9.321 | 0.640 | 9.561 | 0.663 | 16.1 | 1.1 | 1.0 | 1.0000 | no |
| pubmed | 0.01 | 16 | 11.595 | 5.646 | 11.719 | 5.695 | 18.770 | 10.361 | 3.3 | 1.8 | 1.8 | 0.9956 | no |
| pubmed | 0.1 | 1 | 9.243 | 1.441 | 9.323 | 1.482 | 9.565 | 3.201 | 6.6 | 2.2 | 2.2 | 1.0000 | no |
| pubmed | 0.1 | 16 | 11.584 | 18.273 | 11.708 | 18.317 | 18.767 | 50.995 | 1.6 | 4.4 | 4.4 | 0.9953 | no |
| pubmed | 0.11 | 1 | 9.244 | 1.538 | 9.319 | 1.579 | 9.563 | 3.540 | 6.2 | 2.3 | 2.2 | 0.9950 | no |
| pubmed | 0.11 | 16 | 11.590 | 19.680 | 11.713 | 19.731 | 18.780 | 56.565 | 1.6 | 4.9 | 4.8 | 0.9962 | no |
| pubmed | 0.5 | 1 | 9.245 | 5.222 | 9.322 | 5.262 | 9.567 | 24.742 | 1.8 | 4.7 | 4.7 | 0.9875 | no |
| pubmed | 0.5 | 16 | 11.592 | 75.272 | 11.720 | 74.486 | 18.765 | 396.114 | 1.6 | 34.2 | 33.8 | 0.9916 | no |

**What it says.** Against our custom kernels in graph replay, the literal native form costs **0.8–34×**:
- **The 100× is not reproduced on any cell.**
- The index form (nonzero, gather, matmul) grows with pass rate and batch: 0.8–4.7× at bs 1 and 1.7–34× at bs 16. The top,
  34×, is PubMed 10 M d768 at 50 % and bs 16, where 16 host syncs each copy 5 M × 768 fp16 rows.
- The masked form is 1.8–17× at bs 1 and 1.6–4.0× at bs 16. Its worst case is a low pass rate at bs 1 (PubMed 16.9×), where
  V2 reads only the passing rows and the masked form scans the whole catalog.
- At LiNR's ~11 % point the index form costs 2.2–2.3× at bs 1 and 4.9–9.4× at bs 16 across the three datasets. The masked
  form costs 1.6–7.6×.
- At bs 1 and up to 11 %, the index form runs at 0.5–1.2× our *eager* V2 on Goodreads and arXiv. One host sync costs less
  than V2's fixed eager launches.

The paper can say the claim was tested as stated, in eager PyTorch 2.10 on an A100, and that we measure up to 34×, not 100×.
LiNR's setup (V100 / A100, TF and PyTorch versions of 2023, 15.5 M items, a serving stack) is not ours, so this neither
confirms nor refutes their number.

## Check 2: LiNR V3 at 512 bits

LiNR §5.3 / Fig. 7: V3 with **512-bit** codes, keeping **1 %** of the items for the full-precision stage, is "around 10 %"
faster than V1 / V2 at "nearly parity". The final run used 128-bit Sign-OPORP (k_bits = D). Sign-OPORP bins D coordinates
into k_bits groups, so `k_bits` must divide D (`quantize._oporp_k_bits`): 512-bit OPORP does not exist at D = 128, which is
LiNR's D too. The library's stage 1 for more bits than D is `SimHashKNN` (a Gaussian projection, every bit mixing all
coordinates). The arms are therefore:

- `oporp128`: the study's V3.
- `simhash128`: SimHash at 128 bits, which separates the projection from the bit count.
- `simhash512`: LiNR's bit count.
- `oporp768`: PubMed only, the study's V3 there (k_bits = D).

Each arm feeds the exact rescoring (`PrefilterKNN`). Pools: LiNR's 1 % of the catalog (`1%N`), and the harness's fractions
{0.005, 0.01, 0.02, 0.05, 0.1, 0.2} of the mean pass count with its 2,000 floor (`run.resolve_pool`).

**Quality.** Recall@100 = overlap with the exact filtered top-100 (LiNR V2) on the first 2,000 kept queries, measured on
every real clause sweep of the configs and on synthetic 11 / 20 / 50 %.

**Time.** Graph replay at bs 1 and 16, k 100, timed only where the time budget allowed: three real filters per dataset
(including the reverse and the conjunctive kept sweeps) and the three synthetic ones; `oporp128` and `simhash512` only; pools
`1%N`, `f0.01` and `f0.05`. V1 and V2 are timed in the same round-robin group, and the ratio is to the faster of them.
1 timed group is `unstable`. 2 of 864 windows dropped to 1140 MHz; the rest ran at 1395–1410 MHz.

Recall@100 against the exact filtered top-100 at LiNR's pool (1 % of N) and at the pool fraction 0.01 of the
mean pass count (floor 2000); then the smallest pool (items) reaching recall 0.95 per arm.

| cell | pass | oporp128 @1%N / @f0.01 | simhash128 @1%N / @f0.01 | simhash512 @1%N / @f0.01 | oporp768 @1%N / @f0.01 | oporp128 pool≥0.95 | simhash128 pool≥0.95 | simhash512 pool≥0.95 | oporp768 pool≥0.95 |
|---|---|---|---|---|---|---|---|---|---|
| goodreads/c0_genre | 0.3347 | 0.944 / 0.868 | 0.976 / 0.906 | 1.000 / 0.998 | — | 13,341 | 5,336 | 2,000 | — |
| goodreads/c1_lang_reverse | 0.6140 | 0.937 / 0.906 | 0.912 / 0.863 | 0.999 / 0.998 | — | 24,470 | 24,470 | 2,447 | — |
| goodreads/c2_format | 0.4031 | 0.943 / 0.887 | 0.973 / 0.919 | 1.000 / 0.999 | — | 16,066 | 6,426 | 2,000 | — |
| goodreads/c3_year | 0.3673 | 0.935 / 0.863 | 0.976 / 0.917 | 1.000 / 0.998 | — | 14,639 | 5,856 | 2,000 | — |
| goodreads/c0c1 | 0.1986 | 0.963 / 0.865 | 0.959 / 0.816 | 1.000 / 0.993 | — | 7,915 | 7,915 | 2,000 | — |
| goodreads/all4 | 0.0415 | 0.991 / 0.937 | 0.993 / 0.922 | 1.000 / 0.997 | — | 3,305 | 3,305 | 2,000 | — |
| goodreads/p0.11 | 0.1095 | 0.968 / 0.887 | 0.989 / 0.916 | 1.000 / 0.998 | — | 7,971 | 4,365 | 2,000 | — |
| goodreads/p0.2 | 0.1997 | 0.954 / 0.862 | 0.983 / 0.900 | 1.000 / 0.998 | — | 7,958 | 7,958 | 2,000 | — |
| goodreads/p0.5 | 0.5001 | 0.939 / 0.897 | 0.970 / 0.933 | 1.000 / 0.999 | — | 19,930 | 7,971 | 2,000 | — |
| arxiv/c0_maincat | 0.1196 | 0.919 / 0.628 | 0.863 / 0.521 | 0.997 / 0.924 | — | 71,467 | — | 7,147 | — |
| arxiv/c2_year | 0.1611 | 0.911 / 0.655 | 0.840 / 0.532 | 0.997 / 0.941 | — | 48,162 | 96,325 | 9,632 | — |
| arxiv/c3_nversions | 0.4475 | 0.859 / 0.755 | 0.768 / 0.640 | 0.994 / 0.977 | — | 133,746 | 267,492 | 13,375 | — |
| arxiv/c0c2 | 0.0203 | 0.989 / 0.685 | 0.979 / 0.600 | 1.000 / 0.930 | — | 29,890 | 29,890 | 3,041 | — |
| arxiv/all4 | 0.0075 | 0.998 / 0.817 | 0.997 / 0.761 | 1.000 / 0.967 | — | 29,890 | 29,890 | 2,000 | — |
| arxiv/p0.11 | 0.1099 | 0.933 / 0.608 | 0.877 / 0.489 | 0.998 / 0.921 | — | 65,711 | 65,711 | 6,571 | — |
| arxiv/p0.2 | 0.1999 | 0.896 / 0.667 | 0.821 / 0.543 | 0.996 / 0.950 | — | 59,752 | 119,503 | 11,950 | — |
| arxiv/p0.5 | 0.4997 | 0.839 / 0.744 | 0.738 / 0.623 | 0.992 / 0.976 | — | 149,374 | 298,747 | 14,937 | — |
| pubmed/c0_mesh | 0.0002 | 1.000 / 0.982 | 1.000 / 0.973 | 1.000 / 0.994 | 1.000 / 0.999 | 2,000 | 2,000 | 2,000 | 2,000 |
| pubmed/c2_year | 0.1267 | 0.884 / 0.585 | 0.788 / 0.446 | 0.980 / 0.859 | 1.000 / 0.990 | 253,442 | — | 63,361 | 6,336 |
| pubmed/c3_journal_reverse | 0.9994 | 0.680 / 0.680 | 0.535 / 0.535 | 0.908 / 0.908 | 0.998 / 0.998 | 1,998,754 | — | 499,689 | 49,969 |
| pubmed/c0c2 | 0.0001 | 1.000 / 0.999 | 1.000 / 0.999 | 1.000 / 1.000 | 1.000 / 1.000 | 2,000 | 2,000 | 2,000 | 2,000 |
| pubmed/all5 | 0.0001 | 1.000 / 0.999 | 1.000 / 0.999 | 1.000 / 1.000 | 1.000 / 1.000 | 2,000 | 2,000 | 2,000 | 2,000 |
| pubmed/p0.11 | 0.1098 | 0.889 / 0.555 | 0.798 / 0.420 | 0.979 / 0.817 | 1.000 / 0.986 | 219,635 | — | 54,909 | 5,491 |
| pubmed/p0.2 | 0.1999 | 0.834 / 0.593 | 0.720 / 0.453 | 0.963 / 0.847 | 1.000 / 0.992 | 399,834 | — | 99,958 | 9,996 |
| pubmed/p0.5 | 0.4998 | 0.750 / 0.649 | 0.611 / 0.502 | 0.935 / 0.887 | 0.999 / 0.996 | 999,565 | — | 249,891 | 24,989 |

Timing, graph replay: V1 / V2 medians (ms); V3 as a ratio to the faster of V1 and V2 timed in the same group.

| cell | bs | V1 | V2 | oporp128 1%N | oporp128 f0.01 | oporp128 f0.05 | simhash512 1%N | simhash512 f0.01 | simhash512 f0.05 | unstable |
|---|---|---|---|---|---|---|---|---|---|---|
| goodreads/c0_genre | 1 | 0.316 | 0.244 | 1.46 | 1.15 | 1.51 | 1.51 | 1.21 | 1.58 | no |
| goodreads/c0_genre | 16 | 0.567 | 1.293 | 1.95 | 1.73 | 2.06 | 2.41 | 2.20 | 2.52 | no |
| goodreads/c1_lang_reverse | 1 | 0.315 | 0.281 | 1.27 | 1.19 | 1.28 | 1.37 | 1.29 | 1.38 | no |
| goodreads/c1_lang_reverse | 16 | 0.568 | 1.596 | 2.04 | 1.97 | 2.23 | 2.79 | 2.73 | 2.98 | no |
| goodreads/all4 | 1 | 0.376 | 0.303 | 1.44 | 1.17 | 1.17 | 1.48 | 1.21 | 1.21 | no |
| goodreads/all4 | 16 | 0.647 | 1.922 | 3.13 | 2.93 | 2.93 | 3.19 | 2.99 | 2.99 | no |
| goodreads/p0.11 | 1 | 0.292 | 0.186 | 1.64 | 1.22 | 1.51 | 1.70 | 1.28 | 1.58 | no |
| goodreads/p0.11 | 16 | 0.469 | 0.595 | 1.55 | 1.27 | 1.47 | 1.69 | 1.42 | 1.61 | no |
| goodreads/p0.2 | 1 | 0.290 | 0.203 | 1.53 | 1.13 | 1.51 | 1.62 | 1.22 | 1.61 | no |
| goodreads/p0.2 | 16 | 0.470 | 0.668 | 1.58 | 1.30 | 1.59 | 1.88 | 1.61 | 1.89 | no |
| goodreads/p0.5 | 1 | 0.290 | 0.238 | 1.33 | 1.04 | 1.55 | 1.47 | 1.19 | 1.68 | no |
| goodreads/p0.5 | 16 | 0.470 | 0.940 | 1.71 | 1.47 | 2.01 | 2.50 | 2.27 | 2.79 | no |
| arxiv/c0_maincat | 1 | 0.830 | 0.392 | 1.27 | 1.09 | 1.38 | 1.32 | 1.14 | 1.44 | no |
| arxiv/c0_maincat | 16 | 1.783 | 3.591 | 1.85 | 1.70 | 1.84 | 2.05 | 1.90 | 2.04 | no |
| arxiv/c2_year | 1 | 0.831 | 0.453 | 1.09 | 1.04 | 1.08 | 1.20 | 1.14 | 1.19 | no |
| arxiv/c2_year | 16 | 1.792 | 3.964 | 1.86 | 1.75 | 1.82 | 2.22 | 2.11 | 2.18 | no |
| arxiv/all4 | 1 | 1.035 | 0.627 | 1.23 | 1.09 | 1.09 | 1.25 | 1.11 | 1.11 | no |
| arxiv/all4 | 16 | 2.087 | 6.689 | 3.34 | 3.22 | 3.22 | 3.34 | 3.22 | 3.22 | no |
| arxiv/p0.11 | 1 | 0.737 | 0.287 | 1.36 | 1.11 | 1.49 | 1.45 | 1.19 | 1.56 | no |
| arxiv/p0.11 | 16 | 1.455 | 1.890 | 1.38 | 1.21 | 1.36 | 1.58 | 1.40 | 1.55 | no |
| arxiv/p0.2 | 1 | 0.736 | 0.328 | 1.22 | 1.17 | 1.21 | 1.33 | 1.28 | 1.32 | no |
| arxiv/p0.2 | 16 | 1.454 | 2.186 | 1.43 | 1.31 | 1.42 | 1.79 | 1.66 | 1.77 | no |
| arxiv/p0.5 | 1 | 0.737 | 0.482 | 0.86 | 0.92 | 0.91 | 1.03 | 1.09 | 1.08 | no |
| arxiv/p0.5 | 16 | 1.453 | 3.306 | 1.58 | 1.54 | 1.77 | 2.52 | 2.47 | 2.71 | no |
| pubmed/c0_mesh | 1 | 9.578 | 0.793 | 1.22 | 1.08 | 1.08 | 1.23 | 1.09 | 1.09 | no |
| pubmed/c0_mesh | 16 | 11.959 | 6.363 | 1.07 | 1.00 | 1.00 | 1.07 | 1.01 | 1.01 | no |
| pubmed/c2_year | 1 | 9.579 | 2.060 | 0.53 | 0.50 | 0.51 | 0.57 | 0.53 | 0.55 | v2 |
| pubmed/c2_year | 16 | 11.971 | 28.582 | 0.74 | 0.60 | 0.68 | 0.84 | 0.70 | 0.78 | no |
| pubmed/all5 | 1 | 10.209 | 2.191 | 1.08 | 1.03 | 1.03 | 1.08 | 1.03 | 1.03 | no |
| pubmed/all5 | 16 | 13.307 | 16.112 | 1.03 | 0.98 | 0.98 | 1.05 | 1.00 | 1.00 | no |
| pubmed/p0.11 | 1 | 9.226 | 1.538 | 0.52 | 0.46 | 0.49 | 0.55 | 0.49 | 0.52 | no |
| pubmed/p0.11 | 16 | 11.589 | 19.692 | 0.59 | 0.45 | 0.52 | 0.67 | 0.53 | 0.60 | no |
| pubmed/p0.2 | 1 | 9.226 | 2.380 | 0.35 | 0.33 | 0.35 | 0.39 | 0.37 | 0.39 | no |
| pubmed/p0.2 | 16 | 11.587 | 32.516 | 0.61 | 0.49 | 0.61 | 0.75 | 0.63 | 0.75 | no |
| pubmed/p0.5 | 1 | 9.225 | 5.219 | 0.17 | 0.16 | 0.21 | 0.22 | 0.21 | 0.26 | no |
| pubmed/p0.5 | 16 | 11.588 | 75.314 | 0.70 | 0.61 | 0.94 | 1.05 | 0.96 | 1.29 | no |

**What it says.**

- **Bits matter more than the projection.** At LiNR's pool (1 % of N), `simhash512` recovers 0.992–1.000 of the exact
  top-100 on Goodreads and arXiv, and 0.91–0.98 on PubMed's dense filters (≥ 11 % pass). At the same pool, `oporp128`
  recovers 0.84–1.00 and 0.68–0.89 there. To reach 0.95, `simhash512` needs a pool 4–15× smaller than `oporp128` (1.7× on
  Goodreads `all4`, and equal on PubMed's sparse filters, where the 2,000 floor already holds every passing item). SimHash
  at 128 bits is no better than OPORP at 128: it is better on most Goodreads filters and worse on every arXiv and PubMed
  filter.
- **"Nearly parity" at 1 % holds with 512 bits.** It holds where 512 ≥ D (Goodreads and arXiv: ≥ 0.992). On PubMed
  (D = 768) the study's own `oporp768` does better than 512 bits (0.998–1.000 at 1 % of N).
- **"About 10 % faster" is scale-dependent.** At 0.8 M and 3 M items, V3 is slower than the faster exact arm on every timed
  cell (1.04–3.3×) except arXiv at 50 % and bs 1 (0.86–0.92× with `oporp128`). At 10 M (PubMed d768) with ≥ 11 % passing,
  V3 at 1 % of N takes 0.17–0.74× of the faster exact arm with `oporp128` and 0.22–1.05× with `simhash512`. On PubMed's selective
  filters (`c0_mesh` 0.02 %, `all5` 0.01 %) V3 and V2 are even (0.98–1.23×). That is LiNR's own remark: V2 wins at a low
  pass rate.
- **512 bits cost little time at one query.** At the same pool, `simhash512` takes 2–20 % more than `oporp128` at bs 1 on
  Goodreads and arXiv, and 0–31 % more on PubMed. At bs 16 it takes 0–60 % more.

LiNR measured on 15.5 M jobs at d128 with TF / PyTorch 2023 kernels, and its recall is a label recall@2000; ours is fidelity
to the exact top-100. The shape agrees (V3 pays off only at a large N and a high pass rate). The numbers are not comparable.

## Check 3: Meta's bloom with NOT

Four of the kept sweeps contain a reverse clause and were left out of the bloom head-to-head, because the wrapper refuses
NOT in bloom mode: Goodreads `c1_lang_reverse` (C1 language) and `all4` (C0-C3, NOT on C1); PubMed `c3_journal_reverse`
(C3 journal) and `all5` (C0-C4, NOT on C3). Meta's code supports NOT by complementing the bloom mask of `c:v`. A bloom
false positive on `c:v` therefore becomes a false negative of `NOT c:v`.

**Setup.**
- The bloom index is built forward-only, as on every bloom cell, at the study's configuration: `m_bits` 1,024 (configured
  width), `k_hash` 5, no recheck. The query side passes the reverse flags to the adapter's
  `queries_to_expressions(..., clause_is_reverse)`, which emits Meta's `NOT c:v`. Meta's own parser and
  `bloom_index_search_batch` do the rest.
- Per query, over the whole catalog: Meta's full bloom search, unpacked and mapped back from the cluster-sorted doc space,
  against the exact predicate (`ExactAttributeFilter`), for the NOT clause alone and for the whole filter.
- Recall@100 against the exact filtered top-100 (LiNR V2), for SilverTorch with the bloom (NOT included) and with its exact
  filter. Both use one k-means: same seed, `sort_perm` checked equal. The study's `n_lists` (4,096 on both datasets) and
  `n_probe` 24 plus the `filter` suite's other probes. 2,000 kept queries per filter.
- Staging's k-means assigns by inner product, while the final run's records were L2-assigned, so these recalls compare only
  within this index.

| filter | pass rate | NOT clause: FN / passing items | queries with ≥ 1 NOT FN | whole filter: FN / passing items | whole filter: FP / failing items | recall@100, bloom with NOT | recall@100, exact filter |
|---|---|---|---|---|---|---|---|
| goodreads `c1_lang_reverse` | 0.614 | 0 | 0 % | 0 | 0 | 0.7188 (np 24), 0.8715 (np 64) | identical |
| goodreads `all4` | 0.041 | 0 | 0 % | 0 | 0 | 0.4477, 0.6306 | identical |
| pubmed `c3_journal_reverse` | 0.9994 | 2.4e-8 (mean 0.24 items a query, max 40) | 10.8 % | 2.4e-8 | 0 | 0.7918 (np 24), 0.9268 (512), 0.9308 (1,024) | identical |
| pubmed `all5` | 0.0001 | 2.2e-8 (mean 0.22, max 40) | 9.5 % | 0 | 4.5e-10 (9 items over 2,000 queries) | 0.4002, 0.8640, 0.9290 | identical |

"Identical" means equal to four decimals at every probe. The low recall at `n_probe` 24 is the IVF probe, not the filter.

**Forward-only sanity run** (`forward_sanity.jsonl`, same code, Goodreads `c0_genre` / `c2_format` / `c3_year`, PubMed
`c0_mesh` / `c2_year`): 0 false negatives everywhere, so the bloom covers the exact set. False positives appear only on
PubMed MeSH: 4.4e-8 of the failing items, in 11.6 % of queries, max 114 items. Goodreads' clauses (a handful of genre,
format and year values over about 7 terms a document) give none at 1,024 bits per document, so its zeros above are the
bloom's, not a dead code path.

**What it says.** At the study's bloom width, NOT through Meta's complement is practically lossless on these filters:
- At most 2.4e-8 of the passing items are lost, which is 0.24 of about 10 M a query.
- On `all5` the conjunction absorbs the NOT clause's false negatives: the lost items fail the other clauses anyway.
- SilverTorch's recall@100 with the bloom NOT equals its exact-filter recall at every probe.

So the exclusion from the head-to-head cost no quality at this width. The four filters could run in the bloom head-to-head
with NOT, which was not timed here. The loss depends on the width: a narrower bloom (Meta's upstream width, or the
`bloomwidth` suite's smaller `m_bits`) has more false positives, and therefore more NOT false negatives. That was not
measured.
