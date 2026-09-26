---
chain: "silvertorch-official-integration"
branch: "main"
parent: "2026-09-06-200000000-wp5-record-cuda-cute-deletion.md"
nextStep: "WP-9 (roadmap F2): write the paper's official-vs-reimplementation section from these numbers; add TF-9 (CSR / capped-pad probe layout) to §8 for Phase G."
created: "2026-09-15T12:00:00Z"
---

# §16 record: WP-4, the Triton vs official head-to-head (roadmap B3), 2026-09-15

Branch `dev/b3-head-to-head` off `development` @ `e23309c`, merged as `399c231`; closes paper gap G2. Scripts, raw JSON, harness records and the full per-cell table dump: `docs/artifacts/official-silvertorch/b3/` (`b3_e2e_run.sh`, `b3_kernel_h2h.py`, `b3_tables.py`, `kernel_{goodreads,arxiv}.json`, `e2e/**/*.jsonl`, `tables.md`). Not citable until the gate is re-run (rule 2).

## 16.1 Environment and comparability
A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, CUDA runtime 12.8 / nvcc 12.4 build of `silvertorch @ 21aa35e`, Python 3.11, `/venvs/b3`, `code_version 0e67780...`, `dirty: false` on all 30 records. goodreads-work-id d128 (797,084 items, sweep `c0_genre`) and arxiv-papers d128 (2,988,996, `c0_maincat`), `users_limit 10000`, seed 0, `n_lists 1024`, `n_probe` in {24, 32} end to end and 24 kernel-only, bloom `m_bits 1024, k_hash 5`, official `b_multiplier 10.0, hash_k 7, bloom_path="partial"`.
- Estimator: `bench.measure.latency` (50 warm-ups, 3 windows of `clamp(2 s / median, 1000, 5000)` calls, per-call CUDA events, median of window medians, `unstable` > 5 % spread). Host-side rows use `perf_counter`.
- Clocks: unlockable; `sm_mhz` sampled under load after the last window; 1410 MHz under load; rows at 1155-1395 MHz are arms whose GPU idles inside the call (official at bs=1: `.tolist()` sync + CPU parse), host-bound by construction.
- bs=1 noise: 31 of 468 end-to-end perf entries `unstable`, 26 at bs in {1, 8}, spreads to 22.5 %; headline comparisons at bs=16 (5 of 156 over 5 %; worst 19.1 % arxiv clause triton k=500; worst k=100 is 8.1 %, arxiv bloom official n_probe 24).
- Plan cache: `OfficialConfig(cache_plans=False)` on every timed official forward; recorded on all 156 official perf entries.
- Fairness checked: shared `centroids`, official `item_codes` `torch.equal` to Triton `item_codes[sort_perm]`, equal `global_scale` (6/6).
- Kernel-only covers quantization, mask op(s), scorer, epilogue and shared `masked_topk`; phase 1 excluded. End to end is the whole `module.forward` through the harness.

## 16.2 Kernel-only, phases 2+3, bs=16, n_probe 24
Device µs from one profiled call classified by kernel name (a heuristic; raw lists in JSON).

| dataset | mode | arm | wall median ms | spread | sm_mhz | device µs | scorer | mask | prep | topk | launches | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | none | triton | 0.7517 | 0.000 | 1410 | 721 | 337.0 | - | 41 | 335 | 38 | 37.7 |
| goodreads | none | official-fp16 | 1.3074 | 0.006 | 1410 | 1146 | 30.9 | - | 710 | 338 | 75 | 214.6 |
| goodreads | none | official-int32 | 1.3750 | 0.003 | 1410 | 1203 | 33.1 | - | 771 | 335 | 73 | 233.3 |
| goodreads | bloom | triton | 1.0069 | 0.000 | 1410 | 960 | 520.8 | - | 88 | 339 | 58 | 37.7 |
| goodreads | bloom | official-fp16 | 2.1563 | 0.070 | 1410 | 1181 | 29.2 | 9.1 | 730 | 331 | 93 | 214.7 |
| goodreads | exact | triton | 0.8177 | 0.000 | 1410 | 778 | 397.6 | - | 37 | 339 | 35 | 37.7 |
| goodreads | exact | official-fp16 | 3.0496 | 0.001 | 1410 | 1343 | 26.8 | - | 834 | 324 | 79 | 228.3 |
| arxiv | none | triton | 0.4977 | 0.005 | 1410 | 302 | 129.8 | - | 40 | 124 | 38 | 10.8 |
| arxiv | none | official-fp16 | 0.9362 | 0.006 | 1410 | 529 | 113.3 | - | 268 | 120 | 75 | 60.9 |
| arxiv | bloom | triton | 0.8953 | 0.009 | 1410 | 452 | 228.6 | - | 86 | 124 | 58 | 10.8 |
| arxiv | bloom | official-fp16 | 1.6064 | 0.001 | 1395 | 543 | 76.8 | 10.4 | 294 | 120 | 93 | 61.0 |
| arxiv | exact | triton | 0.4770 | 0.003 | 1410 | 423 | 260.8 | - | 33 | 125 | 34 | 10.8 |
| arxiv | exact | official-fp16 | 7.2117 | 0.001 | 1410 | 815 | 81.7 | - | 271 | 119 | 77 | 826.7 |

bs=1 (JSON): Triton 0.380-0.869 ms vs official 0.833-1.467 ms, 1.4-2.5x for host reasons.

1. Wall bs=16: Triton 1.7-2.1x faster unfiltered and bloom, 3.7x (goodreads) / 15.1x (arxiv) in exact mode (see 4).
2. Meta's scorer kernel is faster than ours in every cell: 1.15-3.2x on arxiv (113 vs 130 µs unfiltered, 77 vs 229 bloom), 10.9-17.8x on goodreads (31 vs 337, 29 vs 521). Cause: our padded IVF layout. `n_probe x max_cluster_size` = 611,520 slots on goodreads for ~18.7k real items (97 % `-1` pads) vs 171,648 slots and ~59 % pads on arxiv. The official op reads a CSR. Equal `n_probe` is equal recall, not equal work.
3. Official gives it back in payload prep: 268-896 µs across 73-95 launches vs Triton's 34-58; total device time 1.2-2.4x ours.
4. The arxiv exact 7.21 ms is ours: device time 0.50-0.82 ms; the rest is our adapter's full-N `clause_mask` + `pack_mask` (`[B, N/64, 64]` intermediate = the 826 MiB peak; a 316 µs `reduce_kernel<long>` filed under "quantize"). Reported, not fixed.
5. The `topk` epilogue is identical in both arms (324-339 µs goodreads, 119-125 arxiv) and dominates the unfiltered Triton cell.

## 16.3 Phase 2 alone, bs=16, full N
| dataset | op | arm | median ms | p99 ms | spread | timer |
|---|---|---|---|---|---|---|
| goodreads | `bloom_match` (row-wise) | ours | 0.4517 | 0.5610 | 0.060 | CUDA events |
| goodreads | `bloom_index_search_batch` (transposed, packed) | official | 0.2279 | 0.2588 | 0.003 | CUDA events |
| goodreads | `..._return_partial_response` (probed only) | official | 0.4488 | 0.4971 | 0.029 | CUDA events |
| goodreads | `build_query_signatures` | ours | 0.3687 | 0.4191 | 0.004 | CUDA events |
| goodreads | `queries_to_expressions` (host, incl. D2H) | official | 0.0272 | 0.0376 | - | perf_counter |
| goodreads | `parse_expression_query_batch` (host) | official | 0.0468 | 0.0594 | - | perf_counter |
| arxiv | `bloom_match` | ours | 1.4809 | 1.4898 | 0.000 | CUDA events |
| arxiv | `bloom_index_search_batch` | official | 0.2443 | 0.2814 | 0.003 | CUDA events |
| arxiv | `..._return_partial_response` | official | 0.4513 | 0.5039 | 0.016 | CUDA events |
| arxiv | `build_query_signatures` | ours | 0.3780 | 0.4580 | 0.015 | CUDA events |
| arxiv | `queries_to_expressions` | official | 0.0273 | 0.0363 | - | perf_counter |
| arxiv | `parse_expression_query_batch` | official | 0.0468 | 0.0559 | - | perf_counter |

- S13 replicated with Meta's code: full-N mask official 2.0x faster on goodreads (0.8 M), 6.1x on arxiv (3.0 M); ours grows 3.3x with N, theirs 1.07x. Strongest case for TF-1.
- The official partial response is slower than its own full-N search here (13 launches, 2 syncs, a `repeat_interleave` vs one launch); the co-design wins end to end by shrinking what the scorer reads.
- Our query-side bloom hashing costs 0.37 ms at bs=16, more than the whole official mask search, and is most of the gap between Triton `none` (0.75 ms) and `bloom` (1.01 ms). Largely disappears under CUDA graph. Recorded, not acted on.
- Parse: 46.8 µs/call at bs=16 (2.9 µs/query) + 27.2 µs `queries_to_expressions` = ~74 µs host work per official bloom forward.

| dataset | exact pass rate | our bloom FP rate | official FP rate | our bloom MiB | official index MiB |
|---|---|---|---|---|---|
| goodreads `c0_genre` | 0.3323 | 0.000000 | 0.000000 | 97.3 (`m_bits 1024`) | 33.3 (`b_multiplier 10`) |
| arxiv `c0_maincat` | 0.1357 | 0.000000 | 0.000000 | 364.9 | 71.3 |
Both blooms zero FP (8 batches x 16 queries), so matched-FPR bisection is undefined; at equal (zero) FPR the official index is 2.9x / 5.1x smaller; our `m_bits 1024` is over-provisioned for single-clause sweeps. S8 needs D3's wider sweep.

## 16.4 Parity, and the goodreads / arxiv jaccard split
Kernel-only, 512 queries per cell vs the Triton arm:
| dataset | mode | score path | jaccard@100 | score_max_abs_diff |
|---|---|---|---|---|
| goodreads | none | int32 | 1.000000 | 0.0 |
| goodreads | none | fp16 | 1.000000 | 2.885e-03 |
| goodreads | bloom / exact | int32 | 0.999961 | 0.0 |
| goodreads | bloom / exact | fp16 | 0.999961 | 2.916e-03 |
| arxiv | none | int32 | 1.000000 | 0.0 |
| arxiv | none | fp16 | 0.982928 | 4.814e-04 |
| arxiv | bloom / exact | int32 | 1.000000 | 0.0 |
| arxiv | bloom / exact | fp16 | 0.985788 | 4.814e-04 |
int32 bit-exact against Triton on both datasets in all three modes (goodreads 0.999961 with diff 0.0 = ties). The whole arxiv deficit is the fp16 path; C4's hypothesis ("near-ties under a looser filter") is falsified: unfiltered cells split the same way (arxiv 0.9838 e2e / 0.9829 kernel-only vs goodreads 0.9999 / 1.0000).

| dataset | mode | rank-100/101 gap p10 | gap median | score@100 median | one fp16 ulp | rows with gap < 1 ulp |
|---|---|---|---|---|---|---|
| goodreads | none | 9.46e-04 | 6.97e-03 | 0.5623 | 2.75e-04 | 3.3 % |
| goodreads | bloom / exact | 1.10e-03 | 6.85e-03 | 0.6320 | 3.09e-04 | 3.1 % |
| arxiv | none | 1.38e-05 | 7.92e-05 | 0.8363 | 4.08e-04 | 95.3 % |
| arxiv | bloom / exact | 1.41e-05 | 8.57e-05 | 0.8303 | 4.05e-04 | 94.5 % |
arxiv's nomic embeddings put ranks 100/101 8.6e-5 apart on ~0.83 (a fifth of an ulp); goodreads' gaps are 22x their ulp. Cost: recall@100 vs the exact oracle, 10,000 queries: arxiv `c0_maincat` n_probe 24 official 0.883735 vs Triton 0.884044 (Δ 3.1e-4); goodreads `c0_genre` 0.912796 vs 0.912800 (Δ 4e-6). `torch` matches `triton` at jaccard 1.0, diff 0.0 on all six filter cells (post-L5).

## 16.5 End to end, k=100, seed 0, harness filter + quality suites
30 records (24 ok, 6 partial: quality narrowed to `--k 100 --bs 1 --bs 8 --bs 16`), one `code_version`, `env.sm_mhz_load` 1410 on all 30.

| dataset | filter | n_probe | backend | eager bs=1 | bs=8 | bs=16 | graph bs=16 | qps bs=16 | peak MiB bs=16 | index MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | bloom | 24 | triton | 1.083 | 1.075 | 1.094 | 0.428 | 14518 | 32 | 786 |
| arxiv | bloom | 24 | official | 1.176 | 1.370 | 1.292 | n/a | 12355 | 61 | 482 |
| arxiv | bloom | 24 | torch | 1.162 | 4.253 | 8.170 | 2.319 | 1957 | 1725 | 786 |
| arxiv | bloom | 32 | triton | 1.069 | 1.087 | 1.084 | 0.526 | 14751 | 42 | 786 |
| arxiv | bloom | 32 | official | 1.300 | 1.372 | 1.451 | n/a | 11102 | 81 | 482 |
| arxiv | bloom | 32 | torch | 1.240 | 5.565 | 10.806 | 3.017 | 1480 | 2299 | 786 |
| arxiv | clause | 24 | triton | 0.560 | 0.689 | 0.701 | 0.452 | 22750 | 32 | 786 |
| arxiv | clause | 24 | official | 1.167 | 3.956 | 7.110 | n/a | 2255 | 827 | 776 |
| arxiv | clause | 24 | torch | 0.735 | 4.076 | 7.863 | 2.276 | 2033 | 2060 | 786 |
| arxiv | clause | 32 | triton | 0.695 | 0.704 | 0.709 | 0.560 | 19981 | 42 | 786 |
| arxiv | clause | 32 | official | 1.150 | 3.924 | 7.131 | n/a | 2240 | 827 | 776 |
| arxiv | clause | 32 | torch | 0.902 | 5.334 | 10.410 | 2.990 | 1536 | 2745 | 786 |
| arxiv | none | - | triton | 0.541 | 0.659 | 0.668 | 0.348 | 23821 | 32 | 421 |
| arxiv | none | - | official | 0.834 | 0.834 | 0.846 | n/a | 18851 | 61 | 411 |
| arxiv | none | - | torch | 0.589 | 2.549 | 4.852 | 2.128 | 3294 | 1722 | 421 |
| goodreads | bloom | 24 | triton | 0.999 | 1.020 | 1.112 | 0.970 | 14305 | 112 | 394 |
| goodreads | bloom | 24 | official | 1.330 | 1.411 | 2.063 | n/a | 7501 | 215 | 143 |
| goodreads | bloom | 24 | torch | 2.088 | 14.325 | 28.516 | 7.192 | 561 | 6140 | 394 |
| goodreads | bloom | 32 | triton | 1.055 | 1.096 | 1.350 | 1.206 | 11797 | 150 | 394 |
| goodreads | bloom | 32 | official | 1.344 | 1.637 | 2.260 | n/a | 7088 | 287 | 143 |
| goodreads | bloom | 32 | torch | 2.668 | 19.065 | 37.870 | 9.346 | 422 | 8186 | 394 |
| goodreads | clause | 24 | triton | 0.519 | 0.649 | 0.921 | 0.842 | 17253 | 112 | 394 |
| goodreads | clause | 24 | official | 1.053 | 1.677 | 3.044 | n/a | 5240 | 228 | 207 |
| goodreads | clause | 24 | torch | 1.980 | 13.843 | 27.575 | 6.811 | 580 | 7335 | 394 |
| goodreads | clause | 32 | triton | 0.648 | 0.655 | 1.119 | 1.037 | 14224 | 150 | 394 |
| goodreads | clause | 32 | official | 1.051 | 1.837 | 3.291 | n/a | 4849 | 300 | 207 |
| goodreads | clause | 32 | torch | 2.544 | 18.407 | 36.638 | 9.022 | 437 | 9779 | 394 |
| goodreads | none | - | triton | 0.534 | 0.667 | 0.857 | 0.894 | 18539 | 112 | 297 |
| goodreads | none | - | official | 0.897 | 0.904 | 1.325 | n/a | 11995 | 215 | 110 |
| goodreads | none | - | torch | 1.274 | 8.490 | 16.918 | 6.639 | 945 | 6131 | 297 |

Quality and parity (recall@100 vs oracle; jaccard_vs_first@100; score_max_abs_diff; unstable):
- arxiv bloom 24: triton 0.884044 (ref, unstable), official 0.883735 / 0.984950 / 9.510e-02 (unstable), torch 0.884044 / 1.0 / 0.
- arxiv bloom 32: triton 0.903351 (ref, unstable), official 0.903010 / 0.984797 / 9.795e-02 (unstable), torch 0.903351 / 1.0 / 0 (unstable).
- arxiv clause 24: triton 0.884044 (ref, unstable), official 0.883757 / 0.985033 / 4.827e-04, torch 0.884044 / 1.0 / 0.
- arxiv clause 32: triton 0.903351 (ref, unstable), official 0.903036 / 0.984882 / 4.827e-04, torch 0.903351 / 1.0 / 0.
- arxiv none: official jaccard 0.983778 / 4.827e-04; torch 1.0 / 0 (unstable).
- goodreads bloom 24: triton 0.912800 (ref, unstable), official 0.912796 / 0.999849 / 5.517e-03 (unstable), torch 0.912800 / 1.0 / 0.
- goodreads bloom 32: triton 0.936908 (ref, unstable), official 0.936910 / 0.999805 / 5.622e-03 (unstable), torch 0.936908 / 1.0 / 0.
- goodreads clause 24: triton 0.912800, official 0.912796 / 0.999849 / 5.517e-03 (unstable), torch 1.0 / 0.
- goodreads clause 32: triton 0.936908, official 0.936910 / 0.999805 / 5.622e-03 (unstable), torch 1.0 / 0.
- goodreads none: official 0.999885 / 3.794e-03; torch 1.0 / 0.

Reading:
- Triton is the fastest arm in every cell, eager and graph. vs official at bs=16: 1.5x goodreads none, 1.9x goodreads bloom, 3.3x goodreads clause, 1.2x arxiv none, 1.2x arxiv bloom, 10.1x arxiv clause. At bs=1, 1.1-2.0x, inside bs=1 noise for bloom.
- The co-design shows in Meta's numbers and inverts the filter ordering: for official, bloom (partial masks) is cheaper than exact (full-N mask): 1.29 vs 7.11 ms arxiv, 2.06 vs 3.04 ms goodreads; for Triton the order is reversed (1.09 vs 0.70, 1.11 vs 0.92) because our exact predicate is fused and our bloom pays the query hash. S9 directionally reproduced; the controlled `bloom_path="full"` ablation not run.
- Graph is Triton-only: all 78 official graph entries `null` / `not_capturable`; Triton graph 0.35-1.21 ms at bs=16: 1.6-2.6x over its own eager on arxiv, 1.09-1.15x on goodreads filter cells, 0.96x (slower) on goodreads none (0.894 vs 0.857 ms).
- `torch` is the floor and bit-exact: jaccard 1.0, diff 0.0 on all six filter cells, 4.9-37.9 ms per bs=16 forward (7.3-32.7x Triton), 1.7-9.6 GiB peak vs Triton's 32-150 MiB.
- Memory: official index smaller where padding bites: 2.70x goodreads none (110 vs 297 MiB), 2.75x goodreads bloom, 1.90x goodreads clause, 1.63x arxiv bloom; only 1.01-1.02x on arxiv none / clause. Official forward peak ~1.9x ours on bloom and none, 26x on arxiv clause (827 vs 32 MiB, the packer).
- `bloom_fp_rate` 0.000000 on all four of our bloom cells.
- Stability: 14 of 30 cells `unstable`, each a bs in {1, 8} entry with spread 5-22 % or a `clocks_drift` flag from host-bound arms dropping to 1155-1395 MHz (10 cells, all bloom or arxiv clause). Of 156 bs=16 entries 5 exceed 5 %; read the arxiv bloom official n_probe 24 1.2x as 1.1-1.3x (spread 8.1 %).

## 16.6 Verdicts on §9's expectations
| expectation | verdict |
|---|---|
| (i) kernel-only unfiltered parity within +-20 % | wrong both ways: Meta's scorer 1.15x (arxiv) to 10.9x (goodreads) faster; our forward still 1.7x faster (their prep 268-896 µs); goodreads factor = pad tax |
| (ii) official 2-3x slower at bs=1 | confirmed, milder: 1.4-2.5x kernel-only, 1.1-2.0x e2e |
| (iii) official bloom kernel-only ~2x faster until TF-1 | wrong: their phase 2 beats ours 2.0-6.1x, but their bloom forward is 1.8x (arxiv) / 2.1x (goodreads) slower than our fused one |
| (iv) identical candidates up to fp16 ties | confirmed: int32 bit-exact; fp16 costs 3.1e-4 (arxiv) / 4e-6 (goodreads) recall@100 |
| S13 | replicated against Meta's code, grows with N |
| S9 | directionally reproduced (their partial beats their full-mask path 1.5-5.5x e2e); controlled ablation not run |
| S8 | not measurable (FPR 0.0 both) |

Corrections: TF-1's case strengthened; TF-3 / TF-4 second-order next to the padded layout; "TF-9: CSR or capped-pad probe layout" belongs in §8.

Skipped: `fpr_calibrate.py` (FPR 0), the `bloom_path="full"` S9 cells, S10, S6/S16, S12, the synthetic P ladder (replaced by the real datasets' P: 611,520 goodreads, 171,648 arxiv), ncu (blocked), seeds 1-2, k in {500, 1000} ratios.

Unverified / carried: the official exact path's 826 MiB / 7.2 ms is our `pack_mask`, not fixed (every official-exact number is an upper bound); the 0.37 ms query-bloom hash not attributed to a kernel; the kernel-class split is a name heuristic; parity measured at k=100 only.
