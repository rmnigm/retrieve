# L1 + L2 — fp32 scores for the exact LiNR scorers; no `-1` tail from compaction

Roadmap L1 and L2, bundled into one `code_version` bump. Raw outputs (JSON dumps, logs) are
on the Hub under `artifacts/l1-l2/` ([hub-index](../hub-index.md)).

## L1: where the precision goes, on YFCC-10M d192

[`yfcc_precision.py`](yfcc_precision.py): the `tags_and` sweep's first 10,000 kept queries,
exact clause mask, top-1000, scored four ways and compared with the harness oracle's method
(fp32 `q @ Eᵀ`, TF32 off) and with an fp64 top-1000.

| scoring | recall vs oracle | recall vs fp64 |
|---|---|---|
| fp16 table, fp16 scores (`PostfilterKNN` before L1) | 0.9563 | 0.9563 |
| fp16 table, fp32 scores (`torch.mm(..., out_dtype=torch.float32)`, **L1**) | **0.9930** | 0.9930 |
| fp16 table upcast to fp32 before the GEMM (no tensor-core accumulator) | 0.9930 | 0.9930 |
| fp32 table, fp32 scores | 1.0000 | 0.9999 |

The fp16 **score** costs 0.037 of the 0.044; the remaining 0.007 is the fp16 **storage**
rounding (the upcast row shows the tensor-core accumulator contributes nothing measurable).
The gate is `recall_oracle@1000 ≥ 0.99`.

## L1: the harness gate on YFCC-10M, before and after

`bench run --dataset yfcc10m --suite filter --algo linr_v1_filter_mask --filter-kind clause
--mode eager --skip-perf` (10,000 queries, `users_limit` of the committed config; a scratch
config dir pointing `data_dir` at `/data/yfcc10m`); "before" is the same command with the pre-L1
tree first on `PYTHONPATH`. V2 cannot run in the harness at d192 (the Triton kernel's
power-of-two `D`; the suite folds V2's torch backend into Triton), so
[`yfcc_v2_torch.py`](yfcc_v2_torch.py) scores `LiNRV2(backend="torch")` against the same oracle
blob.

| cell | before | after |
|---|---|---|
| `linr_v1_filter_mask` / triton, `recall_oracle@1000` | 0.9652, `QualityGateError` | **0.9944**, gate passes (`recall_oracle@100` 0.9886, n = 9,817) |
| `linr_v2` / torch (script), `recall_oracle@1000` | 0.9652 | **0.9944** (n = 9,817) |

## L1: memory, at real catalog shapes

[`index_mem.py`](index_mem.py), one process per side, random tensors of the real `N × D`
(items fp32 on the device, as the harness holds them); B=16, k=1000, masked forward.

| shape | `index_mib` before → after | fp32-table alternative | forward transient before → after (MiB) |
|---|---|---|---|
| arxiv 2,988,997 × 128 | 730 → 730 | 1,459 | 192 → 375 |
| yfcc10m 10M × 192 | 3,662 → 3,662 | 7,324 | 615 → 1,226 |
| pubmed 10M × 768 | 14,648 → 14,648 | 29,297 | 615 → 1,226 |

The table is unchanged; the `[B, N]` score buffer doubles. An fp32 table would add
`N·D·2` bytes (+14.3 GiB on pubmed, where the harness already holds a 28.6 GiB fp32 copy).

## L2: who reads a compaction's output, site by site

`clause_compact` / `bloom_compact` (Triton) now write `[:counts]` only; the reference twins'
tail is the argsort's. Every place in `retrieve/src/retrieve/` that takes their output:

| site | how it reads the `[B, N]` ids | verdict |
|---|---|---|
| `ExactAttributeFilter` / `BloomFilter.evaluate_indices`, `FilterModule.evaluate_indices` | producers (return the pair) | — |
| `LiNRV2.forward` → `PrefilterKNN(candidate_ids, counts)` | passes `counts`; pads `P < k` with `-1` lanes past `counts` | safe |
| ↳ Triton `fused_masked_knn_topk` | id loads masked by `n < counts[b]`; epilogue gathers *positions*, `-inf` slots → `-1` | safe |
| ↳ reference `fused_masked_knn_topk` | gathered `item_embs[ids.clamp_min(0)]` over the whole row | **fixed**: `where(n < counts, id, 0)`; faulted on a Triton compaction's tail (`test_linr.py::test_prefilter_with_evaluate_indices_kernel_path[torch]` under the poison) |
| `LiNRV3.forward` → `OneBitKNN(candidate_ids, counts)` | passes `counts` | safe |
| ↳ Triton `oporp_1bit_match_topk_indirect` | loads masked by `n < counts[b]`; same epilogue | safe |
| ↳ reference `oporp_1bit_match_topk_indirect` | gathered `item_bits[ids]` over the whole row | **fixed** the same way (no test fed it a Triton tail) |
| ↳ stage 2 `PrefilterKNN(cand, (cand >= 0).sum)` | reads stage 1's top-k output, not a compaction | not a consumer |
| `functional.combine_indices` | `where(valid, ids, 0)` before `evaluate_subset`; re-compacts by positions | safe; its own output tail is garbage-derived, bounded by its `counts` |
| `FullScanKNN` / `SilverTorch` `candidate_ids` paths | `-1`-padded ids, no `counts` | no library caller feeds them a compaction; docstrings now say to mask first |
| `ops/tune.py` | times `_clause_compact_impl` / `_bloom_compact_impl`, reads no output | — |
| `evaluation/` | no caller of a compaction op or `evaluate_indices` | — |

Tests that read past `counts` (found by filling the output with `2**40` before the scatter and
running every test file in its own process; `audit-poison.txt` on the Hub): the four
compaction tests pinning the `-1` tail, two `test_large_offsets.py` tail asserts, and
`test_compact_kernel_initialises_buffer_to_minus_one` (deleted: the contract it pinned is gone).
`test_launch_to_launch_identity` compared whole tensors and passed only because the poison was
deterministic; it now compares `[:counts]`. `torch.library.opcheck`'s
`test_aot_dispatch_dynamic` compares the whole output eager vs AOT and failed on the
unwritten tail; it runs under `torch.use_deterministic_algorithms(True)`, which fills
`torch.empty`, with every utility kept.

## Golden cells

[`golden_compare.py`](golden_compare.py): the golden cell set (goodreads d128 `c0_genre`,
arXiv d128 `c0_maincat` SilverTorch), `bench run --mode eager --skip-perf`, on the pre-change
tree and on this branch (after L1, and again after L2), same box, same command.

| cell | new − base |
|---|---|
| every SilverTorch cell, `linr_v2`, `linr_v3` | 0 on all 24 oracle and held-out metrics |
| `linr_v1_filter_mask` | oracle recall@100 +2.9e-5, recall@1000 +1.5e-4, ndcg@100 +2.1e-5; held-out within ±3e-5 |
| L2 alone (final vs L1-only) | 0 on every metric of every cell |

V1's new oracle numbers equal `linr_v2`/triton's to the ninth digit at @100 and @500: the two
exact algorithms now score alike. V1's golden cell was re-derived with the old harness on this
library (the golden README's method; [`golden-rederive/`](golden-rederive/) holds the
provenance, clock trace and the two-import harness port); its quality columns equal the v2
harness's new numbers above, so V1 meets the golden at 0.

## Timing

[`bench_kernels.py`](bench_kernels.py) (kernel-opt's, less the SilverTorch probe scorers, plus
the paths L1/L2 touched), [`interleave.sh`](interleave.sh) base / new × 3 rounds,
[`compare.py`](compare.py). Predictions were written first: [predictions.md](predictions.md).

| case | base µs | new µs | new/base | noise band | predicted |
|---|---|---|---|---|---|
| postfilter_masked_b16 | 1619.0 | 2245.4 | **1.387** | 0.004 | +20 to +40 % |
| postfilter_b16 | 1374.5 | 1862.3 | 1.355 | 0.003 | +20 to +40 % |
| prefilter_dense_b16 | 1330.4 | 1834.8 | 1.379 | 0.002 | +20 to +40 % |
| postfilter_b1 | 1122.4 | 1203.7 | 1.072 | 0.003 | +0 to +10 % |
| clause_compact_b16 | 1751.2 | 1546.1 | **0.883** | 0.005 | −5 to −12 % |
| clause_compact_b1 | 208.2 | 205.3 | 0.986 | 0.023 | −3 to −10 % (missed: inside noise) |
| bloom_compact_b16 | 1946.0 | 1843.1 | 0.947 | 0.003 | −5 to −12 % |
| bloom_compact_b1 | 275.0 | 260.7 | 0.948 | 0.011 | −3 to −10 % |
| fmkt_ref_b16_k100 | 44560.1 | 45319.2 | 1.017 | 0.000 | not predicted separately |
| oporp_ref_indirect_b16_k5000 | 53548.8 | 54033.5 | 1.009 | 0.000 | not predicted separately |
| the 8 other cases (Triton KNN, filters, int8) | | | 1.000-1.008 | ≤ 0.010 | within noise |

SM clock 1410 MHz on every window except one new-side `postfilter_masked_b16` round
(1275-1410, flagged `unstable`); all JSONs on the Hub. Where L1's cost goes
([`profile_postfilter.py`](profile_postfilter.py), torch.profiler, 50 calls at B=16): the
fp16-input GEMM is unchanged (0.65 → 0.67 ms per call); `torch.topk`'s digit-count pass goes
0.36 → 0.77 ms (fp32 keys: four radix digits over twice the bytes) and the `masked_fill`
0.23 → 0.37 ms.
