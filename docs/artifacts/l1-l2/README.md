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
