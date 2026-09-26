# Predictions, written before any L1/L2 timing was taken (2026-09-26)

Shapes are `bench_kernels.py`'s (3M items, D=128, B=16 unless `_b1`).

| case | prediction | why |
|---|---|---|
| `postfilter_masked_b16` | **+20 to +40 %** | the table read (768 MB fp16) is unchanged, but the `[16, 3M]` score buffer doubles to 192 MB: written by the GEMM, read + written by `masked_fill`, read by `torch.topk`, which also needs 4 radix passes over fp32 instead of 2 over fp16 |
| `postfilter_b1` (new) | **+0 to +10 %** | at B=1 the table read dominates; the score buffer is 12 MB |
| `prefilter_dense_b16` (new) | same as `postfilter_masked_b16` minus the mask: **+20 to +40 %** | |
| `fmkt_b16_k100` | **within noise** | the Triton kernel is untouched |
| `clause_compact_b16`, `bloom_compact_b16` | **−5 to −12 %** | the tail store is `(N − count) · 8 B` per row: at 3M items and a low pass rate close to 384 MB of the kernel's writes at B=16 |
| `clause_compact_b1`, `bloom_compact_b1` | **−3 to −10 %** | same share of the traffic, but launch latency is a larger part of ~210-280 µs |
| everything else | **within noise** | untouched |
