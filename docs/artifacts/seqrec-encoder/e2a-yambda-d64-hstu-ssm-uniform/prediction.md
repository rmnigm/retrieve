# E2a prediction (written 2026-09-26 13:07 UTC, before the first launch; that launch was killed after epoch 0 (49.9 s cold) to relaunch with W&B)

HSTU body (hidden 256, 4 blocks, 4 heads, use_time), sampled softmax over 8192 uniform ids
only, B=256, L=200, compile, H100. Reference: E1 (sasrec d64, 12,288 candidates) 10.9 s/epoch,
8,400 seq/s, peak 10.7 GB, 30 ms/step.

- **Epoch time ~18 s (range 15-25 s), ~5,000 seq/s.** The body adds ~540 GFLOP/step (f1
  256->1024 dominates; 2-3 ms of matmul) plus the memory-bound per-block float
  `[256, 4, 200, 200]` bias (position + time gather, masked_fill, cast; no flash kernel):
  ~20 ms/step over E1. Dropping 4096 in-batch candidates saves ~1 ms/step at most.
- First epoch (cold inductor cache): 40-60 s.
- Peak ~18 GB (14-25 GB).
- Total 25-35 min.
