# Gate B epoch-time prediction (written before launch, 2026-09-26)

Reference: the published d64 run, A100, old trainer: 2809 s total / 100 epochs = 28.1 s per
epoch including a val eval every 2 epochs (per-position negatives `[P, 256, 64]` gather,
DataLoader with 4 workers and host-side negative sampling).

What was already seen before this was written (so this is not a blind prediction):
- 50-batch smoke on the H100 (eager, cold): 26 ms/step -> ~9.4 s/epoch was the first guess.
- compile A/B (`../compile/ab.txt`, 3 epochs each, no eval): eager 4.8-5.9 s/epoch,
  compiled 3.93-4.03 s/epoch steady; sm_mhz 1980 throughout.

Prediction for Gate B (H100, compile=true): training part 4.0 s/epoch (~23k seq/s); val eval
(45.8k users x 1.87M items) ~5 s every 2 epochs; total ~100 x 4.0 + 50 x 5 + test ~= 11 min,
i.e. ~6.5 s/epoch amortized vs 28.1 s on the A100 (~4x; most of it the removed per-position
gather and the host data path, the rest H100 vs A100). Peak memory well under the A100 run's
14.8 GB: ~4.5 GB (A/B peak) plus the eval scoring buffers.
