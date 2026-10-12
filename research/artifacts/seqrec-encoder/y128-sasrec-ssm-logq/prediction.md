# R-y128 prediction (written 2026-09-26 15:19 UTC, before launch, while E3 ran)

E1c at D=128, ffn 512. Measured basis: E1c 11.6 s/epoch, 7,800 seq/s, peak 10.7 GB, ~26 min.
The published d128 run (A100, gBCE) is the only other d128 reference; no H100 d128 measurement.

- **Epoch time ~16 s (13-20 s).** The candidate-logit matmul `[P, 12288] x D` and the body both
  double in D, but part of E1c's step is D-independent (randperm, gathers of ids, softmax over
  12,288 columns, optimizer launches).
- Peak ~16 GB: the `[P, 12288]` fp32 logits do not grow with D; the tables and activations do.
- Total: up to 100 epochs x ~18 s with evals = ~30 min; E1c's patience stop came at epoch 97.
