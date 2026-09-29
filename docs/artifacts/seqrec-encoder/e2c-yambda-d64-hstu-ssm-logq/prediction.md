# E2c prediction (written 2026-09-26 13:27 UTC, before launch, while E2a ran; E2a was stopped at epoch 58 by user decision)

E2a's config plus 4096 in-batch candidates and logQ. Measured basis: E2a (same body, 8192
uniform candidates) 39-42 s/epoch on a quiet box (up to 65 s under host CPU load), ~2,250
seq/s, peak 9.4 GB.

- **Epoch time ~42 s (range 39-50 s), ~2,200 seq/s.** The HSTU body dominates; 4096 more
  candidates add ~1/3 to a loss matmul that is small next to it (E1 -> Gate B differed by
  0.5 s/epoch), and logQ is one gather + add over the candidate row.
- Peak ~10 GB (the `[P, 12288]` fp32 logits add ~1 GB over E2a).
- Total: up to 100 epochs x ~45 s with evals = ~75 min; less if patience triggers.
