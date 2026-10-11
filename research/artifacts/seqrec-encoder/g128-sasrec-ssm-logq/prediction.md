# R-g128 prediction (written 2026-09-26 15:19 UTC, before launch, while E3 ran)

E3 at D=128, ffn 512, E3's cap (75 epochs, patience 10, eval every epoch). Measured basis: E3
49.3 s train + ~7 s val eval per epoch, peak 6.6 GB.

- **Epoch ~75 s train (60-90 s) + ~10 s eval**, the same D scaling as R-y128's prediction.
- Peak ~10 GB.
- Total: if it stops like E3 (patience), the time scales with E3's epoch count; the cap bounds it
  at 75 x ~85 s = ~1.8 h, inside 2.5 h.
