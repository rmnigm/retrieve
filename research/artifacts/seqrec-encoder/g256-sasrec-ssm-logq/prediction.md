# R-g256 prediction (written 2026-09-26 16:30 UTC, before launch; R-g128 not yet run)

E3's goodreads command at D=256, ffn 1024, same cap (75 epochs, patience 10, eval every epoch).
Measured basis: E3 (d64) 49.4 s train + ~7 s eval per epoch, 6.7 GB; the yambda D scaling
(11.6 -> 13.8 s from d64 to d128) predicts d128 ~59 s, d256 ~ 77 s.

- **Epoch ~77 s train (65-95 s) + ~10 s eval.**
- **Peak ~13 GB** (tables 797k x 256 x 4 B = 0.8 GB each, x4, plus activations).
- Total: if patience stops it about where E3 stopped (epoch 26), ~40 min; the cap bounds it at
  75 x ~87 s = ~1.8 h.
