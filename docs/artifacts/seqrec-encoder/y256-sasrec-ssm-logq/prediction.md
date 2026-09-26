# R-y256 prediction (written 2026-09-26 16:30 UTC, before launch, while R-g128 was held)

E1c recipe at D=256, ffn 1024. Measured basis on this box: E1c (d64) 11.6 s/epoch, 10.7 GB;
R-y128 (d128) 13.8 s/epoch median, 13.6 GB. Doubling D added 2.2 s and 2.9 GB.

- **Epoch time ~19 s (16-23 s)**: the D-dependent part roughly doubles again (+~4.5 s).
- **Peak ~20 GB** (+~6 GB: tables 1.87 M x 256 x 4 B = 1.9 GB each, x4 with grad and AdamW moments).
- Total: 100 epochs x ~21 s with evals = ~35 min (R-y128 ran all 100 epochs, best at 99).
