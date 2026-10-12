# E1c prediction (written 2026-09-26 14:42 UTC, before launch, while E2b ran)

E1's config plus logQ. Measured basis: E1 10.9 s/epoch, 8,400 seq/s, peak 10.7 GB; E2c showed
logQ costs ~nothing in time (42.3 s vs E2a 39.2 s, mostly the 4096 extra candidates E1 already has).

- **Epoch time ~11 s (10.8-12 s), ~8,300 seq/s.** First epoch ~20 s (cold cache).
- Peak ~11 GB.
- Total: up to 100 epochs x ~12.5 s with evals = ~21 min; if it peaks early like E2c, patience
  stops it around epoch 40-60 (~10-13 min).
