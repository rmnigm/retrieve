# E2b prediction (written 2026-09-26 14:31 UTC, before launch)

HSTU body as E2a/E2c, per-position gBCE (K=256, t=0.75). Measured basis on this box: E2a
39.2 s/epoch (peak 9.4 GB), E2c 42.3 s (11.9 GB); Gate B (sasrec, same gBCE) 10.4 s at 11.2 GB
against E1's 10.9 s at 10.7 GB, so the per-position gather costs about what the shared-candidate
matmul does.

- **Epoch time ~41 s (range 39-45 s), ~2,200 seq/s.**
- **Peak ~15 GB (12-18 GB):** E2a's 9.4 GB plus the `[P, 256, 64]` gather and its grad
  (W1 measured ~6 GB for this over shared sampling).
- Val: gBCE on sasrec rose steadily to epoch 99 in Gate B, so expect no early stop before
  epoch ~60; up to 100 epochs x ~45 s = ~75 min.
