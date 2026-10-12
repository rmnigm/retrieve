# k128-refit prediction (written 2026-09-26 22:14 UTC, before the probe and launch)

KuaiRand-27K d128, one table (reuse_item_embeddings), train_on_val, 4 epochs, resume_every=5 (one
_resume.pt write, at the end). Basis: R-k64 249 s/epoch train, 62.6 GB, two d64 tables = the same
4.1 B table parameters as one d128 table.
- **Probe peak ~70 GB (65-80 GB).**
- **Epoch ~290 s train** (d128 body and candidate matmul ~1.2x d64, as on yambda/goodreads) + a best-snapshot
  write only on the last epoch (~50 s). No val eval. 4 epochs **~20 min**, plus the final _resume (~50 s),
  saves and test eval (~3 min).
