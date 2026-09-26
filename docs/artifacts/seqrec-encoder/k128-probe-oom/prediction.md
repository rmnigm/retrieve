# R-k128 prediction (written 2026-09-26 18:54 UTC, before the probe; R-k64 not yet run)

KuaiRand-27K d128, ffn 512, one item table (`reuse_item_embeddings=true`, per the instruction:
two tables would need ~131 GB).

- **Probe peak ~72 GB (65-80 GB).** One table 32.04 M x 128 x 4 B = 16.4 GB, x4 with grad and the
  two AdamW moments = 65.6 GB, plus activations (~3 GB at d128 on yambda) and the eval scoring
  chunks. It may OOM; then I stop and write a blocked note.
- **Train step ~ as d64 with two tables** (the same number of table parameters, 4.1 B), so ~150 s
  per full epoch; the d64 probe measures it first.
- **Val eval ~2x d64's** (D doubles the full-catalog score cost).
