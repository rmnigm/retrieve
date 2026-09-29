# E3 prediction and epoch cap (written 2026-09-26 15:11 UTC, before launch)

goodreads-work-id trainer: 744,332 train rows (8.1x yambda's 91,806), so 2,907 batches/epoch;
797,084 items; val 190,278 rows. E1c config unchanged. Measured basis: E1c on yambda 11.6 s/epoch
(358 batches, fixed L=200 padding, so cost scales with batches), ~8.7 s per val eval
(45.8k users x 1.87 M items).

- **Train epoch ~94 s** (8.1 x 11.6), first epoch ~105 s (cold cache).
- **Val eval ~15 s** (190k x 797k = 1.8x yambda's score volume); eval_every=1, so ~110 s/epoch.
- Peak ~11-12 GB (the train split on the GPU is 8x larger: ~744k x 201 x 8 B = 1.2 GB).
- **Cap:** 2.5 h = 9,000 s / ~110 s = ~80 epochs, so num_epochs=75 with patience=10 (epochs).
  The published goodreads run stopped at 65 epochs. I re-check the cap on the measured first epoch
  and restart only if the 75 x epoch-time estimate exceeds 2.5 h.
