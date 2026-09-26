# R-k64 prediction (written 2026-09-26 16:31 UTC, before the probe)

KuaiRand-27K: 32,038,726 table rows, 561,486 train rows (2,193 batches of 256), val 24,503,
test 26,221 users. E1c recipe at d64, two item tables (input + output).

- **Probe peak memory ~62 GB (50-75 GB).** Each table is 32.04 M x 64 x 4 B = 8.2 GB; with its dense
  grad and two AdamW moments that is 32.8 GB per table, 65.6 GB for two. Minus nothing: fits under
  80 GB only if activations and the eval scoring stay under ~12 GB. If it OOMs,
  reuse_item_embeddings=true halves it to ~33 GB.
- **Train step ~70 ms** (E1c's 32 ms + ~40 ms of memory traffic for the fused AdamW, grad zeroing and
  clip over 4.1 B parameters), so **~150 s per full epoch**.
- **Full val eval ~80 s** (24.5k users x 32 M items = 9x yambda's score volume, which took ~8.7 s).
- Sizing rule after the probe: eval <= 15 % of wall -> eval_every x epoch_time >= 5.7 x eval_time.

Update 2026-09-26 18:54 UTC (instruction 215350758, before the probe ran): the cap is now
~5 h. At ~150 s/epoch train that allows ~100 epochs of training time plus eval; eval_every and
eval_max_users are set from the probe's measured eval time so eval stays <= ~15 % of the wall.

## Probe result (2026-09-26 18:55-18:59 UTC; `probe.sh`, `probe_*.json`)

Two tables fit: **peak 62.6 GB** (predicted ~62 GB), so no `reuse_item_embeddings`.
- Train: 200 batches in 31.1 s including compile; steady state ~62 ms/step (tqdm 104 -> 200 in 6 s),
  so ~137 s per full epoch of 2,193 batches.
- Test eval (26.2k users x 32 M items): ~13 s (item_embs saved 18:59:10.7, eval_quality 18:59:24.7);
  val (24.5k users) ~12 s. Predicted ~80 s: the chunked scoring is far faster than the yambda scaling.
- Checkpoint I/O per epoch: `_resume.pt` 49.2 GB in 45 s, every epoch; a new best
  (`sasrec-ep*.pt`, 16.4 GB) 15 s. That is ~22-28 % of each epoch and is not eval; it is what the
  trainer does and is not changed here.

## Sizing (written before the R-k64 launch)

- Epoch wall ~137 s train + ~12 s val + ~45-60 s saves = **~210 s**.
- `eval_every=1`, `eval_max_users` unset (full val): eval ~12 / 210 = ~6 % of the wall.
- `num_epochs=80`: 80 x 210 s = ~4.7 h, under the ~5 h cap. `patience=10` (epochs), as in E3.
- Disk: ~90 GB of checkpoints (`_resume` 49 + best 16 + best_model 16 + item_embs 8); the probe dir
  is deleted before launch.
