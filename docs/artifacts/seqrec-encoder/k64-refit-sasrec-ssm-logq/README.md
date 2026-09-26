# k64-refit: KuaiRand-27K d64, the E1c recipe, trained on train + val (`train_on_val=true`), 4 epochs

H100 80GB HBM3, sm_mhz 1980. Not yet validated, not citable; artifacts only (not in `docs/validation.md`).
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/f11y636d. `command.sh` is R-k64's command plus
`train_on_val=true num_epochs=4`; R-k64's val peaked at epoch 3, its 4th epoch. `encoder=sasrec` is
dropped because the field left `TrainConfig` in the W6 cleanup. No val eval, no early stop; the last
epoch is `best_model.pt` and is scored on test.

Test (full catalog, 32 M items, 26,221 users): ndcg@10 **0.0276**, ndcg@100 0.0197, recall@10 0.0010,
recall@100 **0.0063**, coverage@10 0.0009.

| calibration | ndcg@10 | recall@100 |
|---|---|---|
| R-k64 (train only, best val epoch 3) | 0.0046 | 0.0016 |
| **k64-refit** | **0.0276** | **0.0063** |
| the val day's most-popular list ([diagnosis](../k64-diagnosis/README.md)) | 0.0314 | 0.0073 |

Training on the val day lifts test 6.0x on ndcg@10 and 3.9x on recall@100, confirming the
diagnosis's drift mechanism. The model still sits 12 % / 14 % below one global list of yesterday's
most-clicked items.
260.8 s/epoch median, 2,229 seq/s, peak 62.7 GB, 1239 s training loop (~23 min wall).
Predicted ~260 s/epoch and ~24 min. `_resume.pt` was deleted after the run (disk; the run is final).
