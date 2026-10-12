# E2a: yambda-500m d64, HSTU body + sampled softmax over 8192 uniform negatives only

**Not run to completion:** stopped after epoch 58 of 100 by user decision (2026-09-26 13:56 UTC),
because its val trailed Gate B's same-epoch curve from epoch ~19 on. Not yet validated, not citable.
H100 80GB HBM3, sm_mhz 1980 (spot checks). W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/2tgbmsng
(it ends as killed).

`train()` never reached its end, so `config.json` and `eval_quality.json` come from
`score_partial.py`: it rebuilds the config from `command.sh`'s overrides and scores the best
checkpoint (`hstu-ep55`) on val (0.07806, reproducing the logged 0.0781) and test.
`log_epochs.json` is the per-epoch loss, time and val parsed from the log; `result.json` the row.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0743**, recall@100 **0.1379**; against
0.0846 / 0.1563: −0.0103 / −0.0184. Best val ndcg@10 0.0781 at epoch 55.
39.2 s/epoch median, 2,249 seq/s, peak 9.4 GB, ~47 min wall to the kill.

Val ndcg@10 against Gate B (sasrec + gBCE) at the same epochs: 1: 0.0002 / 0.0196;
9: 0.0375 / 0.0364; 17: 0.0585 / 0.0583; 29: 0.0692 / 0.0726; 39: 0.0753 / 0.0782;
55: 0.0781 / 0.0855. Level to epoch 17, behind after.

Prediction miss: predicted 15-25 s/epoch, measured 39-42 s on a quiet box (52-65 s for a few
epochs under host CPU load). The GPU sat at 99 % utilization but ~235 W with the training
thread at 100 % CPU, so the body is not matmul-bound; the cause was not profiled.
