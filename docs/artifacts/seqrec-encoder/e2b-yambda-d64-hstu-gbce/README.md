# E2b: yambda-500m d64, HSTU body + per-position gBCE (K=256, t=0.75)

**Not run to completion:** stopped during epoch 15 of 100 (epochs 0-14 complete) by user
decision, 2026-09-26 14:43 UTC. Not yet validated, not citable. H100 80GB HBM3, sm_mhz 1980
(spot checks). W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/0kbrrcwb (ends as killed).

As for E2a, `score_partial.py` rebuilds `config.json` from `command.sh` and scores the best
checkpoint (`hstu-ep13`) on val (0.03008, reproducing the logged 0.0301) and test.
`log_epochs.json` holds the per-epoch log; `result.json` the row.

Test: ndcg@10 **0.0290**, recall@100 **0.0658** (−0.0556 / −0.0905 against 0.0846 / 0.1563), at
epoch 13 of a run that was still climbing. The number says nothing about where it would end.
41.1 s/epoch median (first 51.9 s), 2,193 seq/s, peak 11.6 GB (predicted ~41 s, ~15 GB).

Val ndcg@10 against Gate B (sasrec + the same gBCE) at the same epochs: 1: 0.0166 / 0.0196;
5: 0.0181 / 0.0199; 9: 0.0224 / 0.0364; 13: 0.0301 / 0.0487. Gate B leaves the popularity
plateau at epoch ~9; E2b leaves it later and more slowly.
