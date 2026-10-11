# E1: yambda-500m d64, gSASRec body + sampled softmax (in-batch 4096 + uniform 8192, no logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample, W&B off (launched before W&B was switched on).
Not yet validated, not citable. `command.sh`, `config.json`, `train_metrics.json`,
`eval_quality.json`, `result.json` (made by `../summarize.py`).

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0661**, recall@100 **0.1093**,
against the published checkpoint re-scored on the same file (0.0846 / 0.1563): −0.0185 / −0.0470.
Best val ndcg@10 0.0740 at epoch 95 (Gate B 0.0913 at 99); it ran all 100 epochs.
10.9 s/epoch median (first 19.7 s), 8,337 seq/s, peak 10.7 GB, 1527 s training loop, ~26 min wall.

Val against Gate B (gBCE, same body) at the same epochs, ndcg@10 / coverage@10:

| epoch | E1 | Gate B |
|---|---|---|
| 1 | 0.0001 / 0.0000 | 0.0196 / 0.0000 |
| 9 | 0.0491 / 0.0409 | 0.0364 / 0.0002 |
| 19 | 0.0634 / 0.0592 | 0.0623 / 0.0064 |
| 39 | 0.0685 / 0.0643 | 0.0782 / 0.0245 |
| 99 | 0.0736 / 0.0660 | 0.0913 / 0.0459 |

No collapse: coverage@10 is higher than Gate B's all along. The run learns faster to epoch ~19,
then plateaus ~0.018 below Gate B. The orchestrator's hypothesis (untested): in-batch negatives
without logQ push popular items down; coverage up and recall down fits it.
