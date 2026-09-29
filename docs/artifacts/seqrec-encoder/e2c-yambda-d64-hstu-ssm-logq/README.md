# E2c: yambda-500m d64, HSTU body + sampled softmax (in-batch 4096 + uniform 8192) with logQ

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/zae8wz60. `command.sh`, `prediction.md`
(written before launch), `config.json`, `train_metrics.json`, `eval_quality.json`, `result.json`.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0883**, recall@100 **0.1500**; against
0.0846 / 0.1563: **+0.0037 / −0.0063** (beats the ndcg@10 bar, misses the recall@100 bar).
Best val ndcg@10 0.0921 at epoch 21 (Gate B's best 0.0913 at 99); early stop at epoch 41
(patience 10 evals). 42.3 s/epoch median (first 53.5 s, cold cache), 2,155 seq/s, peak 11.9 GB,
1988 s training loop (~34 min wall). Prediction was ~42 s/epoch and ~10 GB.

Val against Gate B at the same epochs (ndcg@10 / recall@100):

| epoch | E2c | Gate B |
|---|---|---|
| 1 | 0.0238 / 0.0588 | 0.0196 / 0.0457 |
| 9 | 0.0851 / 0.1483 | 0.0364 / 0.0712 |
| 21 | 0.0921 / 0.1548 | 0.0644 / 0.1291 |
| 41 | 0.0907 / 0.1452 | 0.0779 / 0.1505 |
| 99 | – | 0.0913 / 0.1636 |

It learns far faster, peaks by epoch 21, then overfits: the train loss keeps falling while val
recall@100 drops 0.0096 and coverage@10 keeps rising. Early stop selects on ndcg@10, which peaks
where recall@100 is still below Gate B's final value. With logQ, epoch-1 val is above Gate B's;
without it (E1, E2a) epoch-1 val was ~0.0002. That supports the in-batch popularity hypothesis
without testing it directly.
