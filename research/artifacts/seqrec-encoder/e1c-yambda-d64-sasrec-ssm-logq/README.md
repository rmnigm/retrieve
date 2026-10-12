# E1c: yambda-500m d64, gSASRec body + sampled softmax (in-batch 4096 + uniform 8192) with logQ

E1's command plus `logq=true` (orchestrator go, from the user's proposal). H100 80GB HBM3,
sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/zxkmti7a.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0945**, recall@100 **0.1619**; against
0.0846 / 0.1563: **+0.0099 / +0.0056**, both bars beaten. Best val ndcg@10 0.1010 at epoch 77
(Gate B 0.0913, published 0.0920); early stop at epoch 97. 11.6 s/epoch median (first 20.0 s),
7,809 seq/s, peak 10.7 GB, 1564 s training loop (~26 min wall). Predicted ~11 s, ~11 GB.

Val against Gate B at the same epochs (ndcg@10 / recall@100):

| epoch | E1c | Gate B |
|---|---|---|
| 1 | 0.0109 / 0.0361 | 0.0196 / 0.0457 |
| 9 | 0.0861 / 0.1477 | 0.0364 / 0.0712 |
| 21 | 0.0955 / 0.1604 | 0.0644 / 0.1291 |
| 59 | 0.0993 / 0.1669 | 0.0856 / 0.1579 |
| 77 | 0.1010 / 0.1682 | 0.0886 / 0.1598 |
| 97 | 0.0999 / 0.1675 | 0.0910 / 0.1631 |

Unlike E2c (HSTU, same loss), it does not overfit: val recall@100 keeps rising to epoch ~77.
