# R-g128: goodreads-work-id d128, the E1c recipe (gSASRec body, ffn 512, sampled softmax in-batch 4096 + uniform 8192 + logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/fcian7qz. E3's command with
`embedding_dim=128 ffn_hidden_dim=512`: 75-epoch cap, patience 10, eval every epoch.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0410**, recall@100 **0.1518**; against
the published d128 checkpoint re-scored on the same file (0.0361 / 0.1480, E0): **+0.0049 / +0.0038**.
Against E3 (d64, same recipe): +0.0029 / +0.0071.
Best val ndcg@10 0.0440 at epoch 34 (val recall@100 0.1964 there; the published d64 run's best
val recall@100 was 0.1907); early stop at epoch 44.
59.0 s/epoch median train (first 68.8 s), 12,330 seq/s, peak 8.1 GB, 3114 s training loop
(~52 min wall). Predicted ~75 s train + ~10 s eval and ~10 GB; the E3-to-d128 scaling came out
at 1.20x (yambda: 1.19x).
