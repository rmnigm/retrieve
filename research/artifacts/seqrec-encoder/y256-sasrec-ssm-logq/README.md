# R-y256: yambda-500m d256, the E1c recipe (gSASRec body, ffn 1024, sampled softmax in-batch 4096 + uniform 8192 + logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/rqmh9ai9.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0966**, recall@100 **0.1481**; against
the published d256 checkpoint re-scored on the same file (0.0814 / 0.1398, E0b): **+0.0152 / +0.0083**.
Against R-y128 (the same recipe at d128): −0.0040 / −0.0181.
Best val ndcg@10 0.1040 at epoch 97 of 100: **still improving at the cap** and slower than d128
(val ndcg@10 / R@100: 0.0747 / 0.1209 at 9, 0.0904 / 0.1336 at 39, 0.0988 / 0.1479 at 69,
0.1036 / 0.1555 at 99; R-y128 was at 0.1017 by epoch 39). Not converged.
18.7 s/epoch median (first 28.6 s), 4,872 seq/s, peak 19.7 GB, 3220 s training loop (~54 min wall,
of which ~22 min val evals at D=256). Predicted ~19 s and ~20 GB.
