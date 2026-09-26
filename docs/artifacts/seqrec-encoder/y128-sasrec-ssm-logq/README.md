# R-y128: yambda-500m d128, the E1c recipe (gSASRec body, ffn 512, sampled softmax in-batch 4096 + uniform 8192 + logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/03idzv11.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.1006**, recall@100 **0.1662**; against
the published d128 checkpoint re-scored on the same file (0.0811 / 0.1486): **+0.0195 / +0.0176**.
Against E1c (the same recipe at d64): +0.0061 / +0.0043.
Best val ndcg@10 0.1078 at **epoch 99 of 100: still improving at the cap** (val 0.1017 at 39,
0.1044 at 59, 0.1062 at 77, 0.1075 at 97), so this is not a converged number.
13.8 s/epoch median (first 24.0 s), peak 13.6 GB, 2860 s training loop (~48 min wall).
Predicted ~16 s and ~16 GB.

Timing caveat: epochs 27-32 and 38-45 took 18-60 s instead of 13.8 s. Only this job was
visible in `nvidia-smi` and sm_mhz stayed 1980, so this is most likely another tenant on the GPU
that the container does not show. It inflates the mean seq/s (5,182) and the wall time, not the
median epoch time or the metrics.
