# E3: goodreads-work-id d64, the E1c recipe (gSASRec body, sampled softmax in-batch 4096 + uniform 8192 + logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/338ohzq4. `max_batches_per_epoch` unset
(2,907 batches/epoch), `eval_every=1`, `num_epochs=75 patience=10`; the first epoch measured
58.0 s train + ~7 s eval, so the 75-epoch cap fit 2.5 h and the run was not restarted.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0381**, recall@100 **0.1447**; against
the published d64 checkpoint re-scored on the same file (0.0350 / 0.1486, E0): **+0.0031 / −0.0039**.
Best val ndcg@10 0.0402 at epoch 16; early stop at epoch 26. 49.4 s/epoch median train (first
58.0 s), 14,966 seq/s, peak 6.7 GB, 1540 s training loop (~26 min wall).

Prediction miss: predicted ~94 s train epoch and 11-12 GB; measured 49 s and 6.7 GB. The likely
reason (not checked): goodreads histories are shorter than yambda's, so the loss sees fewer real
positions P per batch and the `[P, 12288]` candidate logits, which dominate the step, shrink.

Val: ndcg@10 0.0298 (ep 0), 0.0366 (5), 0.0386 (10), 0.0402 (16), 0.0396 (26); recall@100 peaks at
0.1857 (epoch 20). The published run selected on val recall@100 and reached 0.1907 (epoch 54,
val ndcg@10 0.0365). So on goodreads this recipe ranks the head better and the top-100 worse; the
early-stop metric here is ndcg@10, as the brief sets it.
