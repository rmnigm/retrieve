# R-g256: goodreads-work-id d256, the E1c recipe (gSASRec body, ffn 1024, sampled softmax in-batch 4096 + uniform 8192 + logQ)

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/b42n4obv. E3's command with
`embedding_dim=256 ffn_hidden_dim=1024`: 75-epoch cap, patience 10, eval every epoch.
Per instruction 215350758 this row is not in `docs/validation.md`; the docs agent adds it from here.

Test (full catalog, `trainer/test.parquet`): ndcg@10 **0.0418**, recall@100 **0.1530**; against
the published d256 bar given by the orchestrator (0.0354 / 0.1472; not re-scored by e-runs):
**+0.0064 / +0.0058**. Against R-g128: +0.0008 / +0.0012.
Best val ndcg@10 0.0463 at epoch 13 (val recall@100 kept rising slowly to 0.2017 at epoch 20);
early stop at epoch 23. 80.3 s/epoch median train (first 90.1 s), 9,212 seq/s, peak 11.0 GB,
2225 s training loop (~38 min wall). Predicted ~77 s train and ~13 GB.
