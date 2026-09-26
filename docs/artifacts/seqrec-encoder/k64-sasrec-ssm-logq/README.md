# R-k64: KuaiRand-27K d64, the E1c recipe (gSASRec body, sampled softmax in-batch 4096 + uniform 8192 + logQ), two item tables

H100 80GB HBM3, sm_mhz 1980 in every sample. Not yet validated, not citable. Per instruction
215350758 this result is not in `docs/validation.md`; the docs agent adds it from here.
W&B: https://wandb.ai/pinkmeme/seqrec-encoder/runs/xkk96p6k (the killed first launch was 4nyn3oq8).
Files: `probe.sh` + `probe_*.json` (memory probe), `prediction.md` (probe result, sizing, the
correction and restart), `command.sh` (the run as relaunched: 48 epochs, patience 10, full val every
epoch), `config.json`, `train_metrics.json`, `eval_quality.json`, `result.json`, `cold_targets.{py,txt}`.

Test (full catalog of 32,038,725 items, `test.parquet`, 26,221 users): ndcg@10 **0.0046**,
ndcg@100 0.0040, recall@10 0.0002, recall@100 **0.0016**, coverage@10 0.0006. No published baseline.
Best val ndcg@10 0.0232 at epoch 3; patience stopped it at epoch 13. Val recall@100 peaked at 0.0104.
Train loss kept falling (16.5 -> 11.3) while val stalled from epoch 3.
249 s/epoch median train (2,193 batches), 2,241 seq/s, peak 62.6 GB, 4407 s training loop
(~78 min wall).

What bounds these numbers (measured, `cold_targets.txt`): **55.3 % of test targets and 34.1 % of
val targets never occur in the train split.** Those items get no positive update, so the model cannot
rank them, and the test day is further from train than the val day (median 246 targets per test user
against 118 per val user). This bounds the metrics; it is not shown to be the whole gap.

Context, not a bar: the earlier A100 gSASRec d128 KuaiRand run (`.chains/e4-kuairand/`) logged val
ndcg@10 0.0133 at epoch 2 and has no final test number; R-k64 had val 0.0218 at epoch 2.
