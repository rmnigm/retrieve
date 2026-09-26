#!/usr/bin/env bash
# R-k64: KuaiRand-27K d64, E1c recipe, two item tables (the probe fit at 62.6 GB). Sized in prediction.md:
# 80 epochs x ~210 s = ~4.7 h, patience 10, full val every epoch (~6 % of wall). W&B on, entity pinkmeme.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/k64-sasrec-ssm-logq
nohup uv run train run \
  data_dir=/data/kuairand checkpoint_dir=/scratch/ckpt/k64-sasrec-ssm-logq \
  encoder=sasrec embedding_dim=64 num_blocks=2 num_heads=2 ffn_hidden_dim=256 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=80 patience=10 eval_every=1 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=k64-sasrec-ssm-logq \
  > /scratch/logs/seqrec/k64-sasrec-ssm-logq.log 2>&1 &
echo $!
