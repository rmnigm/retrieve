#!/usr/bin/env bash
# E1: yambda-500m d64, published gSASRec body, shared sampled-softmax recipe (loss change alone).
cd /scratch/wt/runs/evaluation
export UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/e1-yambda-d64-sasrec-ssm
nohup uv run train run \
  data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/e1-yambda-d64-sasrec-ssm \
  encoder=sasrec embedding_dim=64 num_blocks=2 num_heads=2 ffn_hidden_dim=256 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=100 patience=10 eval_every=2 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true wandb_enabled=false \
  > /scratch/logs/seqrec/e1-yambda-d64-sasrec-ssm.log 2>&1 &
echo $!
