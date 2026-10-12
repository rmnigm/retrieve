#!/usr/bin/env bash
# R-y256: the E1c recipe at embedding_dim=256, ffn_hidden_dim=1024 (instruction 192920132). W&B on, entity pinkmeme.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/y256-sasrec-ssm-logq
nohup uv run train run \
  data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/y256-sasrec-ssm-logq \
  encoder=sasrec embedding_dim=256 num_blocks=2 num_heads=2 ffn_hidden_dim=1024 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=100 patience=10 eval_every=2 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=y256-sasrec-ssm-logq \
  > /scratch/logs/seqrec/y256-sasrec-ssm-logq.log 2>&1 &
echo $!
