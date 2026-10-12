#!/usr/bin/env bash
# Gate B: published yambda-500m d64 recipe on the new trainer (shared uniform negatives K=256).
cd /scratch/wt/encoder/evaluation
export UV_PROJECT_ENVIRONMENT=/venvs/wt-encoder TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/gate-b-yambda-d64
nohup uv run train run \
  data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/gate-b-yambda-d64 \
  encoder=sasrec loss=gbce num_negatives=256 gbce_t=0.75 \
  embedding_dim=64 num_blocks=2 num_heads=2 ffn_hidden_dim=256 dropout=0.5 \
  batch_size=256 learning_rate=0.001 weight_decay=0 num_epochs=100 patience=20 \
  eval_every=2 eval_batch_size=1024 early_stop_metric=ndcg@10 seed=42 \
  compile=true wandb_enabled=false \
  > /scratch/logs/seqrec/gate-b-yambda-d64.log 2>&1 &
