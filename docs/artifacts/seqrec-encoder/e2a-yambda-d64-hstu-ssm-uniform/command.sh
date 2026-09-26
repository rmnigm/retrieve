#!/usr/bin/env bash
# e2a-yambda-d64-hstu-ssm-uniform: yambda-500m d64, HSTU body (softmax attention, use_time); orchestrator split-E2 instruction.
# W&B on, entity pinkmeme (orchestrator note 160832334); the key comes from WANDB_API_KEY in the environment.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/e2a-yambda-d64-hstu-ssm-uniform
nohup uv run train run \
  data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/e2a-yambda-d64-hstu-ssm-uniform \
  encoder=hstu embedding_dim=64 hidden_dim=256 num_blocks=4 num_heads=4 dropout=0.2 use_time=true \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=0 \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=100 patience=10 eval_every=2 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=e2a-yambda-d64-hstu-ssm-uniform \
  > /scratch/logs/seqrec/e2a-yambda-d64-hstu-ssm-uniform.log 2>&1 &
echo $!
