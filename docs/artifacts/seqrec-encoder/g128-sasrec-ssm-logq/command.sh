#!/usr/bin/env bash
# R-g128: goodreads-work-id d128, the E1c recipe (published gSASRec body,
# sampled softmax in-batch 4096 + uniform 8192 + logQ). max_batches_per_epoch unset, eval_every=1;
# embedding_dim=128, ffn_hidden_dim=512; E3's epoch cap and patience; final run set note 181840298. W&B on, entity pinkmeme.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/g128-sasrec-ssm-logq
nohup uv run train run \
  data_dir=/data/goodreads-work-id/trainer checkpoint_dir=/scratch/ckpt/g128-sasrec-ssm-logq \
  encoder=sasrec embedding_dim=128 num_blocks=2 num_heads=2 ffn_hidden_dim=512 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=75 patience=10 eval_every=1 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=g128-sasrec-ssm-logq \
  > /scratch/logs/seqrec/g128-sasrec-ssm-logq.log 2>&1 &
echo $!
