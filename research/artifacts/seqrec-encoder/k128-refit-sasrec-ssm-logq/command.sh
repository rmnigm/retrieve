#!/usr/bin/env bash
# k128-refit: KuaiRand-27K d128, one item table, train on train + val, 4 epochs, resume snapshot every 5 epochs
# (so only at the end). Go note 011218745. W&B on, entity pinkmeme.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/k128-sasrec-ssm-logq
nohup uv run train run \
  data_dir=/data/kuairand checkpoint_dir=/scratch/ckpt/k128-refit-sasrec-ssm-logq \
  embedding_dim=128 num_blocks=2 num_heads=2 ffn_hidden_dim=512 dropout=0.5 reuse_item_embeddings=true \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=4 train_on_val=true resume_every=5 patience=10 eval_every=1 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=k128-refit-sasrec-ssm-logq \
  > /scratch/logs/seqrec/k128-refit-sasrec-ssm-logq.log 2>&1 &
echo $!
