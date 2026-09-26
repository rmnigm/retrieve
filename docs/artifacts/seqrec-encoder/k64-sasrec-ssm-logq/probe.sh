#!/usr/bin/env bash
# R-k64 memory probe: KuaiRand-27K d64, E1c recipe, 200 batches, 1 epoch (then val + test eval, which
# also measures the full-catalog eval cost over 32 M items). On OOM: rerun with reuse_item_embeddings=true.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/k64-sasrec-ssm-logq
uv run train run \
  data_dir=/data/kuairand checkpoint_dir=/scratch/ckpt/k64-probe \
  encoder=sasrec embedding_dim=64 num_blocks=2 num_heads=2 ffn_hidden_dim=256 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=1 max_batches_per_epoch=200 eval_every=1 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true "$@" \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=k64-probe \
  > /scratch/logs/seqrec/k64-probe$( [ $# -gt 0 ] && echo -reuse ).log 2>&1
