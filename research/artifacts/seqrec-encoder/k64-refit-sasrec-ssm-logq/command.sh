#!/usr/bin/env bash
# k64-refit: R-k64's command plus train_on_val=true num_epochs=4 (val peaked at epoch 3 = the 4th epoch);
# encoder=sasrec dropped (the field left TrainConfig in the W6 cleanup; sasrec is the only body). Go note 011218745.
cd /scratch/wt/runs/evaluation
export WANDB_ENTITY=pinkmeme UV_PROJECT_ENVIRONMENT=/venvs/wt-runs TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/k64-refit-sasrec-ssm-logq
nohup uv run train run \
  data_dir=/data/kuairand checkpoint_dir=/scratch/ckpt/k64-refit-sasrec-ssm-logq \
  embedding_dim=64 num_blocks=2 num_heads=2 ffn_hidden_dim=256 dropout=0.5 \
  loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096 logq=true \
  batch_size=256 max_seq_length=200 learning_rate=0.001 weight_decay=0 warmup_steps=1000 \
  num_epochs=4 train_on_val=true patience=10 eval_every=1 eval_batch_size=1024 early_stop_metric=ndcg@10 \
  seed=42 compile=true \
  wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=k64-refit-sasrec-ssm-logq \
  > /scratch/logs/seqrec/k64-refit-sasrec-ssm-logq.log 2>&1 &
echo $!
