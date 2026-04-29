#!/usr/bin/env bash
# Yambda-5B Listen+ end-to-end: prep && train. Drives the chained tmux session.
set -euo pipefail

cd "$(dirname "$0")/.."   # → evaluation/

mkdir -p logs /workspace/hf-cache /workspace/yambda

ts=$(date +%Y%m%d-%H%M%S)
prep_log="logs/prep-5b-${ts}.log"
train_log="logs/train-5b-${ts}.log"

export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_XET=1   # plain HTTP download — Xet hangs on FUSE-mounted /workspace

echo "[pipeline] $(date) starting prep → ${prep_log}"
uv run python -m scripts.prep_yambda \
    --variant 5b --interaction listens \
    --hf-repo yandex/yambda \
    --output /workspace/yambda/5b-listens-plus \
    --max-seq-len 200 \
    2>&1 | tee -a "${prep_log}"

echo "[pipeline] $(date) starting train → ${train_log}"
uv run python -m training.train_sasrec \
    --data-dir /workspace/yambda/5b-listens-plus \
    --checkpoint-dir checkpoints/gsasrec-5b-listens-d64 \
    --embedding-dim 64 --num-blocks 2 --num-heads 2 --dropout 0.1 \
    --reuse-item-embeddings \
    --batch-size 2048 --lr 1e-3 \
    --negs-per-pos 256 --gbce-t 0.75 \
    --num-epochs 100 --eval-every 2 --patience 5 \
    --eval-max-users 100000 --eval-batch-size 512 --eval-score-chunk 1048576 \
    --wandb-run-name yambda-5b-d64 \
    2>&1 | tee -a "${train_log}"

echo "[pipeline] $(date) done"
