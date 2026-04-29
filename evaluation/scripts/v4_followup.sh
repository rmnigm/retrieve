#!/usr/bin/env bash
# After v4 (d=64) finishes: upload to HF, then launch v5 (d=128, K=64) on the same GPU.
# Run in its own tmux so v4 can keep running uninterrupted.
set -uo pipefail

cd /root/retrieve/evaluation
export HF_HOME=/root/hf-cache
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

V4_DIR=checkpoints/gsasrec-5b-listens-d64-v4
V4_DONE="${V4_DIR}/train_metrics.json"
V5_DIR=checkpoints/gsasrec-5b-listens-d128-v5

ts=$(date +%Y%m%d-%H%M%S)
fp_log="logs/followup-${ts}.log"
v5_log="logs/train-5b-v5-${ts}.log"

exec > >(tee -a "$fp_log") 2>&1

echo "[followup] $(date) waiting for ${V4_DONE}"
until [ -f "$V4_DONE" ]; do sleep 60; done
echo "[followup] $(date) v4 finished"

# Wait for v4 process to fully release GPU memory.
sleep 60
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[followup] $(date) v4 final test metrics:"
python - <<'PY'
import json
m = json.load(open("checkpoints/gsasrec-5b-listens-d64-v4/train_metrics.json"))
print(json.dumps(m.get("test_metrics", {}), indent=2))
PY

echo "[followup] $(date) uploading v4 to HF (pinkmeme/gsasrec-5b-listens-d64-v4)..."
uv run python -m scripts.upload_checkpoints \
    --owner pinkmeme \
    --checkpoint gsasrec-5b-listens-d64-v4 \
    --private \
  || { echo "[followup] HF upload failed (continuing to v5)"; }

echo "[followup] $(date) launching v5 (d=128, K=64) → ${v5_log}"
uv run python -m training.train_sasrec \
    --data-dir /root/yambda/5b-listens-plus \
    --checkpoint-dir "$V5_DIR" \
    --embedding-dim 128 --num-blocks 2 --num-heads 2 --dropout 0.1 \
    --reuse-item-embeddings \
    --batch-size 2048 --lr 1e-3 \
    --negs-per-pos 64 --gbce-t 0.75 \
    --num-epochs 100 --eval-every 2 --patience 5 \
    --eval-max-users 100000 --eval-batch-size 512 --eval-score-chunk 1048576 \
    --wandb-run-name yambda-5b-d128-v5-bs2048-k64 \
    2>&1 | tee -a "$v5_log"

echo "[followup] $(date) v5 finished"
echo "[followup] $(date) uploading v5 to HF..."
uv run python -m scripts.upload_checkpoints \
    --owner pinkmeme \
    --checkpoint gsasrec-5b-listens-d128-v5 \
    --private \
  || { echo "[followup] v5 HF upload failed"; }

echo "[followup] $(date) all done"
