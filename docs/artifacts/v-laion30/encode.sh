#!/usr/bin/env bash
# V-LAION30 GPU block on pod d: encode the 30 M captions + 10 k queries (nomic d256), then
# `eval-data laion ingest` (targets on the GPU). Run from the worktree's evaluation/ dir.
set -euo pipefail
GPU=${GPU:-1}
CORES=${CORES:-96-127}
LOG=${LOG:-/scratch/laion-logs}
export UV_PROJECT_ENVIRONMENT=/venvs/laion CUDA_VISIBLE_DEVICES=$GPU
exec 9>/scratch/gpu$GPU.lock
echo "$(date -u +%FT%TZ) waiting for gpu$GPU.lock"
flock 9
echo "laion: V-LAION30 encode + ingest, pid $$, since $(date -u +%FT%TZ)" > /scratch/gpu$GPU-holder
trap 'rm -f /scratch/gpu$GPU-holder' EXIT
echo "$(date -u +%FT%TZ) holding gpu$GPU"
nvidia-smi --query-gpu=index,name,memory.used,clocks.sm --format=csv,noheader -i "$GPU"
run() { taskset -c "$CORES" uv run --no-sync eval-data laion "$@"; }
run encode_text --device cuda 2>&1 | tee "$LOG/encode_text.log"
run encode_queries --device cuda 2>&1 | tee "$LOG/encode_queries.log"
run ingest --device cuda 2>&1 | tee "$LOG/ingest.log"
echo "$(date -u +%FT%TZ) released gpu$GPU"
