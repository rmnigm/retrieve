#!/usr/bin/env bash
# One `bench run` process per dataset (SilverTorch and the picked LiNR arms in one job), graph mode,
# seed 0, on GPU 0. usage: run.sh DATASET...   (suites from gen_suites.py in $CFG)
set -u
CFG=${CFG:-/scratch/strerun/config}
OUT=${OUT:-/scratch/strerun/results}
LOGS=${LOGS:-/scratch/strerun/logs}
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/strerun
cd "$(dirname "$0")/../../../evaluation"
mkdir -p "$OUT" "$LOGS"
for ds in "$@"; do
  log=$LOGS/run-$ds.log
  echo "=== $(date -u +%FT%TZ) start $ds" >> "$log"
  nvidia-smi -q -d CLOCK >> "$log" 2>&1
  uv run bench oracle --dataset "$ds" --suite "strerun-$ds" --config-dir "$CFG" >> "$log" 2>&1
  uv run bench run --dataset "$ds" --suite "strerun-$ds" --config-dir "$CFG" --out "$OUT" \
    --mode graph --interleave --resume >> "$log" 2>&1
  rc=$?
  nvidia-smi -q -d CLOCK >> "$log" 2>&1
  echo "=== $(date -u +%FT%TZ) end $ds rc=$rc" >> "$log"
  echo "$(date -u +%FT%TZ) $ds rc=$rc" >> "$LOGS/done.log"
done
