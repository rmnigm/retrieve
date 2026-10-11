#!/usr/bin/env bash
# H-ARMFREE GPU gates (roadmap): one multi-arm 10 M `bench run` without OOM (yfcc10m d192 tags_and clause, V1 + V2 + V3 + SilverTorch
# triton n_probe 24 / 1024, seed 0, bs 16, eager + graph, in one process: the arm-release check runs after every unit), then V2 and
# SilverTorch each in its own process; their records' quality and per-query sidecars must be bit-identical to the multi-arm run's.
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash gate.sh > /scratch/h-armfree/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../../.." && pwd)
OUT=/scratch/h-armfree
PY=/venvs/retrieve/bin/python
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/h-armfree
export PYTHONPATH="$TREE/evaluation:$TREE/retrieve/src"
mkdir -p "$OUT"
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
CELL="--dataset yfcc10m --dim 192 --suite filter --sweep tags_and --filter-kind clause --backend triton --seed 0 --bs 16"
step() {
  local name=$1; shift
  local t0=$(date +%s)
  (cd "$TREE/evaluation" && taskset -c 0-63,128-191 $PY -m bench.cli "$@") > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
step env env
step multi run $CELL --algo linr_v1_filter_mask --algo linr_v2 --algo linr_v3 --algo silvertorch --out "$OUT/multi"
step single-v2 run $CELL --algo linr_v2 --out "$OUT/single"
step single-st run $CELL --algo silvertorch --out "$OUT/single"
grep -c "still allocated" "$OUT/multi.log"
(cd "$TREE/evaluation" && $PY "$HERE/compare.py" "$OUT/single" "$OUT/multi") | tee "$OUT/compare.txt"
echo "$(date -Is) gate done"
