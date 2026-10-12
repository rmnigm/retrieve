# Sourced by the v-short-legs drivers: environment, code_version check, 1 Hz clock trace, and
# `step NAME CMD...` (runs CMD pinned, logs rc and seconds, stops the driver on a non-zero rc).
set -u
REPO=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=${R:-/scratch/campaign-v2/results}
EXPECT=${EXPECT:-408b1188d542634b3d18a2f5077bd23537a845fc}
LOG=/scratch/$LEG
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/$LEG HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$R" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

step() {
  local name=$1; shift
  local t0=$(date +%s)
  $PIN $PY -m bench.cli "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
