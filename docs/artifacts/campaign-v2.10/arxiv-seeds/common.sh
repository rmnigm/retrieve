# Sourced by the drivers: pod d GPU $GPU (default 0; cores 64-95 GPU 0, 96-127 GPU 1) under /scratch/gpu$GPU.lock, the laion v2.10 worktree and its -O3 venv,
# code_version == the campaign-v2.10 library tree, 1 Hz clock trace, `run LOGNAME ARGS...` (one pinned bench call,
# stops the driver on a non-zero rc). Launch:  GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash <driver> > LOG 2>&1 &
set -u
GPU=${GPU:-0}
if flock -n /scratch/gpu$GPU.lock true; then echo "$(date -Is) not launched under /scratch/gpu$GPU.lock, refusing"; exit 3; fi
W=/scratch/wt/laion-v210
PY=/venvs/laion-v210/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=/scratch/laion30m/v210
case $GPU in 0) PIN="taskset -c 64-95" ;; 1) PIN="taskset -c 96-127" ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/arxiv-seeds-v210-gpu$GPU HF_HOME=/scratch/hf
mkdir -p "$L/logs" "$TORCHINDUCTOR_CACHE_DIR"
echo "pod-d laion $(basename "$0") on GPU $GPU ($PIN) since $(date -Is), driver $0" > /scratch/gpu$GPU-holder
trap 'rm -f /scratch/gpu$GPU-holder; kill $SMI 2>/dev/null' EXIT
cd "$W/evaluation"

EXPECT=$(git -C "$W" rev-parse "campaign-v2.10:retrieve/src/retrieve")
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version $cv is not campaign-v2.10's $EXPECT, refusing"; exit 2; }
echo "$(date -Is) code_version $cv"

nvidia-smi -i $GPU --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/logs/clocks-gpu$GPU.csv" &
SMI=$!

run() {  # LOGNAME ARGS...: one bench subcommand
  local name=$1; shift
  local t0=$(date +%s)
  echo "=== $*" >> "$L/logs/$name.log"
  $PIN $PY -m bench.cli "$@" >> "$L/logs/$name.log" 2>&1 < /dev/null
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
