# Sourced by cal.sh and grid.sh: pod d GPU 1 (cores 96-127) under /scratch/gpu1.lock, the laion worktree and venv,
# code_version == the campaign-v2.5 library tree, 1 Hz clock trace, `run LOGNAME ARGS...` (one pinned bench call,
# stops the driver on a non-zero rc). Launch:  setsid nohup flock -n /scratch/gpu1.lock bash <driver> > LOG 2>&1 &
set -u
if flock -n /scratch/gpu1.lock true; then echo "$(date -Is) not launched under /scratch/gpu1.lock, refusing"; exit 3; fi
W=/scratch/wt/laion
PY=/venvs/laion/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=/scratch/laion30m
PIN="taskset -c 96-127"
export CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/laion30m-v25 HF_HOME=/scratch/hf
mkdir -p "$L/logs" "$TORCHINDUCTOR_CACHE_DIR"
echo "pod-d laion $(basename "$0") on GPU 1 ($PIN) since $(date -Is), driver $0" > /scratch/gpu1-holder
trap 'rm -f /scratch/gpu1-holder; kill $SMI 2>/dev/null' EXIT
cd "$W/evaluation"

EXPECT=$(git -C "$W" rev-parse "campaign-v2.5:retrieve/src/retrieve")
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version $cv is not campaign-v2.5's $EXPECT, refusing"; exit 2; }
echo "$(date -Is) code_version $cv"

nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/logs/clocks.csv" &
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
