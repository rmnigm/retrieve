# Sourced by the v-pod1-run drivers (pod 1's legs at $TAG, default campaign-v2.7): environment, code_version ==
# the tag's library tree hash, 1 Hz clock trace, `step NAME CMD...` (bench subcommand, pinned) and
# `stream DS SUITE "ALGOS|BACKENDS"`, `old_oracles DS...` (one `bench run --resume --interleave`, per-stream log under
# $R/_logs with clock blocks). Each stops the driver on a non-zero rc. Launch under the GPU 0 lock:
#   setsid nohup flock -n /scratch/gpu0.lock bash <driver> > /scratch/v25/<leg>/driver.log 2>&1 &
set -u
# a free lock means this driver was not launched under it (another holder would have made flock -n fail)
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
TAG=${TAG:-campaign-v2.7}
TV=$(echo "${TAG#campaign-}" | tr -d .)  # v22
REPO=${REPO:-/workspace/retrieve}
export PYTHONPATH="$REPO/evaluation:$REPO/retrieve/src"  # the tree's own harness and library, whichever checkout REPO names
PY=${PY:-/venvs/retrieve/bin/python}
R=${R:-/scratch/campaign-$TV/results}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG=/scratch/$TV/$LEG
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/$LEG-$TV HF_HOME=/scratch/hf
PIN="taskset -c 0-63,128-191"
mkdir -p "$R/_logs" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

EXPECT=$(git -C "$REPO" rev-parse "$TAG:retrieve/src/retrieve") || { echo "$(date -Is) no tag $TAG"; exit 2; }
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) $TAG code_version $cv"
MANIFEST=$($PY -c 'import yaml; print(yaml.safe_load(open("campaign.yaml"))["default"]["perf"]["code_version"])')
[ "$cv" = "$EXPECT" ] && [ "$cv" = "$MANIFEST" ] || {
  echo "$(date -Is) code_version mismatch (tag $EXPECT, campaign.yaml $MANIFEST), refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
{ echo "=== nvidia-smi -q -d CLOCK at start"; nvidia-smi -i 0 -q -d CLOCK; } >> "$LOG/clocks-q.txt"

clocks() { $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; }

step() {
  local name=$1; shift
  local t0=$(date +%s)
  $PIN $PY -m bench.cli "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}

stream() {
  local ds=$1 suite=$2 algos=${3%|*} backends=${3#*|} a b
  $PY "$HERE/../v-gr-deep/pending.py" "$R" "$cv" "$ds" "$suite" $algos 2>/dev/null | tail -1
  local args=(); for a in $algos; do args+=(--algo "$a"); done; for b in $backends; do args+=(--backend "$b"); done
  local name="${suite}_${ds}-d$(dim "$ds")_${algos// /+}_${backends// /+}.log"
  local cmd=($PIN $PY -m bench.cli run --dataset "$ds" --dim "$(dim "$ds")" --suite "$suite" "${args[@]}" --out "$R" --resume --interleave)
  local t0=$(date +%s)
  { echo "=== ${cmd[*]}"; echo "=== clocks at start"; clocks; } >> "$R/_logs/$name"
  "${cmd[@]}" >> "$R/_logs/$name" 2>&1
  local rc=$?
  { echo "=== clocks at end"; clocks; } >> "$R/_logs/$name"
  echo "$(date -Is) stream $ds/$suite ${algos// /+} ${backends// /+} rc=$rc s=$(( $(date +%s) - t0 )) log=$name"
  [ $rc -eq 0 ] || exit $rc
}

# oracles are rebuilt at the tag: blobs built at any other code_version move to <gt_dir>/before-<tag tree[:8]>/
old_oracles() { $PY "$HERE/move_old_oracles.py" "$cv" "$@" || exit 1; }

dim() { case $1 in yfcc10m*) echo 192 ;; *) echo 128 ;; esac; }

finish() {
  { echo "=== nvidia-smi -q -d CLOCK at end"; nvidia-smi -i 0 -q -d CLOCK; } >> "$LOG/clocks-q.txt"
  $PY -c 'import sys; from pathlib import Path; from bench import records; print("=== aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
  echo "$(date -Is) driver done"
}
