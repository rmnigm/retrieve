# Sourced by pod d's drivers (2x A100, both on NUMA node 1): GPU=$GPU under /scratch/gpu$GPU.lock, cores 64-95 (GPU 0)
# or 96-127 (GPU 1), code_version == the $TAG library tree == campaign.yaml's, 1 Hz clock trace of that GPU,
# `step NAME CMD...` (bench subcommand, pinned), `stream DS SUITE "ALGOS|BACKENDS"` (one `bench run --resume
# --interleave`, log under $R/logs with clock blocks), `upload PREFIX` (bench upload --verify of $R), `note DIR TEXT`.
# Each stops the driver on a non-zero rc. Launch:
#   GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash <driver> > /scratch/v25/<leg>/driver.log 2>&1 &
set -u
: "${GPU:?GPU=0|1}" "${LEG:?}" "${R:?}"
if flock -n /scratch/gpu$GPU.lock true; then echo "$(date -Is) not launched under /scratch/gpu$GPU.lock, refusing"; exit 3; fi
TAG=${TAG:-campaign-v2.5}
TV=$(echo "${TAG#campaign-}" | tr -d .)
REPO=${REPO:-/workspace/retrieve}
PY=${PY:-/venvs/retrieve/bin/python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG=/scratch/$TV/$LEG
CH=$REPO/.chains
case $GPU in 0) PIN="taskset -c 64-95" ;; 1) PIN="taskset -c 96-127" ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/$LEG-$TV HF_HOME=/scratch/hf
mkdir -p "$R/logs" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
echo "pod-d $LEG on GPU $GPU ($PIN) since $(date -Is), driver $0" > /scratch/gpu$GPU-holder
cd "$REPO/evaluation"

EXPECT=$(git -C "$REPO" rev-parse "$TAG:retrieve/src/retrieve") || { echo "$(date -Is) no tag $TAG"; exit 2; }
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) $TAG code_version $cv"
MANIFEST=$($PY -c 'import yaml; print(yaml.safe_load(open("campaign.yaml"))["default"]["perf"]["code_version"])')
[ "$cv" = "$EXPECT" ] && [ "$cv" = "$MANIFEST" ] || {
  echo "$(date -Is) code_version mismatch (tag $EXPECT, campaign.yaml $MANIFEST), refusing"; exit 2; }

nvidia-smi -i $GPU --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
{ echo "=== nvidia-smi -q -d CLOCK at start $(date -Is)"; nvidia-smi -i $GPU -q -d CLOCK; } >> "$LOG/clocks-q.txt"

clocks() { $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; }

note() {  # DIR TEXT: a release note on .chains/<DIR>/
  local f="$CH/$1/$(date -u +%Y-%m-%d-%H%M%S000)-d-run-$LEG.md"
  mkdir -p "$CH/$1"
  printf -- '---\nchain: "%s"\nbranch: "d-run"\ncreated: "%s"\n---\n\n%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" > "$f"
}

step() {
  local name=$1; shift
  local t0=$(date +%s)
  $PIN $PY -m bench.cli "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || { note pod-d "$LEG: step $name failed rc=$rc on GPU $GPU; driver stopped"; exit $rc; }
}

stream() {
  local ds=$1 suite=$2 algos=${3%|*} backends=${3#*|} a b
  local args=(); for a in $algos; do args+=(--algo "$a"); done; for b in $backends; do args+=(--backend "$b"); done
  local name="${suite}_${ds}-d$(dim "$ds")_${algos// /+}_${backends// /+}.log"
  local cmd=($PIN $PY -m bench.cli run --dataset "$ds" --dim "$(dim "$ds")" --suite "$suite" "${args[@]}" --out "$R" --resume --interleave ${SKIP_PERF:+--skip-perf} ${NARROW:-})
  local t0=$(date +%s)
  { echo "=== ${cmd[*]}"; echo "=== clocks at start"; clocks; } >> "$R/logs/$name"
  "${cmd[@]}" >> "$R/logs/$name" 2>&1 < /dev/null
  local rc=$?
  { echo "=== clocks at end"; clocks; } >> "$R/logs/$name"
  rm -rf "$R/_parity"
  echo "$(date -Is) stream $ds/$suite ${algos// /+} ${backends// /+} rc=$rc s=$(( $(date +%s) - t0 )) log=$name"
  [ $rc -eq 0 ] || { note pod-d "$LEG: stream $ds/$suite ${algos// /+} ${backends// /+} failed rc=$rc on GPU $GPU; driver stopped"; exit $rc; }
}

UP=
upload() {  # PREFIX: the whole tree $R, aggregated first; sets UP to the manifest + round-trip lines
  $PY -c 'import sys; from pathlib import Path; from bench import records; print("=== aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
  cp -a "$LOG/clocks-q.txt" "$R/logs/" 2>/dev/null
  UP=$($PY -m bench.cli upload --results "$R" --path-in-repo "$1" --verify 2>&1 | tee -a "$LOG/upload.log" | grep -E "MANIFEST|round trip")
  echo "$(date -Is) upload $1: $UP"
  grep -q "round trip verified" <<< "$UP" || { note pod-d "$LEG: upload $1 failed; $LOG/upload.log; driver stopped"; exit 4; }
}

# oracles are rebuilt at the tag: blobs built at any other code_version move to <gt_dir>/before-<cv[:8]>/
old_oracles() { $PY "$HERE/../../campaign-v2/v-pod1-run/move_old_oracles.py" "$cv" "$@" || exit 1; }

dim() { case $1 in yfcc10m*) echo 192 ;; *) echo 128 ;; esac; }

finish() {
  { echo "=== nvidia-smi -q -d CLOCK at end $(date -Is)"; nvidia-smi -i $GPU -q -d CLOCK; } >> "$LOG/clocks-q.txt"
  echo "$(date -Is) driver done"
}
