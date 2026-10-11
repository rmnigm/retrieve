#!/usr/bin/env bash
# C5-OURS cells on pod b (GPU 0) at campaign-v2.5: our Triton SilverTorch bloom, bloom_path partial vs full, interleaved,
# one (dataset, suite) per call, one `bench run` chunk per sweep, each under the GPU flock. Usage:
#   driver.sh <dataset> <suite> [config_dir]     (arxiv codesign | goodreads codesign | pubmed codesign-pubmed <dir>)
# Runs from the v2.5 worktree $REPO (PYTHONPATH, not the shared checkout) and refuses unless `bench env` reports
# campaign.yaml's code_version. Holds the phase flag for the whole leg (v-pubmed yields between its chunks) and the timed flag inside each
# locked chunk (the neighbour pauses CPU-heavy work); --resume-safe: rerun the same command into the same tree.
set -u
DS=$1; SUITE=$2; CFG=${3:-config}
REPO=${REPO:-/scratch/wt/v25}
R=/scratch/c5-cells/results-$DS-$SUITE LOG=/scratch/c5-cells/log-$DS-$SUITE
PY=/venvs/retrieve/bin/python PIN="taskset -c 0-95"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf PYTHONPATH=$REPO/evaluation:$REPO/retrieve/src
mkdir -p "$R" "$LOG"
cd "$REPO/evaluation"
touch /scratch/gpu0.st-dloop-phase
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.st-dloop-phase /scratch/gpu0.st-dloop-wants' EXIT
EXPECT=$($PY -c 'import yaml; print(yaml.safe_load(open("campaign.yaml"))["default"]["perf"]["code_version"])')
[ "$($PY -c 'import retrieve; print(retrieve.__file__)')" = "$REPO/retrieve/src/retrieve/__init__.py" ] \
  || { echo "$(date -Is) retrieve not imported from $REPO, refusing"; exit 2; }
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv expect $EXPECT"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  # The timed flag is set inside the lock and cleared only if this chunk created it (a neighbour's leg-long flag stays).
  flock /scratch/gpu0.lock bash -c 'T=/scratch/gpu0.timed; [ -e $T ]; own=$?; touch $T; "$@"; rc=$?
    [ $own = 1 ] && rm -f $T; exit $rc' _ "$@"
  local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
gpu $PIN $PY -m bench.cli oracle --dataset "$DS" --suite "$SUITE" --config-dir "$CFG" > "$LOG/oracle.log" 2>&1 || exit 1
SWEEPS=$($PY -c "
import yaml,sys; s=yaml.safe_load(open('$CFG/suites.yaml'))['$SUITE']['sweeps']['$DS']
d=yaml.safe_load(open('$CFG/$DS.yaml'))['filters']['bloom']; print(' '.join(x for x in s if x in d))")
echo "$(date -Is) sweeps: $SWEEPS"
for sw in $SWEEPS; do
  gpu $PIN $PY -m bench.cli run --dataset "$DS" --suite "$SUITE" --config-dir "$CFG" --backend triton --filter-kind bloom \
    --sweep "$sw" --interleave --resume --out "$R" > "$LOG/run-$sw.log" 2>&1
done
echo "$(date -Is) leg done"
