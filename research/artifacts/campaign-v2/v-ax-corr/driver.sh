#!/usr/bin/env bash
# V-AX-CORR driver on pod c (a100-x1-c): the synth suite on arxiv-corr-synth d128 at campaign-v2.1, GPU 0,
# its own venv and results tree, under the pod's GPU lock. Two passes into the same tree:
#   flock /scratch/gpu0.lock bash driver.sh quality    (bench check + oracle + campaign --skip-perf: partial records)
#   flock /scratch/gpu0.lock bash driver.sh timed      (campaign --interleave: resume re-runs every partial cell)
# The venv /venvs/v-ax-corr is an editable install of this worktree, so `retrieve` and `bench` are imported from
# $REPO and code_version is read from the same tree; the driver refuses unless that is so and it is $EXPECT.
set -u
PASS=$1; DS=arxiv-corr-synth; SUITE=synth; TAG=$DS-$SUITE
REPO=/scratch/wt/v-ax-corr
PY=/venvs/v-ax-corr/bin/python
R=/scratch/campaign-v2.1/results-$TAG
LOG=/scratch/v21/$TAG
EXPECT=f01255f106214ec0f540352d0e2a5cd90b84b6a1
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v21-$TAG HF_HOME=/scratch/hf
PIN="taskset -c 64-127"
mkdir -p "$R" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

for m in retrieve bench; do
  case "$($PY -c "import $m; print($m.__file__)")" in
    "$REPO"/*) ;;
    *) echo "$(date -Is) $m not imported from $REPO, refusing"; exit 2 ;;
  esac
done
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks-$PASS.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

step() {
  local name=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  local rc=$?
  echo "$(date -Is) step $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}

case $PASS in
  quality)
    step check $PY -m bench.cli check --dataset $DS --dim 128
    step oracle $PIN $PY -m bench.cli oracle --dataset $DS --dim 128 --suite $SUITE
    step campaign $PIN $PY -m bench.cli campaign --suite $SUITE --dataset $DS --dim 128 --resume --skip-perf \
      --timeout 48 --out "$R" ;;
  timed)
    step campaign $PIN $PY -m bench.cli campaign --suite $SUITE --dataset $DS --dim 128 --resume --interleave \
      --timeout 48 --out "$R" ;;
  *) echo "usage: driver.sh quality|timed"; exit 2 ;;
esac
