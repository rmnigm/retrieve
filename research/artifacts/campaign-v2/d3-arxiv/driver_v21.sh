#!/usr/bin/env bash
# Pod c (a100-x1-c) driver for one timed leg (dataset x suite) at campaign-v2.1: GPU 0, one sequential process
# group, a fresh results tree $RROOT/results-<dataset>-<suite>, under the pod's GPU lock. Usage:
#   setsid nohup flock /scratch/gpu0.lock bash driver.sh arxiv-synth synth > /scratch/v21/arxiv-synth-synth/driver.log 2>&1 &
#   REPO=/scratch/wt/d3-arxiv-v21 setsid nohup flock /scratch/gpu0.lock bash driver.sh arxiv bloomwidth-timed \
#     > /scratch/v21/arxiv-bloomwidth-timed/driver.log 2>&1 &
# --resume-safe: after a crash rerun the same command into the same tree. EXPECT is campaign.yaml's
# default perf code_version at the checked-out staging (campaign-v2.1 for this leg).
# ORACLE_ONLY=1 stops after `bench oracle` (blobs land in <data_dir>/gt_d<dim>, which the campaign reuses).
# The shared venv imports `retrieve` from $LIB while code_version is read from $REPO's git tree:
# the driver refuses unless both library trees are identical and at $EXPECT.
set -u
DS=$1; SUITE=$2; TAG=$DS-$SUITE
REPO=${REPO:-/scratch/wt/v-ax-synth}
LIB=/workspace/retrieve
PY=/venvs/retrieve/bin/python
R=${RROOT:-/scratch/campaign-v2.1}/results-$TAG
LOG=/scratch/v21/$TAG
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v21-$TAG HF_HOME=/scratch/hf
PIN="taskset -c 64-127"
mkdir -p "$R" "$LOG" "$TORCHINDUCTOR_CACHE_DIR"
cd "$REPO/evaluation"

EXPECT=$($PY -c 'import yaml; print(yaml.safe_load(open("campaign.yaml"))["default"]["perf"]["code_version"])')
[ "$EXPECT" != 408b1188d542634b3d18a2f5077bd23537a845fc ] || [ "${ALLOW_V2:-0}" = 1 ] \
  || { echo "$(date -Is) campaign.yaml still at campaign-v2 (408b1188), refusing"; exit 2; }

[ "$($PY -c 'import retrieve; print(retrieve.__file__)')" = "$LIB/retrieve/src/retrieve/__init__.py" ] \
  || { echo "$(date -Is) retrieve not imported from $LIB, refusing"; exit 2; }
diff -rq --exclude=__pycache__ "$REPO/retrieve/src/retrieve" "$LIB/retrieve/src/retrieve" \
  || { echo "$(date -Is) library trees differ, refusing"; exit 2; }
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOG/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
nvidia-smi -q -d CLOCK > "$LOG/clock-q-start.txt"

t0=$(date +%s)
$PIN $PY -m bench.cli oracle --dataset "$DS" --suite "$SUITE"
rc=$?
echo "$(date -Is) step oracle rc=$rc s=$(( $(date +%s) - t0 ))"
[ $rc -eq 0 ] || exit $rc
[ "${ORACLE_ONLY:-0}" = 1 ] && { nvidia-smi -q -d CLOCK > "$LOG/clock-q-end.txt"; exit 0; }

t0=$(date +%s)
$PIN $PY -m bench.cli campaign --suite "$SUITE" --dataset "$DS" --resume --interleave --timeout 48 --out "$R"
rc=$?
echo "$(date -Is) step campaign rc=$rc s=$(( $(date +%s) - t0 ))"
nvidia-smi -q -d CLOCK > "$LOG/clock-q-end.txt"
exit $rc
