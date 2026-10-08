#!/usr/bin/env bash
# V-PUBMED driver on pod a100-x1-b (1x A100-SXM4-80GB, GPU 0, NUMA node 0 cores), one sequential
# process group, after identity.sh reads EQUAL. `driver.sh n95`: the quality-only n95 suite on
# all5, then `bench report` for its n95. `driver.sh filter` (after pubmed's n95 slots in
# suites.yaml are filled): the filter campaign, every arm, 3 sweeps + bloom c0_mesh, seeds 0-2.
# setsid nohup bash driver.sh <phase> > /scratch/v-pubmed/driver-<phase>.log 2>&1 &
set -u
W=/scratch/wt/v-pubmed
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results
L=/scratch/v-pubmed
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open(\"$W/evaluation/campaign.yaml\"))[\"default\"][\"perf\"][\"code_version\"])")
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-pubmed
PIN="flock /scratch/gpu0.lock taskset -c 0-95"   # pod b shares GPU 0: every GPU command takes the lock
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation

cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }

nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks-$1.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-$1-start.txt"

step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
case $1 in
  n95)
    step $PIN $PY -m bench.cli oracle --dataset pubmed --suite n95
    step $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite n95 --out "$R" --resume
    step $PY -m bench.cli report "$R" --out "$L/report-n95" --only matched --dim 768 --bs 16
    ;;
  filter)
    step $PIN $PY -m bench.cli oracle --dataset pubmed --suite filter
    touch /scratch/gpu0.timed   # pod b: the neighbour worker pauses CPU-heavy work while it exists
    step $PIN $PY -m bench.cli campaign --suite filter --dataset pubmed --dim 768 --out "$R" \
      --resume --interleave --timeout 48
    rm -f /scratch/gpu0.timed
    ;;
esac
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-$1-end.txt"
echo "$(date -Is) driver $1 done"
