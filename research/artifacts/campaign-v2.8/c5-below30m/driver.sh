#!/usr/bin/env bash
# C5 below 30 M on Meta's final adapter (controller, 2026-10-11): `codesign`, official only, bloom_path partial vs full
# interleaved, goodreads c0_genre + arXiv c0_maincat, n_probe {8, 32, 128}, seed 0, eager, at campaign-v2.8 with Meta's
# extension built -O3 (/venvs/c7-v28; env.official_build), on pod a100-x1-b GPU 0 (st-dloop first). The real suite,
# selected (records ok, keyed to the suite; bs {1, 16} kept), one bench run per dataset, fresh inductor dir, own tree.
# setsid nohup bash driver.sh > /scratch/c5-below30m/driver.log 2>&1 &
set -u
W=/scratch/wt/c5-below30m-v28
PY=/venvs/c7-v28/bin/python
R=/scratch/campaign-v2.8/results
L=/scratch/c5-below30m
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/c5-below30m-v28
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('campaign.yaml'))['default']['perf']['code_version'])")
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }

yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
for ds in goodreads:c0_genre arxiv:c0_maincat; do
  d=${ds%%:*}; s=${ds##*:}
  yield_gpu
  step $PIN $PY -m bench.cli oracle --dataset $d --dim 128 --suite codesign --sweep $s
  yield_gpu
  touch /scratch/gpu0.timed
  log="$R/_logs/codesign_${d}-d128_official.log"
  { echo "=== bench run --dataset $d --dim 128 --suite codesign --algo silvertorch --backend official --sweep $s --seed 0 --mode eager --resume --interleave"; echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  step $PIN $PY -m bench.cli run --dataset $d --dim 128 --suite codesign --algo silvertorch --backend official --sweep $s \
    --seed 0 --mode eager --out "$R" --resume --interleave >> "$log" 2>&1
  { echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
  rm -f /scratch/gpu0.timed
done
echo "$(date -Is) driver done"
