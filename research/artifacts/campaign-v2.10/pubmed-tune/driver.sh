#!/usr/bin/env bash
# PubMed 10 M d768 ANN tuning, quality phase (night item 8, controller 2026-10-10) at campaign-v2.10 (a3bec4a5): scratch suite `tune-q`
# (tune_config.py, one config dir per n_lists), n_lists-major ascending, stop file between n_lists. Oracles built at v2.10 into a mirror
# data dir's own gt_d768 (the shared /data blobs are never moved; st-dloop reads them), then torch.equal against the shared blob of the
# same name. Pod a100-x1-b GPU 0, st-dloop first.
# setsid nohup bash driver.sh > /scratch/pubmed-tune-v210/driver.log 2>&1 &
set -u
W=/scratch/wt/pubmed-tune-v210
PY=/venvs/c7-v28/bin/python
R=/scratch/campaign-v2.10/results
L=/scratch/pubmed-tune-v210
C=$L/cfg
M=$L/data/pubmed-medcpt
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/pubmed-tune-v210
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('campaign.yaml'))['default']['perf']['code_version'])")
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }
step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }

step $PY $W/docs/artifacts/campaign-v2.10/pubmed-tune/tune_config.py $C $M
yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
yield_gpu
step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite tune-q --config-dir $C/nl1024
step $PY $W/docs/artifacts/campaign-v2.10/pubmed-tune/oracle_compare.py /data/pubmed-medcpt/gt_d768 $M/gt_d768
for nl in 1024 4096 16384 65536; do
  [ -e $L/tune.stop ] && { echo "$(date -Is) tune.stop: stopped before n_lists $nl"; break; }
  log="$R/_logs/tune-q_pubmed-d768_nl$nl.log"
  yield_gpu
  step $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite tune-q --config-dir $C/nl$nl --out $R --resume >> "$log" 2>&1
done
echo "$(date -Is) driver done"
