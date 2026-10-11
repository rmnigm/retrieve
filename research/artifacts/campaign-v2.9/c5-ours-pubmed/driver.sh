#!/usr/bin/env bash
# C5 ours at 10 M on campaign-v2.9 (ST-WIDE; controller, 2026-10-11): st-dloop's v2.5 PubMed codesign-ours point as is, i.e.
# the one-off `codesign-pubmed` suite (suite-pubmed.yaml, appended to the v2.9 suites in a one-off config dir; keys equal to
# v2.5's): triton bloom_path partial vs full interleaved, c0_mesh, n_lists 4096, n_probe {24, 1024}, bs {1, 16}, eager +
# graph, seed 0. Own Triton-only venv, fresh inductor dir, own results tree, pod a100-x1-b GPU 0 (st-dloop first).
# setsid nohup bash driver.sh > /scratch/c5-ours-v29/driver.log 2>&1 &
set -u
W=/scratch/wt/c5-ours-v29
PY=/venvs/c5o-v29/bin/python
R=/scratch/campaign-v2.9/results
L=/scratch/c5-ours-v29
C=$L/config
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/c5-ours-v29
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
[ -f $C/suites.yaml ] || { rm -rf $C; cp -r config $C; { echo; cat $W/docs/artifacts/campaign-v2.9/c5-ours-pubmed/suite-pubmed.yaml; } >> $C/suites.yaml; }
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
yield_gpu
step $PIN $PY -m bench.cli oracle --config-dir $C --dataset pubmed --dim 768 --suite codesign-pubmed
yield_gpu
touch /scratch/gpu0.timed
log="$R/_logs/codesign-pubmed_pubmed-d768_silvertorch_triton.log"
{ echo "=== bench run --config-dir $C --dataset pubmed --dim 768 --suite codesign-pubmed --resume --interleave"; echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
step $PIN $PY -m bench.cli run --config-dir $C --dataset pubmed --dim 768 --suite codesign-pubmed --out "$R" --resume --interleave >> "$log" 2>&1
{ echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
rm -f /scratch/gpu0.timed
echo "$(date -Is) driver done"
