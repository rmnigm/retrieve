#!/usr/bin/env bash
# ARXIV-N95 on pod a100-x1-b (GPU 0 shared through /scratch/gpu0.lock, NUMA node 0 cores):
# the quality-only n95 suite on arxiv d128 (its median kept filter sweep), then `bench report`
# for its n95. Run by /scratch/arxiv-n95/chain-arxiv-n95.sh after D3 PubMed.
set -u
W=/scratch/wt/arxiv-n95
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2/results
L=/scratch/arxiv-n95
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/arxiv-n95
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation

cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-start.txt"

step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }
step $PIN $PY -m bench.cli oracle --dataset arxiv --dim 128 --suite n95
step $PIN $PY -m bench.cli run --dataset arxiv --dim 128 --suite n95 --out "$R" --resume
step $PY $W/docs/artifacts/campaign-v2/arxiv-n95/n95.py "$R/n95/arxiv-d128.jsonl"   # bench report drops perf: false curves
nvidia-smi -q -d CLOCK > "$L/nvidia-smi-clock-end.txt"
echo "$(date -Is) driver done"
