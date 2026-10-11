#!/usr/bin/env bash
# Surprise gate rerun: pubmed bloomwidth-timed default width, triton vs official, one interleaved group, --profile.
set -u
W=/scratch/wt/d3-pubmed-timed
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d3-pubmed-timed-v21
cd $W/evaluation
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> /scratch/surprise/clocks.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
touch /scratch/gpu0.timed
flock /scratch/gpu0.lock taskset -c 0-95 /venvs/retrieve/bin/python -m bench.cli run --config-dir /scratch/surprise/config \
  --dataset pubmed --dim 768 --suite surprise-bw --out /scratch/surprise/results-il --interleave --profile
echo "$(date -Is) surprise rc=$?"
