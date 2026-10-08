#!/usr/bin/env bash
# D1-F: goodreads d128 filter on the E1c encoder (sasrec-ssm-logq-d128), D1's four algos, at code_version c0e42d1.
set -u
cd /workspace/retrieve/evaluation
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1f HF_HOME=/scratch/hf
R=/workspace/retrieve/evaluation/results

echo "start $(date -u +%FT%TZ)"
/venvs/retrieve/bin/python -m bench.cli run --dataset goodreads --dim 128 --suite filter \
    --algo linr_v1_filter_mask --algo linr_v2 --algo linr_v3 --algo silvertorch \
    --out "$R" --config-dir config --resume
echo "run rc=$? $(date -u +%FT%TZ)"
