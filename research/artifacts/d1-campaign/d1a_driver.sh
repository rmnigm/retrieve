#!/usr/bin/env bash
# D1-A: arxiv d128 deep, the remaining linr_v3 jobs, at code_version c0e42d1. One GPU job at a time.
set -u
cd /workspace/retrieve/evaluation
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1a HF_HOME=/scratch/hf
R=/workspace/retrieve/evaluation/results
PY=/venvs/retrieve/bin/python
COMMON=(--dataset arxiv --dim 128 --suite deep --algo linr_v3 --backend triton --filter-kind bloom --out "$R" --config-dir config --resume)

echo "start $(date -u +%FT%TZ)"
$PY -m bench.cli run "${COMMON[@]}" --sweep c0_maincat --seed 1 --seed 2
echo "cmd1 rc=$? $(date -u +%FT%TZ)"
$PY -m bench.cli run "${COMMON[@]}" --sweep c2_year --sweep c3_nversions --sweep c0c2 --sweep all4
echo "cmd2 rc=$? $(date -u +%FT%TZ)"
