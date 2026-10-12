#!/usr/bin/env bash
# D1-B: arxiv codesign campaign (10 jobs) at code_version c0e42d1. One GPU job at a time.
set -u
cd /workspace/retrieve/evaluation
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1b HF_HOME=/scratch/hf
R=/workspace/retrieve/evaluation/results

echo "start $(date -u +%FT%TZ)"
/venvs/retrieve/bin/python -m bench.cli campaign --suite codesign --dataset arxiv --resume --timeout 48 --out "$R" --config-dir config
echo "campaign rc=$? $(date -u +%FT%TZ)"
