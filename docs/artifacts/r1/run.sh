#!/usr/bin/env bash
# Roadmap R1: one goodreads d128 cell on the E1c checkpoint (sasrec-ssm-logq-d128), then the
# norm sanity check. cwd evaluation/ of a worktree, evaluation/data -> /data, one idle GPU.
set -euo pipefail
O=/scratch/r1-results
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/r1 /venvs/h2/bin/python -m bench.cli run \
  --dataset goodreads --dim 128 --suite filter --algo linr_v1_filter_mask --backend triton \
  --filter-kind clause --sweep c0_genre --seed 0 --skip-perf --out $O --config-dir config \
  > $O.log 2>&1
/venvs/h2/bin/python ../docs/artifacts/r1/sanity.py config/goodreads.yaml 128 $O/filter/goodreads-d128.jsonl
