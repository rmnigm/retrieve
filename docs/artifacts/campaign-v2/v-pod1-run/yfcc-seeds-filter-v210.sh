#!/usr/bin/env bash
# V-SEEDS YFCC half at campaign-v2.10 (controller 2026-10-12): yfcc10m `filter` (tags_and), linr_v1_filter_mask / linr_v2 / linr_v3 triton,
# seeds 0-2, k {100, 1000}, bs {1, 16}, eager + graph, --interleave: 9 cells, T2 CIs at 10 M. Own worktree / venv at the tag, fresh inductor.
LEG=yfcc-seeds-filter-v210
TAG=${TAG:-campaign-v2.10}
REPO=${REPO:-/scratch/wt/tree-v210}
PY=${PY:-/venvs/v210/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles yfcc10m
step oracle oracle --dataset yfcc10m --suite filter
step run run --dataset yfcc10m --dim 192 --suite filter --algo linr_v1_filter_mask --algo linr_v2 --algo linr_v3 --backend triton \
  --seed 0 --seed 1 --seed 2 --interleave --resume --out "$R"
finish
