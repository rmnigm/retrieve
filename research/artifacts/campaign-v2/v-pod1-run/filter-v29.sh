#!/usr/bin/env bash
# Night-queue item 4 (controller 2026-10-12): the `filter` grid's V1 / V2 / SilverTorch triton and official-bloom cells at campaign-v2.9
# on goodreads, arXiv and YFCC (52 cells), then its linr_v3 triton cells (11), seed 0, bs {1, 16}, k {100, 1000}, eager + graph,
# --interleave; torch arms left out (C3). Runs from /venvs/v29-o3 (Meta's extension at -O3, env.official_build), own worktree, fresh inductor.
LEG=filter-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29-o3/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads arxiv yfcc10m
for ds in goodreads arxiv yfcc10m; do step "oracle-$ds" oracle --dataset $ds --suite filter; done
for ds in goodreads arxiv yfcc10m; do
  step "main-$ds" run --dataset $ds --dim "$(dim $ds)" --suite filter --algo linr_v1_filter_mask --algo linr_v2 --algo silvertorch \
    --backend triton --backend official --seed 0 --interleave --resume --out "$R"
done
for ds in goodreads arxiv yfcc10m; do
  step "v3-$ds" run --dataset $ds --dim "$(dim $ds)" --suite filter --algo linr_v3 --backend triton --seed 0 --resume --out "$R"
done
finish
