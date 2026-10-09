#!/usr/bin/env bash
# C5 ours at campaign-v2.9, arXiv's other codesign sweeps (controller 2026-10-12): c3_nversions + all4, triton only, bloom_path partial
# vs full interleaved, n_probe {8, 32, 128}, bs {1, 16}, k 100, seed 0. Same leg as codesign-v29.sh (records append to its arXiv file and
# logs, so the -ours upload carries all three sweeps); launch with its log as driver-ax.log.
LEG=codesign-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
CUT="--dim 128 --suite codesign --algo silvertorch --backend triton --filter-kind bloom --seed 0 --interleave --resume --out $R"
step oracle-ax2 oracle --dataset arxiv --suite codesign --sweep c3_nversions --sweep all4
step run-ax2 run --dataset arxiv --sweep c3_nversions --sweep all4 $CUT
finish
