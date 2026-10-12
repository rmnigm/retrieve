#!/usr/bin/env bash
# Seeds 1-2 for yfcc10m-synth V1 + V2 (interleaved) at p 0.05 / 0.1 / 0.2 at campaign-v2.9 (controller 2026-10-12): a CI on the 10 M bs-16
# crossover (p ~0.12 at seed 0). Clause, bs {1, 16}, k 100. Same leg as yfcc-synth-v29.sh (records append to its file); launch with its
# log as driver-seeds.log.
LEG=yfcc-synth-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/seeds-config
$PY "$HERE/synth_cut_config.py" "$C" yfcc10m-synth p0001 p0003 p001 p003 p005 p01 p02 p05 p1 --seeds-at p005 p01 p02 || exit 1
step v1v2-seeds run --config-dir "$C" --dataset yfcc10m-synth --dim 192 --suite synth --algo linr_v1_filter_mask --algo linr_v2 \
  --backend triton --filter-kind clause --seed 1 --seed 2 --sweep p005 --sweep p01 --sweep p02 --interleave --resume --out "$R"
finish
