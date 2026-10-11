#!/usr/bin/env bash
# yfcc10m-synth V1 + V2 (interleaved) at the remaining rates 0.01 / 0.1 / 0.2 / 0.5 / 1.0 at campaign-v2.9 (controller 2026-10-12), so
# pod 1 has the full nine-rate V1 / V2 curve and its own 10 M crossover in one leg on one box. Clause, seed 0, k 100, bs {1, 16}. Same leg
# as yfcc-synth-v29.sh (records append to its file); launch with its log as driver-hi.log.
LEG=yfcc-synth-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/synth-config
[ -d "$C" ] || $PY "$HERE/yfcc_synth_config.py" "$C" || exit 1
X="--config-dir $C --dataset yfcc10m-synth --dim 192 --suite synth --filter-kind clause --backend triton --seed 0 --resume --out $R"
step v1v2-hi run $X --algo linr_v1_filter_mask --algo linr_v2 --sweep p001 --sweep p01 --sweep p02 --sweep p05 --sweep p1 --interleave
finish
