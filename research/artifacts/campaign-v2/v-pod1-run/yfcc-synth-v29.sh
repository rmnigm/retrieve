#!/usr/bin/env bash
# AFTER-QUEUE 2 on YFCC at campaign-v2.9 (controller 2026-10-12): yfcc10m-synth, seed 0, k 100, bs {1, 16}, clause. (a) SilverTorch
# triton n_lists 4096 x n_probe {24, 256, 1024} at the suite's five rates (pre-ST-WIDE wide cells) and (b) V1 + V2 (interleaved) +
# SilverTorch at 0.001 / 0.003 / 0.03 / 0.05 (yfcc_synth_config.py widens the sweep list): 35 cells. Own worktree / venv, fresh inductor.
LEG=yfcc-synth-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/synth-config
$PY "$HERE/yfcc_synth_config.py" "$C" || exit 1
old_oracles yfcc10m-synth
step oracle oracle --config-dir "$C" --dataset yfcc10m-synth --suite synth
X="--config-dir $C --dataset yfcc10m-synth --dim 192 --suite synth --filter-kind clause --backend triton --seed 0 --resume --out $R"
step st run $X --algo silvertorch
step v1v2 run $X --algo linr_v1_filter_mask --algo linr_v2 --sweep p0001 --sweep p0003 --sweep p003 --sweep p005 --interleave
finish
