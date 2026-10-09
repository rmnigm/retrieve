#!/usr/bin/env bash
# V-YFCC deep at campaign-v2.2 (roadmap): first one timed V3 cell at 10 M (pool frac 0.01, seed 0, scratch tree) to
# calibrate the V3 rate, then the whole yfcc10m `deep` suite (SilverTorch n_lists {4096, 16384} x n_probe, V3 pool fracs).
LEG=v-yfcc-deep
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles yfcc10m
step oracle oracle --dataset yfcc10m --suite deep
C=$LOG/calib-config
$PY "$HERE/yfcc_calib_config.py" "$C" || exit 1
step calib-v3 run --config-dir "$C" --dataset yfcc10m --dim 192 --suite deep --out "$LOG/calib" --resume
step campaign-deep campaign --suite deep --dataset yfcc10m --resume --interleave --out "$R"
finish
