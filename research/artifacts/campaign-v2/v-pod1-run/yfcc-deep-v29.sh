#!/usr/bin/env bash
# V-YFCC deep's SilverTorch cells re-timed at campaign-v2.9 (controller 2026-10-12; C6 at 10 M quoted pre-ST-WIDE): SilverTorch triton
# n_lists 4096 x n_probe {24, 256, 1024}, tags_and, bs {1, 16}, seed 0, k 100 (yfcc_deep_config.py's cut of `deep`), and V2 triton
# bs 16 on `filter` as the exact reference, timed only (--skip-quality: its 10 M quality pass at k 1000 is ~1 h and exact V2's recall
# is not the question). The two come from different suites, so they are not interleaved. Own worktree / venv at the tag, fresh inductor.
LEG=yfcc-deep-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles yfcc10m
step oracle-deep oracle --dataset yfcc10m --suite deep
C=$LOG/deep-config
$PY "$HERE/yfcc_deep_config.py" "$C" || exit 1
step deep run --config-dir "$C" --dataset yfcc10m --dim 192 --suite deep --out "$R" --resume
step v2-ref run --dataset yfcc10m --dim 192 --suite filter --algo linr_v2 --backend triton --seed 0 --bs 16 --skip-quality --out "$R" --resume
finish
