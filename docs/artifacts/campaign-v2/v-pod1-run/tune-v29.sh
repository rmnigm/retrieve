#!/usr/bin/env bash
# Night-queue item 8, phase 1 (controller 2026-10-12): YFCC 10 M ANN tuning, quality only: tune_config.py's `tune-q` (SilverTorch triton,
# n_lists {2048 ... 16384} x n_probe {8 ... 4096}, clause on yfcc10m tags_and and yfcc10m-synth p 0.01 / 0.1 / 1, bloom on the synth rates),
# seed 0, k 100: 273 cells. Phase 2 (the timed frontier) is tune-v29-timed.sh. Own worktree / venv at v2.9, fresh inductor.
LEG=tune-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/tune-config
$PY "$HERE/tune_config.py" "$C" || exit 1
for ds in yfcc10m yfcc10m-synth; do step "oracle-$ds" oracle --config-dir "$C" --dataset $ds --suite tune-q; done
for ds in yfcc10m yfcc10m-synth; do step "q-$ds" run --config-dir "$C" --dataset $ds --dim 192 --suite tune-q --seed 0 --resume --out "$R"; done
finish
