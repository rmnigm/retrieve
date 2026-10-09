#!/usr/bin/env bash
# AFTER-QUEUE 1 (controller 2026-10-12): arXiv-synth 3 M V1 + V2 triton (interleaved), clause, bs {1, 16}, seed 0, k 100, eager + graph,
# at nine rates 0.001 / 0.003 / 0.01 / 0.03 / 0.05 / 0.1 / 0.2 / 0.5 / 1 (synth_cut_config.py widens the suite's seven): pod 1's 3 M
# crossover next to its 10 M one; replaces the pre-CLAUSE-SKIP v2.6 pod-1 cells. Own worktree / venv at v2.9, fresh inductor.
# Hub campaign-v2.9/arxiv-synth-v1v2-pod1 (d-run appends SilverTorch rows to campaign-v2.9/arxiv-synth-synth).
LEG=ax-synth-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
RATES="p0001 p0003 p001 p003 p005 p01 p02 p05 p1"
C=$LOG/synth-config
$PY "$HERE/synth_cut_config.py" "$C" arxiv-synth $RATES || exit 1
old_oracles arxiv-synth
step oracle oracle --config-dir "$C" --dataset arxiv-synth --suite synth
step v1v2 run --config-dir "$C" --dataset arxiv-synth --dim 128 --suite synth --algo linr_v1_filter_mask --algo linr_v2 --backend triton \
  --filter-kind clause --seed 0 --interleave --resume --out "$R"
finish
