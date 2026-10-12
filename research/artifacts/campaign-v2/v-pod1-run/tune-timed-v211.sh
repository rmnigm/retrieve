#!/usr/bin/env bash
# Timed tune phase at campaign-v2.11 (controller 2026-10-12): tune_timed_config.py's `tune-timed` from tune-timed-candidates.json (goodreads /
# arXiv threshold + knee points of the v2.10 quality grid; YFCC's v2.9 Pareto points and knees), SilverTorch triton, k 100, bs {1, 16, 64},
# eager + graph, seed 0, --skip-quality (recall from the grids: ST-TOPK / ST-WIDE-2 are bit-exact): 736 cells. YFCC first.
LEG=tune-timed-v211
TAG=${TAG:-campaign-v2.11}
REPO=${REPO:-/scratch/wt/tree-v211}
PY=${PY:-/venvs/v211/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/config
$PY "$HERE/tune_timed_config.py" "$C" "$HERE/tune-timed-candidates.json" || exit 1
for ds in ${DSS:-yfcc10m yfcc10m-synth goodreads goodreads-synth arxiv arxiv-synth arxiv-corr-synth}; do
  d=$(case $ds in yfcc10m*) echo 192 ;; *) echo 128 ;; esac)
  step "oracle-$ds" oracle --config-dir "$C" --dataset $ds --suite tune-timed
  step "t-$ds" run --config-dir "$C" --dataset $ds --dim $d --suite tune-timed --seed 0 --skip-quality --resume --out "$R"
done
finish
