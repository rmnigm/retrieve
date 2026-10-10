#!/usr/bin/env bash
# Night-queue item 8, quality phase on goodreads 0.8 M + arXiv 3 M at campaign-v2.10 (controller 2026-10-12): tune_bench_config.py's `tune-q`
# per bench (SilverTorch triton, n_lists {256 ... 16384} x n_probe {8 ... 4096}, n_probe <= n_lists / 4, clause + bloom on the real kept
# sweeps, uniform synth p 0.01 / 0.1 / 1, arXiv's correlated synth), k 100, seed 0: 1,372 cells. Own worktree / venv at the tag, fresh inductor.
LEG=tune-bench-v210
TAG=${TAG:-campaign-v2.10}
REPO=${REPO:-/scratch/wt/tree-v210}
PY=${PY:-/venvs/v210/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
for bench in ${BENCHES:-goodreads arxiv}; do  # BENCHES / DSS narrow a rerun
  C=$LOG/config-$bench
  $PY "$HERE/tune_bench_config.py" "$C" $bench || exit 1
  case $bench in goodreads) dss="goodreads goodreads-synth" ;; arxiv) dss="arxiv arxiv-synth arxiv-corr-synth" ;; esac
  dss=${DSS:-$dss}
  old_oracles $dss
  for ds in $dss; do step "oracle-$ds" oracle --config-dir "$C" --dataset $ds --suite tune-q; done
  for ds in $dss; do step "q-$ds" run --config-dir "$C" --dataset $ds --dim 128 --suite tune-q --seed 0 --resume --out "$R"; done
done
finish
