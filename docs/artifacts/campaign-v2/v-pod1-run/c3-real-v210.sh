#!/usr/bin/env bash
# C3 coverage on real filters at campaign-v2.10 (controller 2026-10-12): c3_real_config.py's `c3-real` suite, LiNR V1 Triton vs torch.compile
# max-autotune, clause, k 100, bs {1, 16}, seed 0, on goodreads / arXiv / YFCC kept sweeps and PubMed 10 M d768: 20 cells, one run per dataset.
# Own worktree / venv at the tag, fresh inductor.
LEG=c3-real-v210
TAG=${TAG:-campaign-v2.10}
REPO=${REPO:-/scratch/wt/tree-v210}
PY=${PY:-/venvs/v210/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/c3-config
$PY "$HERE/c3_real_config.py" "$C" || exit 1
old_oracles goodreads arxiv yfcc10m pubmed
for ds in goodreads arxiv yfcc10m pubmed; do
  d=$(case $ds in yfcc10m) echo 192 ;; pubmed) echo 768 ;; *) echo 128 ;; esac)
  step "oracle-$ds" oracle --config-dir "$C" --dataset $ds --suite c3-real
  step "run-$ds" run --config-dir "$C" --dataset $ds --dim $d --suite c3-real --seed 0 --resume --out "$R"
done
finish
