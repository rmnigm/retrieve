#!/usr/bin/env bash
# goodreads deep re-time at campaign-v2.9 (controller 2026-10-11): the cells behind C6 at 0.8 M and the QPS@0.95 bands, which were
# pre-ST-WIDE (B * n_probe >= 512). The `deep` suite narrowed to SilverTorch triton clause, seed 0, k 100: n_lists {1024, 4096} x
# n_probe {8 ... 128} x bs {1, 16}, c0_genre first, then c1_lang_reverse and all4. Own worktree / venv at the tag, fresh inductor.
LEG=gr-deep-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
CUT="--dataset goodreads --dim 128 --suite deep --algo silvertorch --backend triton --filter-kind clause --seed 0 --k 100"
old_oracles goodreads
step oracle oracle --dataset goodreads --suite deep
step run-c0 run $CUT --sweep c0_genre --resume --out "$R"
step run-rest run $CUT --sweep c1_lang_reverse --sweep all4 --resume --out "$R"
finish
