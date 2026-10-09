#!/usr/bin/env bash
# C5 ours below 30 M at campaign-v2.9 (controller 2026-10-11; the old cells at n_probe 32-128 bs 16 are pre-ST-WIDE): the codesign
# suite, triton only, bloom_path partial vs full interleaved, goodreads c0_genre + arxiv c0_maincat, n_probe {8, 32, 128}, bs {1, 16},
# k 100, seed 0, eager + graph. Own worktree / venv at the tag, fresh inductor dir.
LEG=codesign-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
CUT="--dim 128 --suite codesign --algo silvertorch --backend triton --filter-kind bloom --seed 0 --interleave --resume --out $R"
old_oracles goodreads arxiv
step oracle-gr oracle --dataset goodreads --suite codesign --sweep c0_genre
step oracle-ax oracle --dataset arxiv --suite codesign --sweep c0_maincat
step run-gr run --dataset goodreads --sweep c0_genre $CUT
step run-ax run --dataset arxiv --sweep c0_maincat $CUT
finish
