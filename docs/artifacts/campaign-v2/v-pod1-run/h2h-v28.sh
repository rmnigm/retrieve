#!/usr/bin/env bash
# T3 / C7 redo at campaign-v2.8 (controller 2026-10-11): the h2h suite on goodreads c0_genre + arxiv c0_maincat, none + bloom, d128,
# SilverTorch triton vs official (score_path fp16 / int32), n_probe 24, bs 1 + 16, k 100 + 1000, seed 0, one interleave group per
# cell, --interleave --profile; from its own worktree (REPO: staging f24f077, library = the tag's 78cfbc72, harness with
# env.official_build) and its own venv built by scripts/build_official_o3.sh (Meta's extension at -O3, OF-11); fresh inductor
# dir. Re-reads C7 / T3 on the reworked official adapter (M1-M4).
LEG=h2h-v28
TAG=${TAG:-campaign-v2.8}
REPO=${REPO:-/scratch/wt/tree-v28}
PY=${PY:-/venvs/v28-o3/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads arxiv
step oracle-gr oracle --dataset goodreads --suite h2h
step oracle-ax oracle --dataset arxiv --suite h2h
step run-gr run --dataset goodreads --dim 128 --suite h2h --seed 0 --interleave --profile --resume --out "$R"
step run-ax run --dataset arxiv --dim 128 --suite h2h --seed 0 --interleave --profile --resume --out "$R"
finish
