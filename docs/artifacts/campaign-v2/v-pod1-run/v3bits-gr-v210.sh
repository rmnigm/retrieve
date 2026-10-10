#!/usr/bin/env bash
# V-V3BITS remainder at campaign-v2.10 (controller 2026-10-12): the `v3bits` suite's goodreads half (the real `filter` kept sweeps; the
# goodreads-synth half is done at v2.2), linr_v3 triton k_bits {64, 128} x candidate_pool_frac {0.01, 0.05}, clause + bloom, seed 0,
# k {100, 1000}, bs {1, 16}, eager + graph: 16 cells. Own worktree / venv at the tag, fresh inductor.
LEG=v3bits-gr-v210
TAG=${TAG:-campaign-v2.10}
REPO=${REPO:-/scratch/wt/tree-v210}
PY=${PY:-/venvs/v210/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles goodreads
step oracle oracle --dataset goodreads --suite v3bits
step run run --dataset goodreads --dim 128 --suite v3bits --seed 0 --resume --out "$R"
finish
