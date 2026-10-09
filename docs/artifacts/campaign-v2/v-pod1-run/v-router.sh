#!/usr/bin/env bash
# V-ROUTER (roadmap; claims-first trim approved 2026-10-10): the `router` suite on goodreads at seed 0, k 100, clause (the 4
# lq_threshold values + its two branch arms; 18 cells), at the tag in common.sh. Needs the router arm in the main checkout.
LEG=v-router
. "$(dirname "$(readlink -f "$0")")/common.sh"
[ -f "$REPO/evaluation/bench/router.py" ] || { echo "$(date -Is) main checkout lacks the router arm, refusing"; exit 2; }
old_oracles goodreads
step oracle oracle --dataset goodreads --suite router
step run run --dataset goodreads --dim 128 --suite router --seed 0 --k 100 --filter-kind clause --out "$R" --resume
finish
