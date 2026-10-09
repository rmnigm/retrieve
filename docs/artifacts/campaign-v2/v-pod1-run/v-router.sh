#!/usr/bin/env bash
# V-ROUTER (roadmap): the `router` suite on goodreads (lq_threshold grid + its two branch arms), at the tag in common.sh.
# Needs dev/v-router in the main checkout (refuses otherwise).
LEG=v-router
. "$(dirname "$(readlink -f "$0")")/common.sh"
[ -f "$REPO/evaluation/bench/router.py" ] || { echo "$(date -Is) main checkout lacks the router arm, refusing"; exit 2; }
old_oracles goodreads
step oracle oracle --dataset goodreads --suite router
step campaign campaign --suite router --dataset goodreads --resume --out "$R"
finish
