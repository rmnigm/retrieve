#!/usr/bin/env bash
# V-YFCC deep (controller ruling 2026-10-10, claims-first): SilverTorch triton at n_lists 4096 x n_probe {24, 256, 1024}, tags_and,
# seed 0, k 100 (a scratch cut of `deep`), and V2 triton bs 16 on `filter` as the exact reference; no V3. 4 cells. C6 / F3 and
# C2's 10 M reference.
LEG=v-yfcc-deep
. "$(dirname "$(readlink -f "$0")")/common.sh"
old_oracles yfcc10m
step oracle-deep oracle --dataset yfcc10m --suite deep
step oracle-filter oracle --dataset yfcc10m --suite filter
C=$LOG/deep-config
$PY "$HERE/yfcc_deep_config.py" "$C" || exit 1
step deep run --config-dir "$C" --dataset yfcc10m --dim 192 --suite deep --out "$R" --resume
# V2 at the suite's ks: the exact gate reads recall at k_max, and YFCC's fp16 ties hold @100 at 0.9886 (known, d1); @1000 passes
step v2-ref run --dataset yfcc10m --dim 192 --suite filter --algo linr_v2 --backend triton --seed 0 --bs 16 --out "$R" --resume
finish
