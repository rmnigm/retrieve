#!/usr/bin/env bash
# Night queue 5, V-SEEDS arXiv half, scope A + seed 0 (controller): `filter` on arXiv d128 `c3_nversions`, seeds 0-2 on one box,
# V1 / V2 / V3 triton (clause + bloom), SilverTorch triton n_lists 2048 n_probe {24, 256} (clause + bloom) and official bloom {24, 256}:
# 27 jobs, 36 cells, k {100, 1000}, bs {1, 16}, eager + graph, interleaved as the suite defines. One venv for all arms: Meta's extension
# built -O3 (scripts/build_official_o3.sh; env.official_build records it), triton and official share interleave units. Pod d GPU 0.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v210/arxiv-filter-seeds
mkdir -p "$R/logs"
S=(--dataset arxiv --dim 128 --suite filter --sweep c3_nversions --seed 0 --seed 1 --seed 2 --out "$R" --resume)
run seeds-v1v2 run "${S[@]}" --algo linr_v1_filter_mask --algo linr_v2 --backend triton --interleave
run seeds-v3 run "${S[@]}" --algo linr_v3 --backend triton
run seeds-st run "${S[@]}" --algo silvertorch --backend triton --backend official --interleave
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/seeds-*.log "$R/logs/"
run upload-seeds upload --results "$R" --path-in-repo campaign-v2.10/arxiv-filter-seeds --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-seeds.log"
echo "$(date -Is) seeds rc=0"
