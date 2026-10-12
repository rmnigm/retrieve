#!/usr/bin/env bash
# One goodreads cell (linr_v2, clause c0_genre, k 100, bs 1, eager, seed 0) at two commits of one
# worktree, then the records compared with timing / provenance fields stripped (record_diff.py).
# Usage: UV_PROJECT_ENVIRONMENT=... cell_diff.sh <worktree> <before-rev> <after-rev> <out-root>
set -euo pipefail
wt=$1 before=$2 after=$3 out=$4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
back=$(git -C "$wt" rev-parse --abbrev-ref HEAD)
for side in before after; do
  rev=${!side}
  git -C "$wt" checkout -q --detach "$rev"
  (cd "$wt/evaluation" && flock /scratch/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
     TORCHINDUCTOR_CACHE_DIR="$out/inductor" uv run --frozen bench run --dataset goodreads --dim 128 \
     --suite filter --algo linr_v2 --backend triton --filter-kind clause --sweep c0_genre \
     --k 100 --bs 1 --mode eager --seed 0 --force --out "$out/$side" > "$out/$side.log" 2>&1)
done
git -C "$wt" checkout -q "$back"
python3 -I "$here/record_diff.py" "$out/before" "$out/after"
