#!/usr/bin/env bash
# Alternate baseline and working-tree processes: base, new, base, new, ... ROUNDS times each,
# so clock drift hits both sides alike. Usage:
#   interleave.sh BASE_SRC NEW_SRC OUTDIR ROUNDS [CASES] [SCRIPT]
set -euo pipefail
BASE=$1; NEW=$2; OUT=$3; ROUNDS=$4; CASES=${5:-}; SCRIPT=${6:-$(dirname "$0")/bench_kernels.py}
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for side in base new; do
    src=$BASE; [ "$side" = new ] && src=$NEW
    echo "== round $r $side"
    uv run --no-sync python "$SCRIPT" "$OUT/$side-$r.json" --src "$src" --cases "$CASES"
  done
done
