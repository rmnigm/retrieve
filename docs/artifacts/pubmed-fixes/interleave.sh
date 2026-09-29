#!/usr/bin/env bash
# base, new, base, new, ... ROUNDS times each at width D (l4-pow2-pad/interleave.sh + --d).
#   interleave.sh PYTHON BASE_SRC NEW_SRC OUTDIR ROUNDS D
set -euo pipefail
PY=$1; BASE=$2; NEW=$3; OUT=$4; ROUNDS=$5; D=$6
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for side in base new; do
    src=$BASE; [ "$side" = new ] && src=$NEW
    echo "== round $r $side d$D"
    PYTHONPATH="$ROOT/retrieve:$HERE/../kernel-opt" "$PY" "$HERE/bench_width.py" "$OUT/$side-d$D-$r.json" --d "$D" --src "$src"
  done
done
