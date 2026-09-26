#!/usr/bin/env bash
# base, new, base, new, ... ROUNDS times each (kernel-opt/interleave.sh, with an explicit python).
#   interleave.sh PYTHON BASE_SRC NEW_SRC OUTDIR ROUNDS
set -euo pipefail
PY=$1; BASE=$2; NEW=$3; OUT=$4; ROUNDS=$5
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for side in base new; do
    src=$BASE; [ "$side" = new ] && src=$NEW
    echo "== round $r $side"
    PYTHONPATH="$ROOT/retrieve:$HERE/../kernel-opt" "$PY" "$HERE/bench_d128.py" "$OUT/$side-$r.json" --src "$src"
  done
done
