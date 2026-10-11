#!/usr/bin/env bash
# C6 recall-ceiling checks of the final pass (README.md): campaign-v2.11/ceiling-int8-laion30m/ceiling_check.py (the v2.10 script with
# $CEIL_CFG and 32-query scoring batches) at each point of ceiling-points.json for one bench, quality only, from evaluation/ of the
# FINAL_TAG worktree. LAION's synth p001 has no SilverTorch job in the final grid, so its points read a scratch config with a `ceil`
# suite (as the v2.11 driver did); every other point reads the final config/suites.yaml.
# usage: ceiling.sh BENCH OUT_DIR   (env: CUDA_VISIBLE_DEVICES, VENV as for every leg; $VENV/bin/python, not `uv run`, so failures exit nonzero)
set -euo pipefail
BENCH=$1 OUT=$2
: "${VENV:?the leg venv}"
PY=$VENV/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
cd "$REPO/evaluation"
if [ "$BENCH" = laion30m ]; then
  export CEIL_CFG=$OUT/ceil-config
  rm -rf "$CEIL_CFG" && cp -r config "$CEIL_CFG"
  cat >> "$CEIL_CFG/suites.yaml" <<'YAML'

ceil:
  datasets: [laion30m, laion30m-synth]
  dims: [256]
  filter_kinds: [clause]
  ks: [100]
  batch_sizes: [16]
  seeds: [0]
  sweeps:
    laion30m: [c0_domain, tags4]
    laion30m-synth: [p001, p01, p1]
  arms:
    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [4096]}}
YAML
fi
"$PY" - "$HERE/ceiling-points.json" "$BENCH" <<'PY' > "$OUT/points.txt"
import json, sys
for ds, suite, sw, nl, npb in json.load(open(sys.argv[1]))[sys.argv[2]]: print(ds, suite, sw, nl, npb)
PY
while read -r ds suite sw nl npb; do
  t0=$(date +%s)
  "$PY" "$REPO/docs/artifacts/campaign-v2.11/ceiling-int8-laion30m/ceiling_check.py" "$ds" "$suite" "$sw" "$nl" "$npb" \
    "$OUT/ceiling-$ds-$sw.json" > "$OUT/ceiling-$ds-$sw.log" 2>&1 < /dev/null
  echo "$(date -Is) $ds/$sw rc=0 s=$(( $(date +%s) - t0 ))"
done < "$OUT/points.txt"
