#!/usr/bin/env bash
# FULL-RERUN picks (recipe.yaml `picks`), on PICKS_POD after both pods' stage-1 seed-0 marks: fetch every stage-1 seed-0 leg's
# results.parquet from fr/, run pick.py on them, dry-resolve the result, upload fr/picks, leave $FR/picks.yaml.
# env as queue.sh (TREE VENV CFG FR).
set -eu
FR=${FR:-/scratch/fr} PY=$VENV/bin/python H=$(cd "$(dirname "$0")" && pwd)
F=$FR/fetch-s0; rm -rf "$F"; mkdir -p "$F" "$FR/picks"
cd "$TREE/evaluation"
for pod in pod1 pode; do
  for leg in curve-d768 curve-d256 curve-d128 curve-synth-d768 curve-synth-d256 curve-synth-d128 v3pool v3pool-laion; do
    "$PY" -m bench.cli fetch --path-in-repo "fr/$leg-$pod-s0" --results "$F/$leg-$pod" > /dev/null 2>&1 \
      && echo "fetched fr/$leg-$pod-s0" || echo "absent fr/$leg-$pod-s0 (no such leg on $pod)"
  done
done
"$PY" "$H/pick.py" "$F"/*/results.parquet > "$FR/picks/suites-timed.yaml" 2> "$FR/picks/pick.log"
cat "$FR/picks/pick.log"
"$PY" "$H/prep_config.py" "$TREE" "$CFG" "$FR/picks/suites-timed.yaml"
"$PY" "$H/check_suites.py" "$CFG" "$CFG/suites.yaml" > "$FR/picks/check.log" 2>&1 || { tail -5 "$FR/picks/check.log"; exit 1; }
"$PY" -m bench.cli upload --results "$FR/picks" --path-in-repo fr/picks --verify
cp "$FR/picks/suites-timed.yaml" "$FR/picks.yaml"
