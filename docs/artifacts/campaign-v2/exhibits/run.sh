#!/usr/bin/env bash
# EXHIBITS, one run (README.md): fetch every campaign leg in hub-index, one tree per code_version,
# bench report per tree, checks.py, figures.py, gpuh.py, then `bench upload --verify` of the output.
# CPU only. usage: run.sh [--no-upload]
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
: "${UV_PROJECT_ENVIRONMENT:?pin your own venv}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
BASE=${EXHIBITS_BASE:-/scratch/exhibits}
STAMP=$(date -u +%Y%m%d-%H%M)
OUT=$BASE/runs/$STAMP
mkdir -p "$BASE/legs" "$BASE/logs" "$OUT"
cd "$REPO/evaluation"

# every campaign-v2* leg in hub-index, plus the d1 legs the manifest reuses (arxiv) or the estimates read
legs=$(grep -o '^| `campaign-v2[^`]*`' "$REPO/docs/artifacts/hub-index.md" | tr -d '|` ')
legs="$legs d1/arxiv d1/arxiv-deep d1/yfcc10m d1/pubmed"
for leg in $legs; do
  d=$BASE/legs/${leg//\//__}
  [ -f "$d/.done" ] && continue
  uv run bench fetch --path-in-repo "$leg" --results "$d" > "$BASE/logs/${leg//\//__}.log" 2>&1 && touch "$d/.done"
done

# one tree per code_version: a resume key carries the code_version, so one tree would hold both
# versions of a cell and the report would draw them into one curve
rm -rf "$BASE/tree" && mkdir -p "$BASE/tree"/{v2,v21,d1}
for d in "$BASE"/legs/*/; do
  n=$(basename "$d")
  case $n in campaign-v2.1__*) t=v21 ;; campaign-v2__*) t=v2 ;; d1__*) t=d1 ;; *) continue ;; esac
  for s in "$d"*/; do
    s=$(basename "$s"); [ "$s" = logs ] && continue
    mkdir -p "$BASE/tree/$t/$s" && cp -al "$d$s/." "$BASE/tree/$t/$s/"
  done
done

for t in v2 v21 d1; do
  uv run bench report "$BASE/tree/$t" --out "$OUT/report-$t" > "$OUT/report-$t.log" 2>&1
  rm -f "$OUT/report-$t/results.parquet"
done
trees="$BASE/tree/v2 $BASE/tree/v21 $BASE/tree/d1"
uv run python "$HERE/checks.py" "$OUT" $trees
uv run python "$HERE/figures.py" "$OUT" $trees
uv run python "$HERE/gpuh.py" "$OUT/synth-arms.csv" "$BASE/tree/d1/filter/yfcc10m-d192.jsonl" > "$OUT/gpuh-v-yfcc.csv"
ls "$BASE/legs" > "$OUT/legs.txt"

[ "${1:-}" = --no-upload ] || uv run bench upload --results "$OUT" --path-in-repo "artifacts/exhibits/$STAMP" --verify
echo "$OUT"
