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
legs="$legs d1/arxiv d1/arxiv-deep d1/yfcc10m d1/pubmed ${EXTRA_LEGS:-}"   # EXTRA_LEGS: record trees published under artifacts/
for leg in $legs; do
  d=$BASE/legs/${leg//\//__}
  [ -f "$d/.done" ] && continue
  uv run bench fetch --path-in-repo "$leg" --results "$d" > "$BASE/logs/${leg//\//__}.log" 2>&1 && touch "$d/.done"
done

# one tree per code_version: a resume key carries the code_version, so one tree would hold both
# versions of a cell and the report would draw them into one curve
rm -rf "$BASE/tree" && mkdir -p "$BASE/tree"/{v2,v21,v22,v23,v24,v25,v26,d1}
# a suite dir goes to the tree of its records' code_version (a leg may sit under artifacts/)
for d in $(ls -d "$BASE"/legs/*/ | grep -v /artifacts__) $(ls -d "$BASE"/legs/artifacts__*/ 2>/dev/null); do
  for s in "$d"*/; do
    f=$(ls "$s"*.jsonl 2>/dev/null | grep -v samples | head -1) || true
    [ -n "$f" ] || continue
    case $(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["env"]["code_version"][:8])' "$f") in
      408b1188) t=v2 ;; f01255f1) t=v21 ;; 0d23c615) t=v22 ;; 1258a63e) t=v23 ;; d67d6263) t=v24 ;; 472f2fc6) t=v25 ;; 20e83bfc) t=v26 ;; 72e5a90c | c0e42d1a) t=d1 ;; *) echo "skip $s: unknown code_version" >&2; continue ;;
    esac
    s=$(basename "$s")
    # a pass that re-records cells already in the tree (H-KSUM's profile-only h2h) keeps its own tree
    [ -e "$BASE/tree/$t/$s/$(basename "$f")" ] && t=$t-$(basename "$d")
    mkdir -p "$BASE/tree/$t/$s" && cp -al "$d$s/." "$BASE/tree/$t/$s/"
  done
done

for t in v2 v21 v22 v23 v24 v25 v26 d1; do
  uv run bench report "$BASE/tree/$t" --out "$OUT/report-$t" > "$OUT/report-$t.log" 2>&1
  rm -f "$OUT/report-$t/results.parquet"
done
trees="$BASE/tree/v2 $BASE/tree/v21 $BASE/tree/v22 $BASE/tree/v23 $BASE/tree/v24 $BASE/tree/v25 $BASE/tree/v26 $BASE/tree/d1"
uv run python "$HERE/checks.py" "$OUT" $trees
uv run python "$HERE/figures.py" "$OUT" $trees
uv run python "$HERE/c5.py" "$OUT" $trees
uv run python "$HERE/qps_bands.py" "$OUT" $trees
uv run python "$HERE/v3bits.py" "$OUT" $trees
uv run python "$HERE/t3x.py" "$OUT" $trees $(ls -d "$BASE"/tree/*-* 2>/dev/null)
uv run python "$HERE/gpuh.py" "$OUT/synth-arms.csv" "$BASE/tree/d1/filter/yfcc10m-d192.jsonl" > "$OUT/gpuh-v-yfcc.csv"
ls "$BASE/legs" > "$OUT/legs.txt"

[ "${1:-}" = --no-upload ] || uv run bench upload --results "$OUT" --path-in-repo "artifacts/exhibits/$STAMP" --verify
echo "$OUT"
