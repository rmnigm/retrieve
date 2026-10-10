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
legs="$legs d1/arxiv d1/arxiv-deep d1/yfcc10m d1/pubmed ${EXTRA_LEGS:-}"   # EXTRA_LEGS: record trees under artifacts/, or legs not yet in hub-index
for leg in $legs; do
  d=$BASE/legs/${leg//\//__}
  [ -f "$d/.done" ] && continue
  uv run bench fetch --path-in-repo "$leg" --results "$d" > "$BASE/logs/${leg//\//__}.log" 2>&1 && touch "$d/.done"
done

# one tree per code_version: a resume key carries the code_version, so one tree would hold both
# versions of a cell and the report would draw them into one curve
rm -rf "$BASE/tree" && mkdir -p "$BASE/tree"/{v2,v21,v22,v23,v24,v25,v26,v27,v28,v29,v210,v211,d1}
# a suite dir goes to the tree of its records' code_version (a leg may sit under artifacts/)
for d in $(ls -d "$BASE"/legs/*/ | grep -v /artifacts__) $(ls -d "$BASE"/legs/artifacts__*/ 2>/dev/null); do
  for s in "$d"*/; do
    f=$(ls "$s"*.jsonl 2>/dev/null | grep -v samples | head -1) || true
    [ -n "$f" ] || continue
    case $(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["env"]["code_version"][:8])' "$f") in
      408b1188) t=v2 ;; f01255f1) t=v21 ;; 0d23c615) t=v22 ;; 1258a63e) t=v23 ;; d67d6263) t=v24 ;; 472f2fc6) t=v25 ;; 20e83bfc) t=v26 ;; 641ec3b8) t=v27 ;; 78cfbc72) t=v28 ;; e8958bd2) t=v29 ;; a3bec4a5) t=v210 ;; 1d390792) t=v211 ;; 72e5a90c | c0e42d1a) t=d1 ;; *) echo "skip $s: unknown code_version" >&2; continue ;;
    esac
    s=$(basename "$s"); to=$s
    # a profile-only pass over cells already in the tree (H-KSUM's `partial` h2h) keeps its own tree;
    # any other leg with the same suite and dataset joins the tree in a dir of its own, and where it
    # re-times a cell, records.latest keeps the dir read last (`<suite>+<leg>` sorts after `<suite>`)
    if [ -e "$BASE/tree/$t/$s/$(basename "$f")" ]; then
      if uv run python "$HERE/overlap.py" "$BASE/tree/$t/$s/$(basename "$f")" "$f"; then t=$t-$(basename "$d"); else to=$s+$(basename "$d"); fi
    fi
    mkdir -p "$BASE/tree/$t/$to" && cp -al "$d$s/." "$BASE/tree/$t/$to/"
    # readers find a record's per-query sidecars under its suite's name (records are hash-named)
    if [ "$to" != "$s" ]; then
      for pq in "$d$s"/*.perquery; do
        [ -d "$pq" ] && mkdir -p "$BASE/tree/$t/$s/${pq##*/}" && cp -aln "$pq/." "$BASE/tree/$t/$s/${pq##*/}/"
      done
    fi
  done
done

for t in v2 v21 v22 v23 v24 v25 v26 v27 v28 v29 v210 v211 d1; do
  uv run bench report "$BASE/tree/$t" --out "$OUT/report-$t" > "$OUT/report-$t.log" 2>&1
  rm -f "$OUT/report-$t/results.parquet"
done
trees="$BASE/tree/v2 $BASE/tree/v21 $BASE/tree/v22 $BASE/tree/v23 $BASE/tree/v24 $BASE/tree/v25 $BASE/tree/v26 $BASE/tree/v27 $BASE/tree/v28 $BASE/tree/v29 $BASE/tree/v210 $BASE/tree/v211 $BASE/tree/d1"
uv run python "$HERE/checks.py" "$OUT" $trees
uv run python "$HERE/figures.py" "$OUT" $trees
uv run python "$HERE/c1.py" "$OUT" v2.9
uv run python "$HERE/c1_real.py" "$OUT" $trees
uv run python "$HERE/c7_filter.py" "$OUT" $trees
uv run python "$HERE/scaling.py" "$OUT" $(ls -d "$BASE"/tree/*)
uv run python "$HERE/c5.py" "$OUT" $trees
uv run python "$HERE/qps_bands.py" "$OUT" $trees
uv run python "$HERE/v3bits.py" "$OUT" $trees
uv run python "$HERE/t3x.py" "$OUT" $trees $(ls -d "$BASE"/tree/*-* 2>/dev/null)
uv run python "$HERE/gpuh.py" "$OUT/synth-arms.csv" "$BASE/tree/d1/filter/yfcc10m-d192.jsonl" > "$OUT/gpuh-v-yfcc.csv"
ls "$BASE/legs" > "$OUT/legs.txt"

[ "${1:-}" = --no-upload ] || uv run bench upload --results "$OUT" --path-in-repo "artifacts/exhibits/$STAMP" --verify
echo "$OUT"
