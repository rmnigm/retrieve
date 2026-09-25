#!/usr/bin/env bash
# d1a_rerun.sh — WP-5's "a rerun is byte-identical in quality", the clause L3 and L5 made
# meetable. Re-runs a handful of stage-a cells with --force into a throwaway JSONL
# (--skip-perf: quality is what the clause is about) and diffs the `quality` block against the
# campaign record, byte for byte.
#
#   flock /workspace/gpu.lock -c 'docs/artifacts/evaluation-harness-v2/d1a/d1a_rerun.sh'
set -euo pipefail
cd "$(dirname "$0")/../../../../evaluation"
OUT=${OUT:-/tmp/d1a-rerun}
rm -rf "$OUT"; mkdir -p "$OUT"
run() {  # dataset algo backend filter_kind sweep
  echo "=== rerun $*"
  python3 -m bench.cli run --dataset "$1" --dim 128 --suite filter --algo "$2" \
    --backend "$3" --filter-kind "$4" --sweep "$5" --seed 0 --skip-perf --force \
    --out "$OUT" --output "$OUT/$1_$2_$3_$4_$5.jsonl" >"$OUT/$1_$2_$3_$4_$5.log" 2>&1
}
# goodreads only: the orchestrator stopped stage a at the dataset boundary, so arxiv has no
# campaign records to diff against. All 5 algos, all 3 backends, both filter kinds, 4 sweeps.
run goodreads linr_v1_filter_mask triton clause c0_genre
run goodreads linr_v2    triton clause c0_genre
run goodreads linr_v3    triton clause c0_genre
run goodreads linr_v4    triton clause c0_genre
run goodreads silvertorch triton   clause c0_genre
run goodreads silvertorch torch    clause c0_genre
run goodreads silvertorch official clause c0_genre
run goodreads linr_v2    torch  bloom  c2_format
run goodreads linr_v3    triton bloom  c3_year
python3 "$(dirname "$0")/d1a_rerun_diff.py" "$OUT"
