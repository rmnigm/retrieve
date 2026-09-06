#!/usr/bin/env bash
# A1 / H §6 WP-0 — the golden baseline on the OLD harness.
#
# What this produces: the reference quality numbers that harness v2's GPU
# gate (C4 / H WP-4) has to reproduce within 1e-6. It is the only
# cross-check the rewritten harness gets, so it runs on the old code path
# exactly as shipped, with one fix on top (`load_query_attrs` trims the
# attrs to the users_limit prefix — commit `fix(A1): users_limit
# row-count in load_query_attrs`).
#
# Cells (11 processes, one per (dataset, algo, backend) — the process
# boundary H §8.2 K settles on, so no dynamo cache or CUDA-graph pool
# outlives the backend under test):
#
#   goodreads d128-filter, --filter-kind clause --sweep c0_genre
#     {linr_v1_filter_mask, linr_v2, linr_v3, linr_v4, silvertorch}
#     x {triton, torch}                                        = 10
#   arxiv     d128-filter, --filter-kind clause --sweep c0_maincat
#     silvertorch x {triton}                                   =  1
#
# `triton` and `torch` are the ONLY golden backends. H §6 WP-0's text
# asks for `--backend cuda cute` on arxiv; that is void per H's amendment
# (2026-09-05) — the CUDA C++ and CuTe backends are deleted after the
# official backend's parity gate (roadmap B4), so nothing downstream
# would ever compare against a cuda/cute golden column.
#
# Each process writes 9 rows per cell (3 ks x 3 batch sizes).
#
# WHERE TO RUN: the A100 box, from the repository root, with
# `evaluation/data` pointing at the dataset root. GPU-only: every cell
# needs CUDA. Needs root for the clock lock (see NO_CLOCK_LOCK below).
#
#   cd <repo>
#   bash docs/plans/evaluation-harness-v2-artifacts/a1_golden_run.sh
#
# Env knobs:
#   NO_CLOCK_LOCK=1   skip nvidia-smi lock/unlock (no sudo on the box).
#                     Quality is unaffected; the latency columns are then
#                     not clock-controlled and must be labelled as such.
#   GOLDEN_DIR=...    output dir (default evaluation/golden).
#
# Reruns are safe: a cell whose JSON already exists is skipped, so an
# interrupted run continues where it stopped (the old harness only writes
# rows at the end of a process — H §1 verdict 7 — so a partial cell is
# simply absent). Delete a JSON to force it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation"
GOLDEN_DIR="${GOLDEN_DIR:-$EVAL_DIR/golden}"
LOG_DIR="$GOLDEN_DIR/_logs"

GOODREADS_CONFIG="config/goodreads/d128-filter.yaml"
GOODREADS_SWEEP="c0_genre"
GOODREADS_ALGOS=(linr_v1_filter_mask linr_v2 linr_v3 linr_v4 silvertorch)
GOODREADS_BACKENDS=(triton torch)

ARXIV_CONFIG="config/arxiv/d128-filter.yaml"
ARXIV_SWEEP="c0_maincat"
ARXIV_ALGOS=(silvertorch)
ARXIV_BACKENDS=(triton)

mkdir -p "$GOLDEN_DIR" "$LOG_DIR"

# ----- clocks ---------------------------------------------------------------
# Locked SM clocks so the latency columns are comparable across the 11
# processes and against C4's rerun (H §2.1; CLAUDE.md hard rule 1).

lock_clocks() {
  [ "${NO_CLOCK_LOCK:-0}" = "1" ] && { echo "[clocks] NO_CLOCK_LOCK=1 — not locking"; return 0; }
  echo "[clocks] locking SM clock to 1410 MHz"
  sudo nvidia-smi -pm 1 || echo "[clocks] WARNING: persistence mode failed"
  sudo nvidia-smi -lgc 1410 || echo "[clocks] WARNING: -lgc failed; latencies are NOT clock-controlled"
}

unlock_clocks() {
  [ "${NO_CLOCK_LOCK:-0}" = "1" ] && return 0
  echo "[clocks] resetting SM clock"
  sudo nvidia-smi -rgc || echo "[clocks] WARNING: -rgc failed; clocks left locked"
}

trap unlock_clocks EXIT

# ----- one cell -------------------------------------------------------------

FAILED=()

run_cell() {
  local tag="$1" config="$2" algo="$3" backend="$4" sweep="$5"
  local out="$GOLDEN_DIR/${tag}-${sweep}-${algo}-${backend}.json"
  local log="$LOG_DIR/${tag}-${sweep}-${algo}-${backend}.log"

  if [ -s "$out" ]; then
    echo "[skip] $out exists"
    return 0
  fi

  echo "[run ] $algo/$backend -> $out"
  # `uv run` from evaluation/: the configs' data_dir and filter attrs_path
  # are cwd-relative (retrieval.loaders.resolve_path), so the working
  # directory is load-bearing.
  ( cd "$EVAL_DIR" && uv run evaluate \
      --config "$config" \
      --algo "$algo" \
      --backend "$backend" \
      --filter-kind clause \
      --sweep "$sweep" \
      --output "$out" ) >"$log" 2>&1

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $algo/$backend rc=$rc — see $log"
    rm -f "$out"          # never leave a truncated golden behind
    FAILED+=("${tag}/${algo}/${backend}")
    return 0              # keep going; the summary reports the failures
  fi
  echo "[ ok ] $algo/$backend"
}

# ----- provenance -----------------------------------------------------------

{
  echo "commit:   $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "branch:   $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  echo "dirty:    $(git -C "$REPO_ROOT" status --porcelain | wc -l) paths"
  echo "host:     $(hostname)"
  echo "utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nvidia-smi:"
  nvidia-smi 2>&1 | head -12
  ( cd "$EVAL_DIR" && uv run python -c \
      "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__)" 2>&1 )
} | tee "$LOG_DIR/provenance.txt"

lock_clocks

for algo in "${GOODREADS_ALGOS[@]}"; do
  for backend in "${GOODREADS_BACKENDS[@]}"; do
    run_cell "goodreads-d128" "$GOODREADS_CONFIG" "$algo" "$backend" "$GOODREADS_SWEEP"
  done
done

for algo in "${ARXIV_ALGOS[@]}"; do
  for backend in "${ARXIV_BACKENDS[@]}"; do
    run_cell "arxiv-d128" "$ARXIV_CONFIG" "$algo" "$backend" "$ARXIV_SWEEP"
  done
done

# ----- summary --------------------------------------------------------------

echo
echo "=== golden cells in $GOLDEN_DIR ==="
for f in "$GOLDEN_DIR"/*.json; do
  [ -e "$f" ] || continue
  n=$( (cd "$EVAL_DIR" && uv run python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f") 2>/dev/null || echo "?")
  echo "  $(basename "$f")  rows=$n"
done

if [ ${#FAILED[@]} -ne 0 ]; then
  echo
  echo "=== FAILED cells (${#FAILED[@]}) ==="
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo
echo "all cells present. Next: commit evaluation/golden/*.json, append the"
echo "validation record to docs/plans/evaluation-harness-v2.md and the"
echo "harness half (steps 4-7) of docs/plans/refactor-validation-handoff.md,"
echo "then flip A1 in docs/plans/00-roadmap.md."
