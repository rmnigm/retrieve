#!/usr/bin/env bash
# L3 gate 5 — the two golden cells the compaction order touches, run TWICE each in the frozen
# golden worktree (old harness, `tmp/golden-rederive`, /venvs/golden) with the L3 library
# swapped in. Same recipe as a1-rederive/a1_golden_rerun.sh: `uv run --no-sync` from the
# worktree's evaluation/ (the configs are cwd-relative), the GPU lock per cell, a private
# inductor cache shared by the four processes, clocks sampled every 30 s (they cannot be locked
# here). Outputs go under this artifacts directory, never into the worktree's evaluation/golden.
#
#   bash docs/artifacts/deterministic-compaction/golden_two_cells_twice.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_WT="${GOLDEN_WT:-/workspace/wt/golden}"
OUT_ROOT="${OUT_ROOT:-$HERE/golden-rederive}"
GPU_LOCK="${GPU_LOCK:-/workspace/gpu.lock}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/venvs/golden}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor-l3-golden}"

CONFIG="config/goodreads/d128-filter.yaml"
SWEEP="c0_genre"
ALGOS=(linr_v2 linr_v3)

mkdir -p "$OUT_ROOT"
{
  echo "worktree: $GOLDEN_WT @ $(git -C "$GOLDEN_WT" rev-parse HEAD) ($(git -C "$GOLDEN_WT" rev-parse --abbrev-ref HEAD))"
  echo "library tree: $(git -C "$GOLDEN_WT" rev-parse HEAD:retrieve)"
  echo "l3 branch: $(git -C "$HERE" rev-parse HEAD)"
  echo "host: $(hostname)  utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi
  (cd "$GOLDEN_WT/evaluation" && uv run --no-sync python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)")
} > "$OUT_ROOT/provenance.txt" 2>&1

CLOCKS="$OUT_ROOT/clocks.csv"
[ -s "$CLOCKS" ] || echo "utc,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu" > "$CLOCKS"
( while true; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$CLOCKS"
    sleep 30
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

for run in run1 run2; do
  mkdir -p "$OUT_ROOT/$run"
  for algo in "${ALGOS[@]}"; do
    out="$OUT_ROOT/$run/goodreads-d128-${SWEEP}-${algo}-triton.json"
    log="$OUT_ROOT/$run/goodreads-d128-${SWEEP}-${algo}-triton.log"
    if [ -s "$out" ]; then echo "[skip] $out"; continue; fi
    echo "[run ] $run $algo/triton -> $out"
    ( cd "$GOLDEN_WT/evaluation" && flock "$GPU_LOCK" uv run --no-sync evaluate \
        --config "$CONFIG" --algo "$algo" --backend triton \
        --filter-kind clause --sweep "$SWEEP" --output "$out" ) > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[FAIL] $run $algo rc=$rc — see $log"; rm -f "$out"; else echo "[ ok ] $run $algo"; fi
  done
done
