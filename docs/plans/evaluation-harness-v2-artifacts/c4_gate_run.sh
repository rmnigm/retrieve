#!/usr/bin/env bash
# Roadmap C4 — the harness-v2 GPU gate (evaluation-harness-v2.md §6 WP-4), A100 lane.
#
# Run from anywhere on the A100 box once the GPU is released to this lane:
#
#   bash docs/plans/evaluation-harness-v2-artifacts/c4_gate_run.sh
#
# Stages (STAGES="goodreads arxiv gate resume", any subset, in this order):
#   goodreads  bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause
#              --sweep c0_genre --seed 0            → 11 jobs / 14 cells: the five algos ×
#              {triton, torch} + silvertorch/official, each silvertorch at n_probe 24 and 32.
#   arxiv      the same for arxiv c0_maincat, silvertorch only (the golden has triton; the
#              torch and official backends run too, BACKENDS_ARXIV to narrow).
#   gate       c4_gate.py over both JSONL files against evaluation/golden/ → c4/gate.txt.
#   resume     WP-4 (6): a throwaway run of two cheap cells in a separate --out, killed as
#              soon as the first record lands, then --resume must skip it and finish the other.
#
# Clocks: this container cannot lock them (nvidia-smi -lgc is denied; A1 measured 1140 MHz
# under load), so H §7's fallback applies — the SM clock is sampled every 30 s into
# c4/clocks.csv for the whole run, every perf entry carries its own under-load sm_mhz, and
# c4_gate.py reads the 5 % latency criterion against the golden's 1140 MHz.
#
# The shared venv's editable `retrieve` pointer may belong to another worktree, so the
# interpreter is used directly with PYTHONPATH on this checkout — never `uv run` here.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EVAL=$ROOT/evaluation
C4=${C4:-$ROOT/docs/plans/evaluation-harness-v2-artifacts/c4}
OUT=${OUT:-$C4/results}
PY=${PY:-/venvs/integration/bin/python}
STAGES=${STAGES:-"goodreads arxiv gate resume"}
BACKENDS_ARXIV=${BACKENDS_ARXIV:-"triton torch official"}
export PYTHONPATH=$ROOT/retrieve/src:$EVAL
unset CUDA_VISIBLE_DEVICES

mkdir -p "$C4" "$OUT"
cd "$EVAL"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$C4/driver.log"; }
bench() { "$PY" -m retrieval.cli run --out "$OUT" "$@"; }

{
  echo "branch:   $(git -C "$ROOT" rev-parse --abbrev-ref HEAD) @ $(git -C "$ROOT" rev-parse HEAD)"
  echo "dirty:    $(git -C "$ROOT" status --porcelain | wc -l) paths"
  echo "stages:   $STAGES"
  echo "host:     $(hostname)"
  echo "utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nvidia-smi:"; nvidia-smi || true
  "$PY" -c 'import torch, triton; print("torch", torch.__version__); print("triton", triton.__version__)'
  nvidia-smi -lgc 1410 2>&1 | head -1 || true
} > "$C4/provenance.txt" 2>&1

# Clock trace for the whole run (H §7 fallback), 30 s cadence.
if [ ! -f "$C4/clocks.csv" ]; then
  echo "utc,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu" > "$C4/clocks.csv"
fi
( while true; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')" >> "$C4/clocks.csv"
    sleep 30
  done ) &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

for stage in $STAGES; do
  say "=== stage $stage"
  case $stage in
    goodreads)
      bench --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre \
        --seed 0 --resume 2>&1 | tee -a "$C4/goodreads-d128-c0_genre.log"
      ;;
    arxiv)
      args=()
      for b in $BACKENDS_ARXIV; do args+=(--backend "$b"); done
      bench --dataset arxiv --dim 128 --suite filter --filter-kind clause --sweep c0_maincat \
        --algo silvertorch "${args[@]}" --seed 0 --resume 2>&1 | tee -a "$C4/arxiv-d128-c0_maincat.log"
      ;;
    gate)
      set +e
      python3 "$ROOT/docs/plans/evaluation-harness-v2-artifacts/c4_gate.py" \
        "$OUT/filter/goodreads-d128.jsonl" "$OUT/filter/arxiv-d128.jsonl" \
        --golden "$EVAL/golden" --golden-sm-mhz 1140 | tee "$C4/gate.txt"
      say "gate exit=${PIPESTATUS[0]}"
      set -e
      ;;
    resume)
      # WP-4 (6): kill the process mid-run, then --resume continues at the next cell.
      R=$C4/resume; rm -rf "$R"; mkdir -p "$R"
      "$PY" -m retrieval.cli run --out "$R" --dataset goodreads --dim 128 --suite filter \
        --filter-kind clause --sweep c0_genre --seed 0 --algo linr_v1_filter_mask \
        --backend triton --backend torch --resume > "$R/first.log" 2>&1 &
      RUN_PID=$!
      say "resume check: pid $RUN_PID, waiting for the first record"
      until [ -f "$R/filter/goodreads-d128.jsonl" ] && [ "$(wc -l < "$R/filter/goodreads-d128.jsonl")" -ge 1 ]; do
        sleep 5
        kill -0 $RUN_PID 2>/dev/null || { say "the run ended before its first record"; break; }
      done
      sleep 20   # well inside the second cell (its build / quality)
      say "killing $RUN_PID (SIGTERM) mid-cell"
      kill -TERM $RUN_PID 2>/dev/null || true
      wait $RUN_PID 2>/dev/null || true
      say "records after the kill: $(wc -l < "$R/filter/goodreads-d128.jsonl")"
      "$PY" -m retrieval.cli run --out "$R" --dataset goodreads --dim 128 --suite filter \
        --filter-kind clause --sweep c0_genre --seed 0 --algo linr_v1_filter_mask \
        --backend triton --backend torch --resume 2>&1 | tee "$R/second.log"
      say "resume lines: $(grep -c 'resume:' "$R/second.log" || true); records now: $(wc -l < "$R/filter/goodreads-d128.jsonl")"
      ;;
    *) say "unknown stage $stage"; exit 2 ;;
  esac
done
say "done"
