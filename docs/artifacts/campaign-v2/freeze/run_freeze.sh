#!/usr/bin/env bash
# The campaign-v2 freeze gate, gates 3-6 (brief 2026-10-08-023200000-cv2-freeze-brief.md), on one
# A100 granted whole to this lane. Gates 1-2 and 7 are plain commands (see the brief).
#
#   STAGES="golden oracle smoke timed report" bash docs/artifacts/campaign-v2/freeze/run_freeze.sh
#
#   golden  the H2 golden cell set (suites_freeze.py golden: 9 cells, quality only, eager, seed 0,
#           goodreads pinned to gsasrec-d128-drop0.5-id) → golden_vs_h2.py against the H2 records
#           (every metric, max |diff| 0) and c4_gate.py's quality check against evaluation/golden/
#   oracle  bench oracle for every (suite, dataset) of the smoke and timed configs
#   smoke   gate 4: one --skip-perf cell per (arm, filter kind) per suite per dataset, seed 0
#   timed   gate 5: codesign arxiv (one sweep) and h2h goodreads bloom, k 100 bs 16, interleaved
#   report  gate 6: bench report over smoke ∪ timed, with and without --manifest
#
# Outputs under $OUT (never the repo's results tree); SM clock sampled every 10 s into clocks.csv.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
HERE=$ROOT/docs/artifacts/campaign-v2/freeze
OUT=${OUT:-/scratch/cv2-freeze}
H2=${H2:-/scratch/h2-golden-up}  # bench fetch --path-in-repo artifacts/h2 elsewhere
STAGES=${STAGES:-"golden oracle smoke timed report"}
GOLDEN_CKPT=data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/best_model.pt
# The pod profile exports UV_PROJECT_ENVIRONMENT=/venvs/retrieve, so a `:-` default never applies.
export UV_PROJECT_ENVIRONMENT=${FREEZE_VENV:-/venvs/cv2-freeze}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/scratch/inductor/cv2-freeze}
mkdir -p "$OUT"
cd "$ROOT/evaluation"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
py() { uv run python "$@"; }
bench() { uv run bench "$@"; }
datasets_of() {  # suite config → its datasets, one per line
  py -c "import sys, yaml; print('\n'.join(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]['datasets']))" "$1" "$2"
}
suites_of() { py -c "import sys, yaml; print('\n'.join(k for k in yaml.safe_load(open(sys.argv[1])) if k != 'bloom'))" "$1"; }

{ echo "commit $(git -C "$ROOT" rev-parse HEAD) dirty $(git -C "$ROOT" status --porcelain | wc -l)"
  echo "utc $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
  bench env; } >> "$OUT/provenance.txt" 2>&1
[ -f "$OUT/clocks.csv" ] || echo "utc,sm_mhz,mem_mhz,temp_c,power_w,util" > "$OUT/clocks.csv"
( while true; do
    echo "$(date -u +%FT%TZ),$(nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" >> "$OUT/clocks.csv"
    sleep 10
  done ) &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null' EXIT

for stage in $STAGES; do
  say "=== $stage"
  case $stage in
    golden)
      C=$OUT/config-golden; G=$OUT/golden
      py "$HERE/suites_freeze.py" golden "$C" > "$OUT/golden.cells.txt" 2>&1
      bench run --config-dir "$C" --out "$G" --dataset goodreads --dim 128 --suite filter \
        --seed 0 --mode eager --skip-perf --checkpoint "$GOLDEN_CKPT" > "$OUT/golden.goodreads.log" 2>&1
      say "golden goodreads rc=$?"
      bench run --config-dir "$C" --out "$G" --dataset arxiv --dim 128 --suite filter \
        --seed 0 --mode eager --skip-perf > "$OUT/golden.arxiv.log" 2>&1
      say "golden arxiv rc=$?"
      python3 "$HERE/golden_vs_h2.py" "$G" "$H2" --golden "$ROOT/evaluation/golden" | tee "$OUT/golden_vs_h2.md"
      say "golden_vs_h2 rc=${PIPESTATUS[0]}"
      ;;
    oracle)
      for C in "$OUT/config-smoke" "$OUT/config-timed"; do
        for s in $(suites_of "$C/suites.yaml"); do
          for d in $(datasets_of "$C/suites.yaml" "$s"); do
            bench oracle --config-dir "$C" --dataset "$d" --suite "$s" >> "$OUT/oracle.log" 2>&1
            say "oracle $(basename "$C") $s $d rc=$?"
          done
        done
      done
      ;;
    smoke)
      C=$OUT/config-smoke; S=$OUT/smoke
      for s in $(suites_of "$C/suites.yaml"); do
        for d in $(datasets_of "$C/suites.yaml" "$s"); do
          [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $s:$d "* ]] && continue
          [ -n "${SKIP:-}" ] && [[ " $SKIP " == *" $s:$d "* ]] && { say "smoke $s $d skipped (SKIP)"; continue; }
          bench run --config-dir "$C" --out "$S" --dataset "$d" --suite "$s" --skip-perf \
            > "$OUT/smoke.$s.$d.log" 2>&1
          say "smoke $s $d rc=$?"
        done
      done
      py "$HERE/check_records.py" "$S" "$C" | tee "$OUT/smoke.check.txt"
      say "smoke check rc=${PIPESTATUS[0]}"
      ;;
    timed)
      C=$OUT/config-timed; T=$OUT/timed
      bench run --config-dir "$C" --out "$T" --dataset arxiv --suite codesign --interleave \
        > "$OUT/timed.codesign.arxiv.log" 2>&1
      say "timed codesign arxiv rc=$?"
      bench run --config-dir "$C" --out "$T" --dataset goodreads --suite h2h --interleave --profile \
        > "$OUT/timed.h2h.goodreads.log" 2>&1
      say "timed h2h goodreads rc=$?"
      py "$HERE/check_records.py" "$T" "$C" --timed | tee "$OUT/timed.check.txt"
      say "timed check rc=${PIPESTATUS[0]}"
      ;;
    report)
      U=$OUT/union; rm -rf "$U"; mkdir -p "$U"
      for src in "$OUT/smoke" "$OUT/timed"; do  # timed last: its records win a shared key
        (cd "$src" && find . -path ./_parity -prune -o -path ./_logs -prune -o -type f ! -name '*.jsonl' -print) |
          while read -r f; do mkdir -p "$U/$(dirname "$f")"; cp "$src/$f" "$U/$f"; done
        for j in "$src"/*/*.jsonl; do
          rel=${j#"$src"/}; mkdir -p "$U/$(dirname "$rel")"; cat "$j" >> "$U/$rel"
        done
      done
      bench report "$U" --out "$OUT/report-plain" > "$OUT/report-plain.log" 2>&1
      say "report plain rc=$?"
      bench report "$U" --out "$OUT/report-manifest" --manifest "$ROOT/evaluation/campaign.yaml" \
        > "$OUT/report-manifest.log" 2>&1
      say "report manifest rc=$?"
      for R in report-plain report-manifest; do
        n_tex=$(find "$OUT/$R" -name '*.tex' | wc -l)
        no_banner=$(grep -L "NOT CITABLE" $(find "$OUT/$R" -name '*.tex') 2>/dev/null | wc -l)
        say "$R: $(find "$OUT/$R" -type f | wc -l) files, $n_tex .tex, $no_banner .tex without NOT CITABLE"
      done
      ;;
    *) say "unknown stage $stage"; exit 2 ;;
  esac
done
say "done"
