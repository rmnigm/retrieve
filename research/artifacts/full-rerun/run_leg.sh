#!/usr/bin/env bash
# One FULL-RERUN leg at one seed (runbook-pod1.md, runbook-pode.md): resolved-arm check, bench check, oracle, run under the GPU
# lock, kill + retry, upload to fr/<leg>-s<seed>, manifest sha appended to $FR/manifests.tsv.
#
# usage: run_leg.sh LEG SEED MODE SUITE[,SUITE...] DATASET [DATASET...]
#   MODE   quality | timed (timed adds --interleave: the suite's comparison groups round-robin in one process)
#   env    TREE (worktree at final-freeze)  VENV (the pod's /venvs/fr)  CFG (prep_config.sh's dir)
#          FR (out root, default /scratch/fr)  LOCK (default /scratch/gpu0.lock)  MAXH (per bench run, default 12)
#          NOUPLOAD=1 runs without the upload (a leg split over several calls uploads on its last)
# Retry rules: a failed run is retried once with --resume (finished cells are skipped); a CUDA out-of-memory failure is retried
# one algo per process instead (--algo, as B2: those cells are then not interleaved with each other, which the records show);
# a run past MAXH is killed (exit 124) and not retried: its unfinished cells are reported, never quoted. A failure never stops
# the pod's queue: it lands in $OUT/logs/FAILED and the controller decides.
set -u
LEG=$1 SEED=$2 MODE=$3 SUITES=$4
shift 4
: "${TREE:?}" "${VENV:?}" "${CFG:?}"
FR=${FR:-/scratch/fr} LOCK=${LOCK:-/scratch/gpu0.lock} MAXH=${MAXH:-12}
OUT=$FR/out/$LEG-s$SEED LOG=$FR/out/$LEG-s$SEED/logs PY=$VENV/bin/python
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/fr-$LEG HF_HOME=/scratch/hf RETRIEVE_DATA_ROOT=/data
IL=""; [ "$MODE" = timed ] && IL="--interleave"
cd "$TREE/evaluation" || exit 1
say() { echo "$(date -u +%FT%TZ) $LEG-s$SEED $*" | tee -a "$LOG/driver.log"; }

"$PY" ../research/artifacts/full-rerun/check_suites.py "$CFG" "$CFG/suites.yaml" --only "$SUITES" > "$LOG/arms.log" 2>&1 \
  || { say "ARM CHECK FAILED (see arms.log): not run"; echo "arm check" >> "$LOG/FAILED"; exit 2; }
nvidia-smi -q -d CLOCK > "$LOG/clocks-before.txt" 2>&1

run() {  # suite dataset [extra args]
  local s=$1 d=$2; shift 2
  timeout "${MAXH}h" flock "$LOCK" "$PY" -m bench.cli run --config-dir "$CFG" --out "$OUT" --suite "$s" --dataset "$d" \
    --seed "$SEED" $IL --resume "$@" >> "$LOG/run-$s-$d.log" 2>&1
}
in_suite() {  # suite dataset: the merged suites.yaml has the suite and lists the dataset (the picks may leave a suite out)
  "$PY" -c "import sys, yaml; s = yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]); sys.exit(0 if s and sys.argv[3] in s['datasets'] else 1)" \
    "$CFG/suites.yaml" "$1" "$2"
}
for d in "$@"; do
  "$PY" -m bench.cli check --dataset "$d" --config-dir "$CFG" >> "$LOG/check.log" 2>&1 \
    || { say "bench check $d FAILED"; echo "check $d" >> "$LOG/FAILED"; continue; }
  for s in ${SUITES//,/ }; do
    in_suite "$s" "$d" || { say "skip $s/$d: not in the suites"; continue; }
    flock "$LOCK" "$PY" -m bench.cli oracle --dataset "$d" --suite "$s" --config-dir "$CFG" >> "$LOG/oracle.log" 2>&1 \
      || say "oracle $s/$d rc=$? (bench run builds what is missing)"
    say "run $s/$d"
    run "$s" "$d"; rc=$?
    if [ $rc -eq 124 ]; then say "KILLED at ${MAXH} h: $s/$d"; echo "timeout $s $d" >> "$LOG/FAILED"; continue; fi
    if [ $rc -ne 0 ] && grep -q "OutOfMemoryError\|CUDA out of memory" "$LOG/run-$s-$d.log"; then
      say "OOM in $s/$d: one algo per process"
      for a in linr_v1_filter_mask linr_v2 linr_v3 silvertorch postfilter; do run "$s" "$d" --algo "$a"; done
      run "$s" "$d"; rc=$?
    elif [ $rc -ne 0 ]; then
      say "rc=$rc in $s/$d: retry once with --resume"
      run "$s" "$d"; rc=$?
    fi
    [ $rc -eq 0 ] || { say "FAILED $s/$d rc=$rc"; echo "run $s $d rc=$rc" >> "$LOG/FAILED"; }
  done
done
nvidia-smi -q -d CLOCK > "$LOG/clocks-after.txt" 2>&1
[ "${NOUPLOAD:-0}" = 1 ] && { say "done (no upload)"; exit 0; }
rm -rf "$OUT/_parity"
"$PY" -m bench.cli upload --results "$OUT" --path-in-repo "fr/$LEG-s$SEED" --verify > "$LOG/upload.log" 2>&1 \
  || { say "UPLOAD FAILED"; echo "upload" >> "$LOG/FAILED"; exit 4; }
SHA=$(grep -o 'MANIFEST.json sha256 [0-9a-f]*' "$LOG/upload.log" | awk '{print $3}')
printf '%s\t%s\t%s\t%s\n' "fr/$LEG-s$SEED" "$SHA" "$(hostname)" "$(date -u +%FT%TZ)" >> "$FR/manifests.tsv"
say "uploaded fr/$LEG-s$SEED $SHA"
[ -s "$LOG/FAILED" ] && { say "uploaded with failures (see FAILED): not marked done"; exit 5; }
exit 0
