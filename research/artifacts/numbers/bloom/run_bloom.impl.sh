#!/bin/bash
# One NUMBERS leg on pod 1 GPU 0: prep config, bench check + oracle per dataset, bench run per dataset under the GPU lock,
# (bloom-runner copy: pauses below 3 GB free on / before each dataset; deletes its own finished _parity/*.npz while running (never dirs or .tmp: a dir removal raced the rename, 3 laion3m cells failed); never uploaded)
# then one corrected upload of every dataset's files (cp -al into $T/<suite>) with an n_records check.
#   run_leg.sh <tree> <venv> <leg> <suite> <upload> <ds...>      env: LIB (expected code_version, required), EXTRA (more bench run flags), INTERLEAVE (default --interleave; "" for none), FORKTREE (put that tree first on PYTHONPATH), INDUCTOR_TAG, NOUPLOAD (skip the upload: a later call to the same leg uploads all its suites),
#   NUM (plan dir; default the dev/numbers worktree), SEED (an earlier out dir: its stable ok records are copied in first so --resume re-runs only the rest; only the new lines are uploaded)
set -u
: "${LIB:?LIB = expected library tree}"
TREE=$1 VENV=$2 LEG=$3 SUITE=$4 UP=$5; shift 5
NUM=${NUM:-/scratch/wt/numbers/research/artifacts/numbers}
B=/scratch/numbers/p1 OUT=/scratch/numbers/p1/out/$LEG L=/scratch/numbers/p1/logs/$LEG
mkdir -p $OUT $L
say(){ echo "$(date -Is) $LEG $*" | tee -a $B/logs/driver.log; }
[ -f $B/hold-$LEG ] && { say "held ($B/hold-$LEG): skipped, to be started by hand"; exit 0; }
[ -f $B/drop-$LEG ] && { say "dropped ($B/drop-$LEG): moved off p1, not run here"; exit 0; }
bash $NUM/prep_config.sh $TREE $OUT/config > $L/prep.log 2>&1 || { say "prep_config failed"; exit 1; }
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/numbers/inductor-p1-0${INDUCTOR_TAG:-} RETRIEVE_DATA_ROOT=/data
[ -n "${FORKTREE:-}" ] && export PYTHONPATH="$FORKTREE/retrieve/src:$FORKTREE/retrieve:$FORKTREE/evaluation"
cd $TREE/evaluation
for ds in "$@"; do
  if [ -n "${SEED:-}" ]; then mkdir -p $OUT/$SUITE; for f in $SEED/$SUITE/$ds-d*.jsonl; do case $f in *.samples.jsonl) continue;; esac
    g=$OUT/$SUITE/$(basename $f); [ -e $g.seedn ] && continue
    python3 -c "import json,sys; [sys.stdout.write(l) for l in open(sys.argv[1]) if (r:=json.loads(l))['status']=='ok' and not r['unstable']]" $f > $g
    grep -c . $g > $g.seedn; say "seeded $(basename $g): $(cat $g.seedn) stable of $(grep -c . $f)"; done; fi
  while [ $(df --output=avail -B1G / | tail -1) -lt 3 ]; do say "paused: / below 3 GB free before $ds"; sleep 300; done
  ( while sleep 30; do find $OUT/_parity -name "*.npz" -type f -delete 2>/dev/null; done ) & REAP=$!
  $VENV/bin/python -m bench.cli check --dataset $ds --config-dir $OUT/config > $L/check-$ds.log 2>&1 || { say "check $ds failed"; exit 1; }
  flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli oracle --dataset $ds --suite $SUITE --config-dir $OUT/config > $L/oracle-$ds.log 2>&1 \
    || { say "oracle $ds failed rc=$?"; exit 1; }
  t0=$(date +%s)
  flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli run --config-dir $OUT/config --out $OUT --suite $SUITE --dataset $ds ${INTERLEAVE---interleave} --resume ${EXTRA:-} > $L/run-$ds.log 2>&1
  rc=$?; kill $REAP; rm -rf $OUT/_parity; say "run $ds rc=$rc s=$(( $(date +%s) - t0 )) free=$(df --output=avail -h / | tail -1)"; [ $rc = 0 ] || exit 1
done
[ -n "${NOUPLOAD:-}" ] && { say "runs done, upload deferred (NOUPLOAD)"; exit 0; }
T=$(mktemp -d /scratch/numbers/upload.XXXXXX)
for d in $OUT/*/; do s=$(basename $d); case $s in config|_*) continue;; esac; compgen -G "$d*.json*" >/dev/null || continue; mkdir -p $T/$s; cp -al $d* $T/$s/
  for n in $T/$s/*.seedn; do [ -e "$n" ] || continue; j=${n%.seedn}; tail -n +$(( $(cat $n) + 1 )) $j > $j.new; rm $j $n; mv $j.new $j; [ -s $j ] || rm $j; done; done
NREC=$(ls $T/*/*.jsonl | grep -v "\.samples\.jsonl$" | xargs cat | grep -c .)
$VENV/bin/python -m bench.cli upload --results $T --path-in-repo $UP --verify > $L/upload.log 2>&1 || { say "upload failed"; exit 1; }
SHA=$(grep -o "MANIFEST.json sha256 [0-9a-f]*" $L/upload.log | awk '{print $3}')
$VENV/bin/python - "$UP" "$NREC" "$LIB" >> $L/upload.log 2>&1 <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
up, n, lib = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(open(hf_hub_download("pinkmeme/eval-results", f"{up}/MANIFEST.json", repo_type="dataset", force_download=True)))
files = {f["path"] for f in m["files"]}
ok = m["n_records"] == n and m["code_versions"] == [lib] and "results.parquet" in files
print(f"manifest check {'OK' if ok else 'FAILED'}: n_records {m['n_records']} (expect {n}), code_versions {m['code_versions']}, "
      f"commits {m['commits']}, status {m['status']}, unstable {m['unstable']}, results.parquet {'results.parquet' in files}")
sys.exit(0 if ok else 1)
PY
rc=$?; rm -rf $T
[ $rc = 0 ] && rm -rf $OUT/_parity   # controller-authorized: own parity spills, after a verified upload
say "uploaded $UP manifest $SHA check rc=$rc $(grep 'manifest check' $L/upload.log)"
exit $rc
