#!/usr/bin/env bash
set -u
T=/scratch/wt/e-k1000 NUM=/scratch/e/plan-2b3f662/research/artifacts/numbers OUT=/scratch/e/out/A2-bloom-k1000-e L=/scratch/e/out/A2-bloom-k1000-e-logs PY=/venvs/e-run/bin/python
mkdir -p $L
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/A2-bloom-k1000-e
export PYTHONPATH=$T/official-fork:$T/retrieve/src:$T/retrieve:$T/evaluation
cd $T/evaluation
$PY -c "import silvertorch_fork, hashlib, glob, os; p=glob.glob(os.path.dirname(silvertorch_fork.__file__)+'/_C*.so')[0]; print(p, hashlib.sha256(open(p,'rb').read()).hexdigest())" > $L/so_sha.txt
grep -q "^$T/official-fork/silvertorch_fork/_C.*8ae8c68c056f5ad4fe22014a4fec3ac0964a967dc815e25123c80609ddc853ca$" $L/so_sha.txt || { echo "SO-SHA-FAILED"; exit 1; }
$PY $NUM/check_b2.py $OUT/config > $L/check_b2.log 2>&1 && tail -1 $L/check_b2.log | grep -q OK || { echo "CHECK_B2-FAILED"; exit 1; }
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
run(){ s=$1; shift; for ds in "$@"; do
  echo "check $s $ds $(date -u +%T)"; $PY -m bench.cli check --dataset $ds --config-dir $OUT/config > $L/check-$ds.log 2>&1 || { echo "CHECK-FAILED $ds"; return 1; }
  echo "oracle $s $ds $(date -u +%T)"; flock /scratch/gpu0.lock $PY -m bench.cli oracle --dataset $ds --suite $s --config-dir $OUT/config > $L/oracle-$ds.log 2>&1 || { echo "ORACLE-FAILED $ds"; return 1; }
  echo "run $s $ds $(date -u +%T)"; flock /scratch/gpu0.lock $PY -m bench.cli run --config-dir $OUT/config --out $OUT --suite $s --dataset $ds --interleave --resume > $L/run-$ds.log 2>&1; echo "rc=$? $ds"
  grep -h "done:" $L/run-$ds.log | tail -1
done; }
run n-bloom-timed-k1000-d128 arxiv goodreads || exit 1
run n-bloom-timed-k1000-d256 laion30m || exit 1
if grep -q "done:.*failed" $L/run-laion30m.log; then
  echo "laion30m failed cells -> one algo per process $(date -u +%T)"
  for a in linr_v1_filter_mask linr_v2 silvertorch; do
    flock /scratch/gpu0.lock $PY -m bench.cli run --config-dir $OUT/config --out $OUT --suite n-bloom-timed-k1000-d256 --dataset laion30m --interleave --resume --algo $a > $L/run-laion30m-$a.log 2>&1; echo "rc=$? laion30m $a"
    grep -h "done:" $L/run-laion30m-$a.log | tail -1
  done
fi
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
U=$(mktemp -d /scratch/e/upload-k1000.XXXXXX); mkdir -p $U/logs
for s in n-bloom-timed-k1000-d128 n-bloom-timed-k1000-d256; do mkdir -p $U/$s; cp -al $OUT/$s/. $U/$s/; done
cp $L/* $U/logs/; cp -r $OUT/config $U/logs/config; cp /scratch/e/run_k1000.sh $U/logs/
$PY -m bench.cli upload --results $U --path-in-repo numbers/A2-bloom-k1000-e --verify > $L/upload.log 2>&1 || { echo UPLOAD-FAILED; exit 1; }
echo "UPLOADED A2-bloom-k1000-e $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log | tail -1) $(grep -o 'round trip verified: [0-9]* files' $L/upload.log)"
$PY - $U <<'PY'
import glob, json, sys, collections
U = sys.argv[1]; recs = {}
for f in glob.glob(f"{U}/n-bloom-timed-k1000-*/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        recs[(f, r["algo"], json.dumps(r["params"], sort_keys=True), r["sweep"])] = r  # newest per key
c = collections.Counter((r["dataset"], r["algo"], r["status"]) for r in recs.values())
sha = {((r.get("env") or {}).get("official_fork_build") or {}).get("so_sha256") for r in recs.values() if r["backend"] == "official-fork"}
code = {(r.get("env") or {}).get("code_version") for r in recs.values()}
uns = [(r["dataset"], r["algo"], r["sweep"], r["params"]) for r in recs.values() if r.get("unstable")]
print("MANIFEST-CHECK latest records", len(recs), dict(c), "so_sha256", sha, "code", code, "unstable", len(uns))
for u in uns: print("UNSTABLE", u)
PY
[ -f $U/results.parquet ] && echo "results.parquet present"
rm -rf $OUT/_parity
echo K1000-DONE
