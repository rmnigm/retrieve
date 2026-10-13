#!/usr/bin/env bash
# run_leg.sh LEG SUITE "EXTRA" DS...  -> check, oracle, run --skip-perf (no interleave) per dataset under the GPU lock
set -u
LEG=$1 SUITE=$2 EXTRA=$3; shift 3
B=/scratch/e/selfx CFG=${CFG:-/scratch/e/selfx/config} OUT=$B/out/$LEG L=$B/logs/$LEG LOCK=/scratch/gpu0.lock PY=/venvs/e-run/bin/python
mkdir -p $OUT $L
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/selfx-fork RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf PYTHONPATH=/scratch/wt/selfx/evaluation
cd /scratch/wt/selfx/evaluation
for ds in "$@"; do
  echo "$LEG $SUITE $ds check $(date -u +%T)"; $PY -m bench.cli check --dataset $ds --config-dir $CFG > $L/check-$SUITE-$ds.log 2>&1 || { echo "CHECK-FAILED $LEG $ds"; exit 1; }
  echo "$LEG $SUITE $ds oracle $(date -u +%T)"; flock $LOCK $PY -m bench.cli oracle --dataset $ds --suite $SUITE --dim 128 --config-dir $CFG > $L/oracle-$SUITE-$ds.log 2>&1 || { echo "ORACLE-FAILED $LEG $ds"; exit 1; }
  echo "$LEG $SUITE $ds run $(date -u +%T)"; flock $LOCK $PY -m bench.cli run --config-dir $CFG --out $OUT --suite $SUITE --dataset $ds --dim 128 --skip-perf --resume $EXTRA > $L/run-$SUITE-$ds.log 2>&1 || { echo "RUN-FAILED $LEG $ds"; exit 1; }
  grep -h "done:" $L/run-$SUITE-$ds.log | tail -1
done
rm -rf $OUT/_parity
echo "LEG-DONE $LEG $SUITE $(date -u +%T)"
