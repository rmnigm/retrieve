#!/bin/bash
# A5-yambda on one GPU (CUDA_VISIBLE_DEVICES=0, /scratch/gpu0.lock): check, the a5-yambda suite, the oracle recall of its
# unfiltered cells (oracle_none.py; `bench oracle` builds filter sweeps only), corr.py, one upload to numbers/A5-yambda.
#   run_a5.sh <tree> <venv>
set -u
TREE=$1 VENV=$2 HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/scratch/yambda/out L=/scratch/yambda/logs O=/scratch/yambda/oracle C=/scratch/yambda/corr UP=numbers/A5-yambda
mkdir -p $OUT/config $L
say(){ echo "$(date -Is) A5-yambda $*" | tee -a $L/driver.log; }
for f in "$TREE"/evaluation/config/*.yaml; do [ "$(basename "$f")" = suites.yaml ] || cp "$f" $OUT/config/; done
cp $HERE/suites.yaml $OUT/config/suites.yaml
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/yambda RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
cd $TREE/evaluation
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
$VENV/bin/python -m bench.cli check --dataset yambda-500m --config-dir $OUT/config > $L/check.log 2>&1 || { say "check failed"; exit 1; }
t0=$(date +%s)
flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli run --config-dir $OUT/config --out $OUT --suite a5-yambda --dataset yambda-500m --resume > $L/run.log 2>&1
rc=$?; say "run rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc = 0 ] || exit 1
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
T=$(mktemp -d /scratch/yambda/upload.XXXXXX); mkdir -p $T/a5-yambda $T/logs
cp -al $OUT/a5-yambda/* $T/a5-yambda/
$VENV/bin/python -c "import sys, pathlib; from bench import records; records.aggregate(pathlib.Path(sys.argv[1]))" $T \
  || { say "aggregate failed"; exit 1; }
flock /scratch/gpu0.lock $VENV/bin/python $HERE/oracle_none.py $OUT/config $OUT $T/results.parquet $O > $L/oracle_none.log 2>&1 \
  || { say "oracle_none failed"; exit 1; }
$VENV/bin/python $HERE/corr.py $O/results_oracle.parquet $C > $L/corr.log 2>&1 || { say "corr failed"; exit 1; }
cp -a $O $T/oracle; cp -a $C $T/corr; cp $L/*.log $L/clocks-*.txt $T/logs/
$VENV/bin/python -m bench.cli upload --results $T --path-in-repo $UP --verify > $L/upload.log 2>&1 || { say "upload failed"; exit 1; }
say "uploaded $UP $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log)"
