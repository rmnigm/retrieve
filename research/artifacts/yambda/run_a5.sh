#!/bin/bash
# A5-yambda on pod 1 GPU 0: check, oracle and the a5-yambda suite under the GPU lock, upload to numbers/A5-yambda, then corr.py.
#   run_a5.sh <tree> <venv>
set -u
TREE=$1 VENV=$2 HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/scratch/yambda/out L=/scratch/yambda/logs UP=numbers/A5-yambda
mkdir -p $OUT/config $L
say(){ echo "$(date -Is) A5-yambda $*" | tee -a $L/driver.log; }
for f in "$TREE"/evaluation/config/*.yaml; do [ "$(basename "$f")" = suites.yaml ] || cp "$f" $OUT/config/; done
cp $HERE/suites.yaml $OUT/config/suites.yaml
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/yambda RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
cd $TREE/evaluation
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
$VENV/bin/python -m bench.cli check --dataset yambda-500m --config-dir $OUT/config > $L/check.log 2>&1 || { say "check failed"; exit 1; }
t0=$(date +%s)
flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli oracle --dataset yambda-500m --suite a5-yambda --config-dir $OUT/config > $L/oracle.log 2>&1 \
  || { say "oracle failed rc=$?"; exit 1; }
say "oracle s=$(( $(date +%s) - t0 ))"; t0=$(date +%s)
flock /scratch/gpu0.lock $VENV/bin/python -m bench.cli run --config-dir $OUT/config --out $OUT --suite a5-yambda --dataset yambda-500m --resume > $L/run.log 2>&1
rc=$?; say "run rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc = 0 ] || exit 1
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
T=$(mktemp -d /scratch/yambda/upload.XXXXXX); mkdir -p $T/a5-yambda $T/logs
cp -al $OUT/a5-yambda/* $T/a5-yambda/; cp $L/*.log $L/clocks-*.txt $T/logs/
$VENV/bin/python -m bench.cli upload --results $T --path-in-repo $UP --verify > $L/upload.log 2>&1 || { say "upload failed"; exit 1; }
say "uploaded $UP $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log)"
P=$T/results.parquet
$VENV/bin/python $HERE/corr.py $P /scratch/yambda/corr > $L/corr.log 2>&1
say "corr rc=$? parquet=$P"
