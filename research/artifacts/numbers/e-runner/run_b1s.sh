#!/usr/bin/env bash
set -u
OUT=/scratch/e/A1-bloom-synth/out L=/scratch/e/A1-bloom-synth/logs CFG=/scratch/e/A1-bloom-synth/config LOCK=/scratch/gpu0.lock PY=/venvs/e-run/bin/python
mkdir -p $OUT $L
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/A1-bloom-synth RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
cd /scratch/wt/e-run/evaluation
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
for ds in goodreads-synth arxiv-synth; do
  echo "check $ds $(date -u +%T)"; $PY -m bench.cli check --dataset $ds --config-dir $CFG > $L/check-$ds.log 2>&1 || { echo "CHECK-FAILED $ds"; exit 1; }
  echo "oracle $ds $(date -u +%T)"; flock $LOCK $PY -m bench.cli oracle --dataset $ds --suite n-bloom-curve-synth-d128 --config-dir $CFG > $L/oracle-$ds.log 2>&1 || { echo "ORACLE-FAILED $ds"; exit 1; }
  echo "run $ds $(date -u +%T)"; flock $LOCK $PY -m bench.cli run --config-dir $CFG --out $OUT --suite n-bloom-curve-synth-d128 --dataset $ds --resume > $L/run-$ds.log 2>&1 || { echo "RUN-FAILED $ds"; exit 1; }
  rm -rf $OUT/_parity
done
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
echo "RUN-DONE $(date -u +%T)"
