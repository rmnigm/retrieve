#!/usr/bin/env bash
set -u
NUM=/scratch/e/plan-picks/research/artifacts/numbers OUT=/scratch/e/out/A2-bloom-synth L=/scratch/e/out/A2-bloom-synth-logs PY=/venvs/e-run/bin/python S=n-bloom-timed-synth-d128
mkdir -p $L
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/A2-bloom-synth
export PYTHONPATH=/scratch/wt/e-run/retrieve/src:/scratch/wt/e-run/retrieve:/scratch/wt/e-run/evaluation
cd /scratch/wt/e-run/evaluation
$PY $NUM/check_b2.py $OUT/config > $L/check_b2.log 2>&1 && tail -1 $L/check_b2.log | grep -q OK || { echo "CHECK_B2-FAILED"; exit 1; }
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
for ds in goodreads-synth arxiv-synth; do
  echo "check $ds $(date -u +%T)"; $PY -m bench.cli check --dataset $ds --config-dir $OUT/config > $L/check-$ds.log 2>&1 || { echo "CHECK-FAILED $ds"; exit 1; }
  echo "oracle $ds $(date -u +%T)"; flock /scratch/gpu0.lock $PY -m bench.cli oracle --dataset $ds --suite $S --config-dir $OUT/config > $L/oracle-$ds.log 2>&1 || { echo "ORACLE-FAILED $ds"; exit 1; }
  echo "run $ds $(date -u +%T)"; flock /scratch/gpu0.lock $PY -m bench.cli run --config-dir $OUT/config --out $OUT --suite $S --dataset $ds --interleave --resume > $L/run-$ds.log 2>&1 || { echo "RUN-FAILED $ds"; exit 1; }
  grep -h "done:" $L/run-$ds.log | tail -1
done
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
T=$(mktemp -d /scratch/e/upload-b2s.XXXXXX); mkdir -p $T/$S $T/logs
cp -al $OUT/$S/. $T/$S/; cp $L/* $T/logs/; cp -r $OUT/config $T/logs/config; cp /scratch/e/run_b2s.sh $T/logs/
$PY -m bench.cli upload --results $T --path-in-repo numbers/A2-bloom-synth --verify > $L/upload.log 2>&1 || { echo "UPLOAD-FAILED"; exit 1; }
echo "UPLOADED A2-bloom-synth $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log | tail -1) $(grep -o 'round trip verified: [0-9]* files' $L/upload.log)"
$PY -c "
import polars as pl,sys; df=pl.read_parquet(sys.argv[1]+'/results.parquet')
print('MANIFEST-CHECK records', df.select(['dataset','algo','params','filter_kind','sweep']).unique().height, 'rows', df.height, 'status', df['status'].value_counts().to_dicts(), 'code', df['env_code_version'].unique().to_list(), 'unstable', int(df['unstable'].sum()))" $T
rm -rf $OUT/_parity
echo B2S-DONE
