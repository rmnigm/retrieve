#!/usr/bin/env bash
set -u
S=n-bloom-timed-synth-d128 R=/scratch/e/out/A2-bloom-synth-rerun L=/scratch/e/out/A2-bloom-synth-rerun-logs PY=/venvs/e-run/bin/python NUM=/scratch/e/plan-picks/research/artifacts/numbers
mkdir -p $L
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/A2-bloom-synth
export PYTHONPATH=/scratch/wt/e-run/retrieve/src:/scratch/wt/e-run/retrieve:/scratch/wt/e-run/evaluation
cd /scratch/wt/e-run/evaluation
$PY $NUM/check_b2.py $R/config > $L/check_b2.log 2>&1 && tail -1 $L/check_b2.log | grep -q OK || { echo CHECK_B2-FAILED; exit 1; }
nvidia-smi -q -d CLOCK > $L/clocks-before.txt
flock /scratch/gpu0.lock $PY -m bench.cli run --config-dir $R/config --out $R --suite $S --dataset goodreads-synth --interleave --resume > $L/run-goodreads-synth.log 2>&1 || { echo RUN-FAILED; exit 1; }
nvidia-smi -q -d CLOCK > $L/clocks-after.txt
grep -h "done:" $L/run-goodreads-synth.log | tail -1
N=$(cat $R/.seedn); T=$(mktemp -d /scratch/e/upload-b2s-rerun.XXXXXX); mkdir -p $T/$S $T/logs
tail -n +$((N+1)) $R/$S/goodreads-synth-d128.jsonl > $T/$S/goodreads-synth-d128.jsonl
[ -d $R/$S/goodreads-synth-d128.perquery ] && cp -al $R/$S/goodreads-synth-d128.perquery $T/$S/
cp $L/* $T/logs/; cp /scratch/e/run_b2s_rerun.sh $T/logs/
echo "NEW-LINES $(wc -l < $T/$S/goodreads-synth-d128.jsonl)"
$PY -m bench.cli upload --results $T --path-in-repo numbers/A2-bloom-synth-rerun --verify > $L/upload.log 2>&1 || { echo UPLOAD-FAILED; exit 1; }
echo "UPLOADED A2-bloom-synth-rerun $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $L/upload.log | tail -1) $(grep -o 'round trip verified: [0-9]* files' $L/upload.log)"
rm -rf $R/_parity
echo RERUN-DONE
