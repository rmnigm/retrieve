#!/usr/bin/env bash
set -u
B=/scratch/e/selfx
up() {
  local LEG=$1 T; T=$(mktemp -d $B/upload-$LEG.XXXXXX)
  cp -al $B/out/$LEG/. $T/; mkdir -p $T/logs; cp -r $B/logs/$LEG/. $T/logs/; cp -r $B/config-a4 $T/logs/config; cp $B/run_leg.sh $B/queue-a4.sh $T/logs/
  cd /scratch/wt/selfx/evaluation && PYTHONPATH=/scratch/wt/selfx/evaluation RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf \
    /venvs/e-run/bin/python -m bench.cli upload --results $T --path-in-repo numbers/$LEG --verify > $B/logs/upload-$LEG.log 2>&1 \
    && echo "UPLOADED $LEG $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $B/logs/upload-$LEG.log | tail -1) $(grep -o 'round trip verified: [0-9]* files' $B/logs/upload-$LEG.log)" \
    || echo "UPLOAD-FAILED $LEG"
}
CFG=$B/config-a4 $B/run_leg.sh A4-bloomwidth-selfx n-bloomwidth "" arxiv && up A4-bloomwidth-selfx
echo QUEUE-A4-DONE
