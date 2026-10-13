#!/usr/bin/env bash
set -u
B=/scratch/e/selfx R=$B/run_leg.sh
up() {  # up LEG
  local LEG=$1 T; T=$(mktemp -d $B/upload-$LEG.XXXXXX)
  cp -al $B/out/$LEG/. $T/; mkdir -p $T/logs; cp -r $B/logs/$LEG/. $T/logs/; cp $R $B/queue.sh $B/check_arms.py $T/logs/
  cd /scratch/wt/selfx/evaluation && PYTHONPATH=/scratch/wt/selfx/evaluation RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf \
    /venvs/e-run/bin/python -m bench.cli upload --results $T --path-in-repo numbers/$LEG --verify > $B/logs/upload-$LEG.log 2>&1 \
    && echo "UPLOADED $LEG $(grep -o 'MANIFEST.json sha256 [0-9a-f]*' $B/logs/upload-$LEG.log | tail -1) $(grep -o 'round trip verified: [0-9]* files' $B/logs/upload-$LEG.log)" \
    || echo "UPLOAD-FAILED $LEG"
}
$R A1-bloom-selfx n-bloom-curve-d128 "" arxiv && $R A1-bloom-selfx n-bloom-curve-synth-d128 "" arxiv-synth && up A1-bloom-selfx
$R A3-st-k2048-selfx n-st-k2048-d128 "" arxiv && up A3-st-k2048-selfx
$R L2-v3pool-selfx n-v3pool "" arxiv && up L2-v3pool-selfx
until ls /data/arxiv-papers/synth_corr.json >/dev/null 2>&1 && ! pgrep -f "synth-filter --dataset arxiv --correlated" >/dev/null; do sleep 20; done
echo "corr attrs ready $(date -u +%T)"
$R A1-small-selfx n-curve-q "--filter-kind clause" arxiv arxiv-synth arxiv-corr-synth && up A1-small-selfx
echo QUEUE-DONE
