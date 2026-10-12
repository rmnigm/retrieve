#!/usr/bin/env bash
# ST-LANE GPU phase on pod b (GPU 0): SASS hashes of the scorers against campaign-v2.4 (none / exact must match; bloom
# changes by design), the bit-exact gate and the keep-rule timing at D 128 / 192 / 768, the library suite (fresh inductor
# dir per run via the tests' conftest; no fixed TORCHINDUCTOR_CACHE_DIR). Want-flag + flock per job. Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
W=/scratch/wt/st-lane ART=$W/docs/artifacts/campaign-v2/st-lane
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in /scratch/wt/v24 $W; do
  TRITON_CACHE_DIR=$(mktemp -d -p /scratch/st-lane) PYTHONPATH=$W/retrieve gpu $PY \
    $W/docs/artifacts/campaign-v2/st-skip128/sass_ungated.py --src $t/retrieve/src \
    > $OUT/sass_$(basename $t).txt 2> $OUT/sass_$(basename $t).err
done
export PYTHONPATH=/scratch/st-lane/pkgs:/scratch/st-dloop/v21:$W/retrieve/src:$W/retrieve:$W/docs/artifacts/campaign-v2/st-dloop
for d in 128 192 768; do gpu $PY $ART/lane_gate.py exact $d $OUT/exact_d$d.json > $OUT/exact_d$d.log 2>&1; done
for d in 128 192 768; do gpu $PY $ART/lane_gate.py time $d $OUT/time_d$d.json > $OUT/time_d$d.log 2>&1; done
(cd $W/retrieve && PYTHONPATH=$W/retrieve/src:$W/retrieve gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/suite.log 2>&1)
echo "$(date -Is) phase done"
