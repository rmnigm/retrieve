#!/usr/bin/env bash
# ST-SKIP128 GPU phase on pod b (GPU 0): ungated SASS identity against the v2.3 candidate (f537c91), the bit-exact gate at
# D 128 / 192 / 768, the keep-rule timing at D 128 / 192 / 768. Each job under the want-flag + flock; no fixed inductor
# cache (testing.md § Running). Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
W=/scratch/wt/st-skip128 ART=$W/docs/artifacts/campaign-v2/st-skip128
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in v2-highp st-skip128; do
  TRITON_CACHE_DIR=$(mktemp -d -p /scratch/st-skip128) PYTHONPATH=$W/retrieve gpu $PY $ART/sass_ungated.py \
    --src /scratch/wt/$t/retrieve/src > $OUT/sass_$t.txt 2> $OUT/sass_$t.err
done
export PYTHONPATH=/scratch/st-skip128/pkgs:/scratch/st-dloop/v21:$W/retrieve/src:$W/retrieve:$W/docs/artifacts/campaign-v2/st-dloop
for d in 128 192 768; do gpu $PY $ART/skip_gate.py exact $d $OUT/exact_d$d.json > $OUT/exact_d$d.log 2>&1; done
for d in 128 192 768; do gpu $PY $ART/skip_gate.py time $d $OUT/time_d$d.json > $OUT/time_d$d.log 2>&1; done
echo "$(date -Is) phase done"
