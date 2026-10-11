#!/usr/bin/env bash
# C5-OURS GPU phase on pod b (GPU 0): SASS of the layer's default triton forward against the tree before the option
# (dev/v2-fill 394d3e9), the layer bit-exact gates (c5_gate.py) at 200 k d128 (with the torch arm), 2 M d128 and 1 M d768,
# then the library suite on a fresh inductor dir. Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
touch /scratch/gpu0.st-dloop-phase
trap 'rm -f /scratch/gpu0.st-dloop-phase /scratch/gpu0.st-dloop-wants' EXIT
W=/scratch/wt/c5-ours ART=$W/docs/artifacts/campaign-v2/c5-ours
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in v2-fill c5-ours; do
  TRITON_CACHE_DIR=$(mktemp -d -p /scratch/c5-ours) PYTHONPATH= gpu $PY $ART/sass_layer.py --src /scratch/wt/$t/retrieve/src \
    > $OUT/sass_$t.txt 2> $OUT/sass_$t.err
done
export PYTHONPATH=/scratch/c5-ours/pkgs:$W/retrieve/src:$W/retrieve
for nd in "200000 128" "2000000 128" "1000000 768"; do
  set -- $nd
  gpu $PY $ART/c5_gate.py $1 $2 $OUT/exact_n$1_d$2.json > $OUT/exact_n$1_d$2.log 2>&1
done
cd $W/retrieve
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p /scratch/c5-ours)
gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1
echo "$(date -Is) phase done"
