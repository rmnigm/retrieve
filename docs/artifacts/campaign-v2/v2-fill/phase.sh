#!/usr/bin/env bash
# V2-FILL GPU phase on pod b (GPU 0): SASS identity of the SPLIT body (D 768 / 1024) against campaign-v2.4, the fmkt
# bit-exact gate and the keep-rule timing against campaign-v2.4 at 0.8 M / 3 M d128, 3 M d64 / d192, 10 M d768, then the
# LiNR compile + fmkt parity files and the library suite on a fresh inductor dir. Each job under the want-flag + flock.
# Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
# The phase flag lives exactly as long as this GPU block.
touch /scratch/gpu0.st-dloop-phase
trap 'rm -f /scratch/gpu0.st-dloop-phase /scratch/gpu0.st-dloop-wants' EXIT
W=/scratch/wt/v2-fill ART=$W/docs/artifacts/campaign-v2/v2-fill
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in v24 v2-fill; do
  TRITON_CACHE_DIR=$(mktemp -d -p /scratch/v2-fill) PYTHONPATH= gpu $PY $ART/sass_fmkt.py --src /scratch/wt/$t/retrieve/src \
    > $OUT/sass_$t.txt 2> $OUT/sass_$t.err
done
# No fixed TORCHINDUCTOR_CACHE_DIR (testing.md § Running): pytest gets a fresh one.
export PYTHONPATH=/scratch/v2-fill/pkgs:$W/retrieve/src:$W/retrieve:$ART
for nd in "800000 128" "3000000 128" "3000000 64" "3000000 192" "10000000 768"; do
  set -- $nd
  gpu $PY $ART/fill_gate.py exact $1 $2 $OUT/exact_n$1_d$2.json > $OUT/exact_n$1_d$2.log 2>&1
  gpu $PY $ART/fill_gate.py time $1 $2 $OUT/time_n$1_d$2.json --pairs 12 > $OUT/time_n$1_d$2.log 2>&1
done
cd $W/retrieve
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p /scratch/v2-fill)
gpu $PY -m pytest tests/compile/test_linr_compile.py tests/parity/test_fused_masked_knn_topk.py -q -p no:cacheprovider > $OUT/compile_parity.log 2>&1
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p /scratch/v2-fill)
gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1
echo "$(date -Is) phase done"
