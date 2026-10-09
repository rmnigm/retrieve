#!/usr/bin/env bash
# campaign-v2.5 bundle gate on the combined tree (dev/c5-ours = ST-LANE + V2-FILL + C5-OURS + staging), against
# campaign-v2.4 (f482b30, library d67d6263; package retrieve_v24, worktree /scratch/wt/v24):
#   ST-LANE  scorer SASS (none / exact identical), bit-exact + keep-rule timing at D 128 / 192 / 768   (st-lane/lane_gate.py)
#   V2-FILL  SPLIT SASS identical, bit-exact + keep-rule timing at 0.8 M d128, 3 M d64 / d128 / d192, 10 M d768 (v2-fill/)
#   C5-OURS  layer default-forward SASS against the tree without the option, and the layer gates with before = v2.4
#   the library suite on a fresh inductor dir (parity + compile files included).
# Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
touch /scratch/gpu0.st-dloop-phase
trap 'rm -f /scratch/gpu0.st-dloop-phase /scratch/gpu0.st-dloop-wants' EXIT
W=/scratch/wt/c5-ours A=$W/docs/artifacts/campaign-v2 S=/scratch/v25-bundle
mkdir -p $S
export CUDA_VISIBLE_DEVICES=0 PY=/venvs/retrieve/bin/python
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
for t in v24 c5-ours; do
  TRITON_CACHE_DIR=$(mktemp -d -p $S) PYTHONPATH=$W/retrieve gpu $PY $A/st-skip128/sass_ungated.py \
    --src /scratch/wt/$t/retrieve/src > $OUT/sass_scorers_$t.txt 2> $OUT/sass_scorers_$t.err
  TRITON_CACHE_DIR=$(mktemp -d -p $S) PYTHONPATH= gpu $PY $A/v2-fill/sass_fmkt.py \
    --src /scratch/wt/$t/retrieve/src > $OUT/sass_fmkt_$t.txt 2> $OUT/sass_fmkt_$t.err
done
for t in v2-fill c5-ours; do
  TRITON_CACHE_DIR=$(mktemp -d -p $S) PYTHONPATH= gpu $PY $A/c5-ours/sass_layer.py \
    --src /scratch/wt/$t/retrieve/src > $OUT/sass_layer_$t.txt 2> $OUT/sass_layer_$t.err
done
export PYTHONPATH=/scratch/st-lane/pkgs:/scratch/st-dloop/v21:$W/retrieve/src:$W/retrieve:$A/st-dloop
for d in 128 192 768; do
  gpu $PY $A/st-lane/lane_gate.py exact $d $OUT/lane_exact_d$d.json > $OUT/lane_exact_d$d.log 2>&1
  gpu $PY $A/st-lane/lane_gate.py time $d $OUT/lane_time_d$d.json > $OUT/lane_time_d$d.log 2>&1
done
export PYTHONPATH=/scratch/v2-fill/pkgs:$W/retrieve/src:$W/retrieve:$A/v2-fill
for nd in "800000 128" "3000000 128" "3000000 64" "3000000 192" "10000000 768"; do
  set -- $nd
  gpu $PY $A/v2-fill/fill_gate.py exact $1 $2 $OUT/fill_exact_n$1_d$2.json > $OUT/fill_exact_n$1_d$2.log 2>&1
  gpu $PY $A/v2-fill/fill_gate.py time $1 $2 $OUT/fill_time_n$1_d$2.json --pairs 12 > $OUT/fill_time_n$1_d$2.log 2>&1
done
export PYTHONPATH=/scratch/v2-fill/pkgs:$W/retrieve/src:$W/retrieve C5_BEFORE=retrieve_v24
for nd in "200000 128" "2000000 128" "1000000 768"; do
  set -- $nd
  gpu $PY $A/c5-ours/c5_gate.py $1 $2 $OUT/c5_exact_n$1_d$2.json > $OUT/c5_exact_n$1_d$2.log 2>&1
done
unset C5_BEFORE
cd $W/retrieve
export PYTHONPATH=$W/retrieve/src:$W/retrieve TORCHINDUCTOR_CACHE_DIR=$(mktemp -d -p $S)
gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1
echo "$(date -Is) phase done"
