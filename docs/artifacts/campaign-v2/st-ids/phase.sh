#!/usr/bin/env bash
# ST-IDS GPU phase on pod b (GPU 0): probe-scorer parity files, the bit-exact gate at D 128 / 192 / 768, the id-epilogue
# timing, the library suite. Each job under the want-flag + flock. Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
W=/scratch/wt/st-ids ART=$W/docs/artifacts/campaign-v2/st-ids
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-ids PY=/venvs/retrieve/bin/python
export PYTHONPATH=/scratch/st-ids/pkgs:/scratch/st-dloop/v21:$W/retrieve/src:$W/retrieve:$W/docs/artifacts/campaign-v2/st-dloop
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
cd $W/retrieve
gpu $PY -m pytest tests/parity/test_codesigned_probe_score.py tests/parity/test_codesigned_probe_score_exact.py -q -p no:cacheprovider > $OUT/parity.log 2>&1
for d in 128 192 768; do gpu $PY $ART/ids_gate.py exact $d $OUT/exact_d$d.json > $OUT/exact_d$d.log 2>&1; done
gpu $PY $ART/ids_gate.py time 128 $OUT/time_d128.json > $OUT/time_d128.log 2>&1
gpu $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1
echo "$(date -Is) phase done"
