#!/usr/bin/env bash
# ST-DLOOP GPU phase on pod b (A100 GPU 0): bit-exact gates, real-data gate, interleaved PubMed before/after
# (two rounds, before = campaign-v2.1 tree), keep-rule timing grid, library suite. Every GPU job under the
# want-flag + flock; /scratch/gpu0.timed set around timed jobs. Usage: phase.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
NEW=/scratch/wt/st-dloop BASE=/scratch/wt/st-dloop-base V21=/scratch/st-dloop/v21 ART=$NEW/docs/artifacts/campaign-v2/st-dloop
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf PY=/venvs/retrieve/bin/python
gpu() {  # gpu <timed 0|1> <cmd...>
  local timed=$1; shift
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock bash -c '[ "$0" = 1 ] && touch /scratch/gpu0.timed; "${@}"; rc=$?; [ "$0" = 1 ] && rm -f /scratch/gpu0.timed; exit $rc' "$timed" "$@"
  local rc=$?; rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
syn() { PYTHONPATH=$V21:$NEW/retrieve/src TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-dloop gpu "$@"; }

for d in 768 192 128; do syn 0 $PY $ART/dloop_gate.py exact $d $OUT/gate_exact_d$d.json > $OUT/gate_exact_d$d.log 2>&1; done
(cd $NEW/evaluation && PYTHONPATH=$V21:$NEW/retrieve/src:$NEW/evaluation TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-dloop \
  gpu 0 $PY $ART/real_gate.py pubmed 768 $OUT/real_gate_pubmed768.json > $OUT/real_gate_pubmed768.log 2>&1)

for r in 1 2; do
  for arm in before after; do
    T=$([ $arm = before ] && echo $BASE || echo $NEW)
    (cd $T/evaluation && PYTHONPATH=$T/evaluation:$T/retrieve/src TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-dloop-$arm \
      gpu 1 taskset -c 0-95 $PY -m bench.cli run --config-dir /scratch/st-dloop/config --dataset pubmed --dim 768 \
      --suite st-dloop --out $OUT/bench-$arm-r$r --interleave > $OUT/bench-$arm-r$r.log 2>&1)
  done
done

for d in 768 192 128; do syn 1 $PY $ART/dloop_gate.py time $d $OUT/time_d$d.json > $OUT/time_d$d.log 2>&1; done
for d in 192 128; do syn 1 $PY $ART/dloop_gate.py time $d $OUT/time_forceskip_d$d.json --force-skip > $OUT/time_forceskip_d$d.log 2>&1; done

(cd $NEW/retrieve && PYTHONPATH=$NEW/retrieve/src:$NEW/retrieve TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-dloop \
  gpu 0 $PY -m pytest tests/ -q -p no:cacheprovider > $OUT/pytest_retrieve.log 2>&1)
echo "$(date -Is) phase done"
