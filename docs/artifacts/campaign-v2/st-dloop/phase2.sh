#!/usr/bin/env bash
# ST-DLOOP GPU phase 2: the real-data bit-exact gate (PubMed d768), the d768 kernel split after the change, and the d128
# kernel split of both backends (arXiv c0_maincat; the d128 code is the v2.1 code). Usage: phase2.sh <out_dir>
set -u
OUT=$1; mkdir -p "$OUT"
NEW=/scratch/wt/st-dloop V21=/scratch/st-dloop/v21 ART=$NEW/docs/artifacts/campaign-v2/st-dloop
export CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf PY=/venvs/retrieve/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/st-dloop
gpu() {
  touch /scratch/gpu0.st-dloop-wants
  flock /scratch/gpu0.lock "$@"; local rc=$?
  rm -f /scratch/gpu0.st-dloop-wants; echo "$(date -Is) rc=$rc: $*"; return $rc
}
cd $NEW/evaluation
PYTHONPATH=$V21:$NEW/retrieve/src:$NEW/evaluation gpu $PY $ART/real_gate.py pubmed 768 $OUT/real_gate_pubmed768.json > $OUT/real_gate_pubmed768.log 2>&1
PYTHONPATH=$NEW/retrieve/src:$NEW/evaluation:$NEW/docs/artifacts/kernel-opt gpu $PY $ART/real_cell.py profile pubmed 768 c0_mesh \
  $OUT/profile_after_pubmed768.json --modes bloom,none > $OUT/profile_after_pubmed768.log 2>&1
PYTHONPATH=$NEW/retrieve/src:$NEW/evaluation:$NEW/docs/artifacts/kernel-opt gpu $PY $ART/real_cell.py profile arxiv 128 c0_maincat \
  $OUT/profile_arxiv128.json --modes bloom,none > $OUT/profile_arxiv128.log 2>&1
echo "$(date -Is) phase2 done"
