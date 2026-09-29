#!/usr/bin/env bash
set -euo pipefail
while kill -0 85640 2>/dev/null; do sleep 20; done
sleep 5
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "GPU still busy: $busy procs"; nvidia-smi --query-compute-apps=pid,process_name --format=csv; exit 3; }
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader
cd /scratch/wt/cleanup/evaluation
export UV_PROJECT_ENVIRONMENT=/venvs/wt-cleanup
B=${BEFORE:?git archive dev/hstu evaluation/training, extracted}/evaluation
mkdir -p ${OUT:-.}
PYTHONPATH=$B uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py e1c ${OUT:-.}/before-e1c-1.json
PYTHONPATH=$B uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py e1c ${OUT:-.}/before-e1c-2.json
uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py e1c-defaults ${OUT:-.}/after-e1c.json
PYTHONPATH=$B uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py gbce ${OUT:-.}/before-gbce-1.json
PYTHONPATH=$B uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py gbce ${OUT:-.}/before-gbce-2.json
uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py gbce ${OUT:-.}/after-gbce.json
PYTHONPATH=$B uv run python ../docs/artifacts/seqrec-encoder/gate-a/gate_a.py ${OUT:-.}/gate-a-before.json
uv run python ../docs/artifacts/seqrec-encoder/gate-a/gate_a.py ${OUT:-.}/gate-a-after.json
echo ALL-DONE
