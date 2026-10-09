#!/usr/bin/env bash
# V-V3BITS -> V-GR-DEEP boundary on pod 1 (orchestrator, 2026-10-09), one lock hold:
# (1) H-INDCACHE's last gate: the GPU library suite on the main checkout (its conftest makes a fresh inductor dir).
# (2) campaign-v2.3 candidate gate: origin/dev/ctl-v23 worktree (library 1258a63e), PYTHONPATH to it, a fresh inductor
#     cache: the GPU library suite, then one goodreads `filter` smoke per changed path (V1 + V2 clause and bloom, interleaved;
#     SilverTorch triton bloom), seed 0, k 100, bs 16, eager + graph, into a scratch tree; capture checked by gate_smoke.py.
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash boundary-gates.sh > /scratch/v22/boundary/driver.log 2>&1 &
set -u
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/venvs/retrieve/bin/python
OUT=/scratch/v22/boundary
WT=/scratch/wt/ctl-v23
PIN="taskset -c 0-63,128-191"
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
mkdir -p "$OUT"
run() { local name=$1; shift; local t0=$(date +%s); "$@" > "$OUT/$name.log" 2>&1; local rc=$?
        echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"; tail -2 "$OUT/$name.log"; return $rc; }

echo "main checkout $(git -C /workspace/retrieve log --oneline -1) library $(git -C /workspace/retrieve rev-parse HEAD:retrieve/src/retrieve)"
run lib-main bash -c "cd /workspace/retrieve/retrieve && $PIN $PY -m pytest tests/ -q -p no:cacheprovider"
rc1=$?

got=$(git -C "$WT" rev-parse HEAD:retrieve/src/retrieve)
echo "candidate $(git -C "$WT" log --oneline -1) library $got"
[ "$got" = 1258a63e7ebd170d7270111b5206b4ea0b8ca7f9 ] || { echo "candidate tree mismatch, refusing"; exit 2; }
export PYTHONPATH=$WT/evaluation:$WT/retrieve/src TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /scratch/inductor/v23-gate-XXXX)
echo "inductor cache $TORCHINDUCTOR_CACHE_DIR"
run lib-v23 bash -c "cd $WT/retrieve && $PIN $PY -m pytest tests/ -q -p no:cacheprovider"
rc2=$?
S=$OUT/smoke
rm -rf "$S"
common=(--dataset goodreads --dim 128 --suite filter --sweep c0_genre --seed 0 --k 100 --bs 16 --out "$S" --force)
run smoke-v1v2 bash -c "cd $WT/evaluation && $PIN $PY -m bench.cli run ${common[*]} --algo linr_v1_filter_mask --algo linr_v2 --backend triton --filter-kind clause --filter-kind bloom --interleave"
rc3=$?
run smoke-st bash -c "cd $WT/evaluation && $PIN $PY -m bench.cli run ${common[*]} --algo silvertorch --backend triton --filter-kind bloom"
rc4=$?
$PY "$HERE/gate_smoke.py" "$S" | tee "$OUT/smoke-summary.txt"
rc5=${PIPESTATUS[0]}
echo "$(date -Is) gates done: lib-main=$rc1 lib-v23=$rc2 smoke-v1v2=$rc3 smoke-st=$rc4 capture=$rc5"
