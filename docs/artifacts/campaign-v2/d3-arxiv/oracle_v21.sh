#!/usr/bin/env bash
# Rebuild arXiv d128's kept-sweep oracle blobs at campaign-v2.1 (controller: v2.1 legs read v2.1 oracles; old
# blobs are moved, never deleted), for `bloomwidth-timed` and `codesign`, and compare every rebuilt blob with
# the 408b1188 blob of the same name. Run under the GPU lock: flock /scratch/gpu0.lock bash oracle_v21.sh
set -u
REPO=/scratch/wt/v-codesign-arxiv-v21
PY=/venvs/retrieve/bin/python
GT=/data/arxiv-papers/gt_d128
OLD=$GT/_408b1188
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf
cd "$REPO/evaluation"
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = f01255f106214ec0f540352d0e2a5cd90b84b6a1 ] || { echo "code_version mismatch"; exit 2; }
mkdir -p "$OLD"
for f in "$GT"/oracle_v4_{c3_nversions,c0_maincat,all4}_*.pt; do
  [ -e "$f" ] || continue
  $PY -c 'import sys,torch; sys.exit(torch.load(sys.argv[1], weights_only=False).get("code_version") != "f01255f106214ec0f540352d0e2a5cd90b84b6a1")' "$f" && continue
  mv -n "$f" "$OLD/"
done
for s in bloomwidth-timed codesign; do
  taskset -c 64-127 $PY -m bench.cli oracle --dataset arxiv --dim 128 --suite $s || exit $?
done
$PY "$(dirname "$0")/compare_oracles.py" "$GT" "$OLD"
