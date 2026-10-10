#!/usr/bin/env bash
# st-dloop's 30 M profile ask (.chains/control/2026-10-13-190000000-st-dloop-st-xq-sharing.md): laion30m d256 bloom partial, c0_domain + tags4,
# n_lists 16384, n_probe 128, bs 16 / 64, at e16512f5 (dev/v1-topk 8c9b336 tree): bounds.py (true vs bloom pass rate, sparse share, rarest-bit
# bounds) and st-topk/prof.py with official int32 (kernel splits, ours + Meta's), pod d GPU 1. Hub artifacts/st-xq-30m-profile.
#   flock /scratch/gpu1.lock bash run.sh
set -u
[ -e /scratch/v211/st-xq.skip ] && { echo "$(date -Is) skipped (controller: laion runs this profile on GPU 0)"; exit 0; }
W=/scratch/wt/v1-topk-8c9b336; PY=/venvs/d-run-v1topk/bin/python
R=/scratch/campaign-v211/st-xq-30m-profile; mkdir -p "$R"
echo "pod-d st-xq-30m-profile on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
cd "$W/evaluation"
got=$($PY -m bench.cli env | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"][:8], d["dirty"])')
[ "$got" = "e16512f5 False" ] || { echo "code_version $got, refusing"; exit 2; }
g() { CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/freeze-B PYTHONPATH=.:../retrieve/src taskset -c 96-127 "$@"; }
g $PY ../docs/artifacts/campaign-v2.9/st-wide-2/bounds.py laion30m 256 c0_domain,tags4 16384 "$R/bounds.json" > "$R/bounds.log" 2>&1 < /dev/null
echo "$(date -Is) bounds rc=$?"
for sw in c0_domain tags4; do
  g $PY ../docs/artifacts/campaign-v2.10/st-topk/prof.py laion30m 256 bloom $sw 16384 bloom 128 16,64 "$R/prof-$sw.json" official > "$R/prof-$sw.log" 2>&1 < /dev/null
  echo "$(date -Is) prof $sw rc=$?"
done
cp "$0" "$R/"
$PY -m bench.cli upload --results "$R" --path-in-repo artifacts/st-xq-30m-profile --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) profile done"
rm -f /scratch/gpu1-holder
