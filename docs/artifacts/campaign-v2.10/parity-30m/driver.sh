#!/usr/bin/env bash
# Parity at 30 M (controller 2026-10-12): diagnose official int32 vs our triton on laion30m d256 bloom c0_domain, n_probe 128, at bloom_path partial
# and full, campaign-v2.10 (a3bec4a5) from the tag worktree + its -O3 venv, pod d GPU 1, quality only (diagnose.py). No library change.
# Hub artifacts/parity-30m.
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v210/parity-30m.driver.log 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=/scratch/campaign-v210/parity-30m
CFG=/scratch/v210/parity-config
PY=/venvs/d-run-v210/bin/python
mkdir -p "$R"
echo "pod-d parity-30m (diagnosis, quality only) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
rm -rf "$CFG" && cp -r /scratch/wt/v210-tag/evaluation/config "$CFG"
printf 'parity30m:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [16]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain]\n  arms:\n    - {algo: silvertorch, backends: [triton], build: {bloom_path: [partial, full], n_lists: [16384]}, query: {n_probe: [128]}}\n    - {algo: silvertorch, backends: [official], build: {bloom_path: [partial, full], n_lists: [16384], score_path: [int32]}, query: {n_probe: [128]}}\n' >> "$CFG/suites.yaml"
cd /scratch/wt/v210-tag/evaluation
got=$($PY -m bench.cli env | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
echo "$(date -Is) tree: $got"
[ "$got" = "a3bec4a50c1a3bb06dd9e99ab638867a42b29e29 False" ] || { echo "code_version mismatch, refusing"; exit 2; }
CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/parity-30m taskset -c 96-127 \
  $PY "$HERE/diagnose.py" "$CFG" "$R/diagnose.json" 2000 25 > "$R/diagnose.log" 2>&1 < /dev/null
rc=$?; echo "$(date -Is) diagnose rc=$rc"
cp "$HERE/diagnose.py" "$R/"
$PY -m bench.cli upload --results "$R" --path-in-repo artifacts/parity-30m --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
exit $rc
