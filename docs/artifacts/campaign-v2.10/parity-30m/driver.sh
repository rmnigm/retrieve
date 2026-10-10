#!/usr/bin/env bash
# Parity at 30 M (controller 2026-10-12): official int32 vs our triton on laion30m d256 bloom c0_domain, n_probe 128, at bloom_path partial and full,
# campaign-v2.10 (a3bec4a5) from the tag worktree + its -O3 venv, pod d GPU 1, quality only, one 30 M module per process (diagnose.py run / items),
# CPU diff + report. No library change. Hub artifacts/parity-30m.
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v210/parity-30m.driver.log 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=/scratch/campaign-v210/parity-30m
export PARITY_CFG=/scratch/v210/parity-config
PY=/venvs/d-run-v210/bin/python
rm -rf "$R" && mkdir -p "$R"
echo "pod-d parity-30m (diagnosis, quality only) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
rm -rf "$PARITY_CFG" && cp -r /scratch/wt/v210-tag/evaluation/config "$PARITY_CFG"
printf 'parity30m:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [16]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain]\n  arms:\n    - {algo: silvertorch, backends: [triton], build: {bloom_path: [partial, full], n_lists: [16384]}, query: {n_probe: [128]}}\n    - {algo: silvertorch, backends: [official], build: {bloom_path: [partial, full], n_lists: [16384], score_path: [int32]}, query: {n_probe: [128]}}\n' >> "$PARITY_CFG/suites.yaml"
cd /scratch/wt/v210-tag/evaluation
got=$($PY -m bench.cli env | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
echo "$(date -Is) tree: $got"
[ "$got" = "a3bec4a50c1a3bb06dd9e99ab638867a42b29e29 False" ] || { echo "code_version mismatch, refusing"; exit 2; }
gpu() { CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/parity-30m taskset -c 96-127 "$@"; }
step() { local n=$1; shift; "$@" >> "$R/$n.log" 2>&1 < /dev/null; local rc=$?; echo "$(date -Is) $n rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
for k in triton-partial official-partial triton-full official-full; do step "run-$k" gpu $PY "$HERE/diagnose.py" run $k "$R/$k.pt"; done
step diff $PY "$HERE/diagnose.py" diff "$R" "$R/diff.json"
for k in triton-partial official-partial; do step "items-$k" gpu $PY "$HERE/diagnose.py" items $k "$R/diff.json" "$R/items-$k.json"; done
step report $PY "$HERE/diagnose.py" report "$R" "$R/report.json"
cp "$HERE/diagnose.py" "$R/"; rm -f "$R"/*.pt.tmp
$PY -m bench.cli upload --results "$R" --path-in-repo artifacts/parity-30m --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
