#!/usr/bin/env bash
# Freeze-library 30 M gate (controller 2026-10-13; gates the freeze tag): dev/v1-topk 8c9b336 (library e16512f5 = the 2^24 fix + V1-TOPK) vs
# campaign-v2.11 (1d390792): the V1-BS1 gate's V1 cells plus SilverTorch triton bloom partial on laion30m (n_lists 16384, n_probe 128, bs 16 / 64, suite stg).
# (c0_domain, tags4), laion30m-synth d256 and yfcc10m-synth d192 (p 0.01 / 0.1 / 1), bs 1 / 16 / 64, eager + graph, k 100, seed 0, pod d GPU 1;
# 3 rounds A B / B A / A B, one process per tree per dataset per round, each tree its own worktree / venv / inductor dir; then score_dump.py per tree
# (ids + scores torch.equal, and the bs-1 scorer each table's register-time check chose). Hub artifacts/v1-bs1-30m.
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v211/v1-bs1-30m.driver.log 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R:-/scratch/campaign-v211/freeze-30m}
declare -A REPO=([A]=/scratch/wt/v211-tag [B]=${B_REPO:-/scratch/wt/v1-topk-8c9b336})
declare -A PY=([A]=/venvs/d-run-v211/bin/python [B]=${B_PY:-/venvs/d-run-v1topk/bin/python})
declare -A CV=([A]=1d390792 [B]=${B_CV:-e16512f5})
DS=("laion30m:256" "laion30m-synth:256" "yfcc10m-synth:192")
mkdir -p "$R/logs"
echo "pod-d freeze-30m (cross-tree v2.11 / v1-topk 8c9b336) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 1000 >> "$R/logs/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
for s in A B; do
  cfg=/scratch/v211/frz-config-$s
  rm -rf "$cfg" && cp -r "${REPO[$s]}/evaluation/config" "$cfg"
  printf 'v1g:\n  datasets: [laion30m, laion30m-synth, yfcc10m-synth]\n  dims: [192, 256]\n  filter_kinds: [clause]\n  ks: [100]\n  batch_sizes: [1, 16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n    laion30m-synth: [p001, p01, p1]\n    yfcc10m-synth: [p001, p01, p1]\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [triton]}\n' >> "$cfg/suites.yaml"
  printf 'stg:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n  arms:\n    - {algo: silvertorch, backends: [triton], build: {bloom_path: [partial], n_lists: [16384]}, query: {n_probe: [128]}}\n' >> "$cfg/suites.yaml"
  got=$(cd "${REPO[$s]}/evaluation" && ${PY[$s]} -m bench.cli env | ${PY[$s]} -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
  echo "$(date -Is) tree $s: $got"
  case "$got" in "${CV[$s]}"*" False") ;; *) echo "$(date -Is) tree $s code_version mismatch, refusing"; exit 2 ;; esac
done
for round in 1 2 3; do
  order="A B"; [ $round -eq 2 ] && order="B A"
  echo "$(date -Is) round $round order $order"
  for s in $order; do
    for dd in "${DS[@]}"; do
      ds=${dd%%:*}; dim=${dd#*:}; t0=$(date +%s)
      ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/freeze-$s \
          taskset -c 96-127 ${PY[$s]} -m bench.cli run --config-dir /scratch/v211/frz-config-$s --dataset $ds --dim $dim --suite v1g \
          --algo linr_v1_filter_mask --backend triton --force --out "$R/r$round-$s" ) >> "$R/logs/r$round-$s-$ds.log" 2>&1 < /dev/null
      rc=$?
      echo "$(date -Is) round $round tree $s $ds rc=$rc s=$(( $(date +%s) - t0 ))"
      [ $rc -eq 0 ] || exit $rc
    done
    t0=$(date +%s)
    ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/freeze-$s \
        taskset -c 96-127 ${PY[$s]} -m bench.cli run --config-dir /scratch/v211/frz-config-$s --dataset laion30m --dim 256 --suite stg \
        --algo silvertorch --backend triton --force --out "$R/r$round-$s" ) >> "$R/logs/r$round-$s-stg.log" 2>&1 < /dev/null
    rc=$?; echo "$(date -Is) round $round tree $s stg rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
  done
done
for s in A B; do
  ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/freeze-$s taskset -c 96-127 \
      ${PY[$s]} "$HERE/score_dump.py" dump /scratch/v211/frz-config-$s "$R/scores-$s.pt" ) >> "$R/logs/scores-$s.log" 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) score dump $s rc=$rc"; [ $rc -eq 0 ] || exit $rc
done
( cd "${REPO[A]}/evaluation" && ${PY[A]} "$HERE/score_dump.py" compare "$R/scores-A.pt" "$R/scores-B.pt" ) | tee "$R/scores-compare.txt"
kill $SMI 2>/dev/null; sleep 2
cd "${REPO[A]}/evaluation" && ${PY[A]} -m bench.cli upload --results "$R" --path-in-repo artifacts/freeze-30m --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
