#!/usr/bin/env bash
# ST-TOPK 30 M gate (gates v2.11): dev/st-topk e7e4dec (library 1d390792) vs campaign-v2.10 (a3bec4a5): laion30m d256 bloom partial, n_lists 16384,
# n_probe 128 x bs {1, 16, 64}, c0_domain + tags4, eager + graph, k 100, seed 0, pod d GPU 1; 3 rounds A B / B A / A B, one process per tree per round,
# each tree its own worktree / venv / inductor dir. Tree A also carries Meta -O3 at score_path fp16 + int32 (one interleave group by [backend,
# score_path]) for the bonus ST-TOPK / Meta row. Then score_dump.py per tree (ids + scores, torch.equal). Hub artifacts/st-topk-30m.
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v210/st-topk-30m.driver.log 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=/scratch/campaign-v210/laion30m-st-topk-30m
declare -A REPO=([A]=/scratch/wt/v210-tag [B]=/scratch/wt/st-topk-e7e4dec)
declare -A PY=([A]=/venvs/d-run-v210/bin/python [B]=/venvs/d-run-sttopk/bin/python)
declare -A CV=([A]=a3bec4a50c1a3bb06dd9e99ab638867a42b29e29 [B]=1d390792)
declare -A BACK=([A]="triton official" [B]="triton")
mkdir -p "$R/logs"
echo "pod-d st-topk-30m (cross-tree v2.10 / st-topk e7e4dec) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 1000 >> "$R/logs/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
ARM_T='    - {algo: silvertorch, backends: [triton], build: {bloom_path: [partial], n_lists: [16384]}, query: {n_probe: [128]}}\n'
ARM_O='    - {algo: silvertorch, backends: [official], build: {bloom_path: [partial], n_lists: [16384], score_path: [fp16, int32]}, query: {n_probe: [128]}}\n'
for s in A B; do
  cfg=/scratch/v210/stt-config-$s
  rm -rf "$cfg" && cp -r "${REPO[$s]}/evaluation/config" "$cfg"
  arms=$ARM_T; [ $s = A ] && arms=$ARM_T$ARM_O
  printf "stt-laion30m:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [1, 16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n  interleave: [{by: [backend, score_path]}]\n  arms:\n$arms" >> "$cfg/suites.yaml"
  got=$(cd "${REPO[$s]}/evaluation" && ${PY[$s]} -m bench.cli env | ${PY[$s]} -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
  echo "$(date -Is) tree $s: $got"
  case "$got" in "${CV[$s]}"*" False") ;; *) echo "$(date -Is) tree $s code_version mismatch, refusing"; exit 2 ;; esac
done
for round in 1 2 3; do
  order="A B"; [ $round -eq 2 ] && order="B A"
  echo "$(date -Is) round $round order $order"
  for s in $order; do
    args=(); for b in ${BACK[$s]}; do args+=(--backend $b); done
    t0=$(date +%s)
    ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/stt-$s \
        taskset -c 96-127 ${PY[$s]} -m bench.cli run --config-dir /scratch/v210/stt-config-$s --dataset laion30m --dim 256 --suite stt-laion30m \
        --algo silvertorch "${args[@]}" --interleave --force --out "$R/r$round-$s" ) >> "$R/logs/r$round-$s.log" 2>&1 < /dev/null
    rc=$?
    rm -rf "$R/r$round-$s/_parity"
    echo "$(date -Is) round $round tree $s rc=$rc s=$(( $(date +%s) - t0 ))"
    [ $rc -eq 0 ] || exit $rc
  done
done
for s in A B; do
  ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/stt-$s taskset -c 96-127 \
      ${PY[$s]} "$HERE/score_dump.py" dump /scratch/v210/stt-config-$s "$R/scores-$s.pt" ) >> "$R/logs/scores-$s.log" 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) score dump $s rc=$rc"; [ $rc -eq 0 ] || exit $rc
done
( cd /scratch/wt/v210-tag/evaluation && /venvs/d-run-v210/bin/python "$HERE/score_dump.py" compare "$R/scores-A.pt" "$R/scores-B.pt" ) | tee "$R/scores-compare.txt"
kill $SMI 2>/dev/null; sleep 2
cd /scratch/wt/v210-tag/evaluation && /venvs/d-run-v210/bin/python -m bench.cli upload --results "$R" --path-in-repo artifacts/st-topk-30m --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
