#!/usr/bin/env bash
# ST-WIDE at 30 M, cross-tree: the 4 n_probe-128 laion30m cells (c0_domain + tags4, bs 16 + 64, bloom partial, eager, k 100, seed 0) on pod d
# GPU 1, 3 rounds; each round A = v2.8 (ours + Meta -O3 interleaved by backend, /venvs/d-run-v28) then B = dev/st-wide 36aa2f4 (ours,
# /venvs/d-run-stwide), one process per sweep, --profile; each tree its own inductor dir. Ratios per round; CI across rounds (analyze.py).
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v28/c7-st-wide.driver.log 2>&1 &
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=/scratch/campaign-v28/laion30m-c7-st-wide
declare -A REPO=([A]=/scratch/wt/v28-official [B]=/scratch/wt/st-wide-36aa2f4)
declare -A PY=([A]=/venvs/d-run-v28/bin/python [B]=/venvs/d-run-stwide/bin/python)
declare -A CV=([A]=78cfbc721f0c7ab62831832b3791e1f30dc7357c [B]=e8958bd234cf9f814d1afcb2e5eb3f95e25dbec3)
declare -A BACK=([A]="triton official" [B]="triton")
mkdir -p "$R/logs"
echo "pod-d c7-st-wide (cross-tree v2.8 / st-wide 36aa2f4) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 1000 >> "$R/logs/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
for s in A B; do
  cfg=/scratch/v28/c7sw-config-$s
  rm -rf "$cfg" && cp -r "${REPO[$s]}/evaluation/config" "$cfg"
  printf 'c7w-laion30m:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n  interleave: [{by: backend, values: [triton, official]}]\n  arms:\n    - algo: silvertorch\n      backends: [%s]\n      build: {bloom_path: [partial], n_lists: [16384]}\n      query: {n_probe: [128]}\n' \
    "$(echo ${BACK[$s]} | sed 's/ /, /')" >> "$cfg/suites.yaml"
  got=$(cd "${REPO[$s]}/evaluation" && ${PY[$s]} -m bench.cli env | ${PY[$s]} -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
  echo "$(date -Is) tree $s: $got"
  [ "$got" = "${CV[$s]} False" ] || { echo "$(date -Is) tree $s code_version mismatch, refusing"; exit 2; }
done
for round in 1 2 3; do
  for s in A B; do
    for sw in c0_domain tags4; do
      args=(); for b in ${BACK[$s]}; do args+=(--backend $b); done
      t0=$(date +%s)
      ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/c7sw-$s \
          taskset -c 96-127 ${PY[$s]} -m bench.cli run --config-dir /scratch/v28/c7sw-config-$s --dataset laion30m --dim 256 --suite c7w-laion30m \
          --algo silvertorch "${args[@]}" --sweep $sw --mode eager --profile --interleave --force --out "$R/r$round-$s" ) >> "$R/logs/r$round-$s-$sw.log" 2>&1 < /dev/null
      rc=$?
      rm -rf "$R/r$round-$s/_parity"
      echo "$(date -Is) round $round tree $s $sw rc=$rc s=$(( $(date +%s) - t0 ))"
      [ $rc -eq 0 ] || exit $rc
    done
  done
done
cd /scratch/wt/v28-official/evaluation && /venvs/d-run-v28/bin/python -m bench.cli upload --results "$R" --path-in-repo artifacts/c7-laion30m-st-wide --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
