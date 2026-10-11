#!/usr/bin/env bash
# ST-WIDE-2 30 M gate (gates v2.10): laion30m d256 bloom partial, n_probe 128 x bs {4, 16, 64}, c0_domain + tags4, eager + graph, k 100, seed 0, on pod d
# GPU 1; 3 rounds of A = v2.9 (staging e8958bd2: ours + Meta -O3 interleaved by backend, /venvs/d-run-v29-ax) and B = dev/st-wide-2 a323acf (library
# a3bec4a5: ours, /venvs/d-run-stw2), the tree built first alternating (A B, B A, A B), one process per tree per round (H-ARMFREE), each tree its own
# inductor dir. analyze.py: B / A and ours / Meta per cell with a t-CI over rounds, ids_sha256 A vs B.
#   setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v29/st-wide-2-30m.driver.log 2>&1 &
set -u
R=/scratch/campaign-v29/laion30m-st-wide-2-30m
declare -A REPO=([A]=/scratch/wt/v29-axsynth [B]=/scratch/wt/st-wide-2-a323acf)
declare -A PY=([A]=/venvs/d-run-v29-ax/bin/python [B]=/venvs/d-run-stw2/bin/python)
declare -A CV=([A]=e8958bd234cf9f814d1afcb2e5eb3f95e25dbec3 [B]=a3bec4a5)
declare -A BACK=([A]="triton official" [B]="triton")
mkdir -p "$R/logs"
echo "pod-d st-wide-2-30m (cross-tree v2.9 / st-wide-2 a323acf) on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 1000 >> "$R/logs/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
for s in A B; do
  cfg=/scratch/v29/stw2-config-$s
  rm -rf "$cfg" && cp -r "${REPO[$s]}/evaluation/config" "$cfg"
  printf 'stw2-laion30m:\n  datasets: [laion30m]\n  dims: [256]\n  filter_kinds: [bloom]\n  ks: [100]\n  batch_sizes: [4, 16, 64]\n  seeds: [0]\n  sweeps:\n    laion30m: [c0_domain, tags4]\n  interleave: [{by: backend, values: [triton, official]}]\n  arms:\n    - algo: silvertorch\n      backends: [%s]\n      build: {bloom_path: [partial], n_lists: [16384]}\n      query: {n_probe: [128]}\n' \
    "$(echo ${BACK[$s]} | sed 's/ /, /')" >> "$cfg/suites.yaml"
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
    ( cd "${REPO[$s]}/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/stw2-$s \
        taskset -c 96-127 ${PY[$s]} -m bench.cli run --config-dir /scratch/v29/stw2-config-$s --dataset laion30m --dim 256 --suite stw2-laion30m \
        --algo silvertorch "${args[@]}" --interleave --force --out "$R/r$round-$s" ) >> "$R/logs/r$round-$s.log" 2>&1 < /dev/null
    rc=$?
    rm -rf "$R/r$round-$s/_parity"
    echo "$(date -Is) round $round tree $s rc=$rc s=$(( $(date +%s) - t0 ))"
    [ $rc -eq 0 ] || exit $rc
  done
done
kill $SMI 2>/dev/null; sleep 2
cd /scratch/wt/v29-axsynth/evaluation && /venvs/d-run-v29-ax/bin/python -m bench.cli upload --results "$R" --path-in-repo artifacts/st-wide-2-30m --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
