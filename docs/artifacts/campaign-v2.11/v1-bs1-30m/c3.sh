#!/usr/bin/env bash
# C3 re-run on dev/v1-bs1 9b7e451 (library a5c67b68): LiNR V1 torch.compile max-autotune vs Triton, clause p 0.01 / 0.1 / 1, bs 1 + 16, k 100, seed 0,
# on laion30m-synth d256 and yfcc10m-synth d192, one process per dataset, pod d GPU 1, after the V1-BS1 gate. Hub artifacts/v1-bs1-c3.
#   setsid nohup flock -n /scratch/gpu1.lock bash c3.sh > /scratch/v211/v1-bs1-c3.driver.log 2>&1 &
set -u
R=${R:-/scratch/campaign-v211/v1-bs1-c3}
REPO=${B_REPO:-/scratch/wt/v1-bs1-9b7e451}
PY=${B_PY:-/venvs/d-run-v1bs1/bin/python}
CFG=/scratch/v211/c3b-config
mkdir -p "$R/logs"
echo "pod-d v1-bs1-c3 on GPU 1 (taskset -c 96-127) since $(date -Is), driver $0" > /scratch/gpu1-holder
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG"
printf 'c3b:\n  datasets: [laion30m-synth, yfcc10m-synth]\n  dims: [192, 256]\n  filter_kinds: [clause]\n  ks: [100]\n  batch_sizes: [1, 16]\n  seeds: [0]\n  sweeps:\n    laion30m-synth: [p001, p01, p1]\n    yfcc10m-synth: [p001, p01, p1]\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [triton]}\n    - {algo: linr_v1_filter_mask, backends: [torch], build: {compile: [max-autotune]}}\n' >> "$CFG/suites.yaml"
got=$(cd "$REPO/evaluation" && $PY -m bench.cli env | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d["code_version"], d["dirty"])')
echo "$(date -Is) tree: $got"
case "$got" in "${B_CV:-a5c67b68}"*" False") ;; *) echo "code_version mismatch, refusing"; exit 2 ;; esac
for dd in laion30m-synth:256 yfcc10m-synth:192; do
  ds=${dd%%:*}; dim=${dd#*:}; t0=$(date +%s)
  ( cd "$REPO/evaluation" && CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v1g-B${INDUCTOR_SUFFIX:-} taskset -c 96-127 \
      $PY -m bench.cli run --config-dir "$CFG" --dataset $ds --dim $dim --suite c3b --algo linr_v1_filter_mask --backend triton --backend torch \
      --force --out "$R" ) >> "$R/logs/$ds.log" 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) $ds rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc
done
cd "$REPO/evaluation" && $PY -m bench.cli upload --results "$R" --path-in-repo artifacts/v1-bs1-c3 --verify 2>&1 | grep -E "MANIFEST|round trip"
echo "$(date -Is) driver done"
rm -f /scratch/gpu1-holder
