#!/usr/bin/env bash
# IVF-TUNE, PubMed piece, on pod a100-x1-b (GPU 0 shared through /scratch/gpu0.lock, NUMA node 0
# cores): SilverTorch triton, clause all5, seed 0, bs 16, quality only, at campaign-v2. Per
# n_lists, n_probe doubles from 8 in rounds of three (one index build a round) until a round
# reaches recall_oracle@100 >= 0.95, or n_probe = n_lists. ks {100, 1000} so the all5 oracle
# blob (k_gt 1000) is reused; the pick reads k 100. Each round is a scratch `ivf-tune` suite
# appended to a copy of config/suites.yaml: records are artifacts, in their own results tree.
set -u
W=/scratch/wt/ivf-tune
PY=/venvs/retrieve/bin/python
R=/scratch/ivf-tune/results
L=/scratch/ivf-tune
EXPECT=408b1188d542634b3d18a2f5077bd23537a845fc
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/ivf-tune
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R" "$L/logs" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation

cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -q -d CLOCK > "$L/logs/nvidia-smi-clock-start.txt"

for nl in "$@"; do
  np=8
  while :; do
    grid=(); for _ in 1 2 3; do [ $np -le $nl ] && grid+=($np); np=$(( np * 2 )); done
    [ ${#grid[@]} -gt 0 ] || { echo "$(date -Is) n_lists $nl: n_probe reached n_lists without 0.95"; break; }
    cfg=$L/config-$nl-${grid[0]}; mkdir -p $cfg
    cp config/pubmed.yaml $cfg/
    { cat config/suites.yaml; cat <<YAML

ivf-tune:                      # IVF-TUNE scratch suite (docs/artifacts/campaign-v2/ivf-tune)
  perf: false
  datasets: [pubmed]
  dims: [768]
  filter_kinds: [clause]
  ks: [100, 1000]
  batch_sizes: [16]
  seeds: [0]
  sweeps: {pubmed: [all5]}
  arms:
    - {algo: silvertorch, backends: [triton], build: {n_lists: [$nl]}, query: {n_probe: [$(IFS=,; echo "${grid[*]}")]}}
YAML
    } > $cfg/suites.yaml
    t0=$(date +%s)
    $PIN $PY -m bench.cli run --config-dir $cfg --dataset pubmed --dim 768 --suite ivf-tune --out "$R" --resume \
      > "$L/logs/run-$nl-${grid[0]}.log" 2>&1
    rc=$?
    echo "$(date -Is) n_lists $nl n_probe ${grid[*]} rc=$rc s=$(( $(date +%s) - t0 ))"
    [ $rc -eq 0 ] || exit $rc
    $PY $W/docs/artifacts/campaign-v2/ivf-tune/tune.py "$R/ivf-tune/pubmed-d768.jsonl" --reached $nl && break
  done
done
$PY $W/docs/artifacts/campaign-v2/ivf-tune/tune.py "$R/ivf-tune/pubmed-d768.jsonl" | tee "$L/table.txt"
nvidia-smi -q -d CLOCK > "$L/logs/nvidia-smi-clock-end.txt"
echo "$(date -Is) driver done"
