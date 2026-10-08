#!/usr/bin/env bash
# IVF-TUNE PubMed, the last n_lists 16384 points (n_probe 2048, then the 4096 cap) at ks {100} only: with k 1000
# codesigned_probe_score_exact's [next_pow2(k), next_pow2(n_probe)] tile is 1024 x 2048 = 2^21 elements, above
# Triton's 2^20 numel limit (compile error), so ks {100, 1000} cannot run n_probe >= 2048. k 100 builds its own
# all5 oracle (k_gt 100). Then the table and "ivf-tune rc=0" (chain-v21's gate).
set -u
W=/scratch/wt/ivf-tune
PY=/venvs/retrieve/bin/python
R=/scratch/ivf-tune/results
L=/scratch/ivf-tune
J=$R/ivf-tune/pubmed-d768.jsonl
T="$PY $W/docs/artifacts/campaign-v2/ivf-tune/tune.py $J"
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/ivf-tune
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
cd $W/evaluation
for np in 2048 4096; do
  cfg=$L/config-16384-$np-k100; mkdir -p $cfg
  cp config/pubmed.yaml $cfg/
  { cat config/suites.yaml; cat <<YAML

ivf-tune:                      # IVF-TUNE scratch suite (docs/artifacts/campaign-v2/ivf-tune), k 100 only
  perf: false
  datasets: [pubmed]
  dims: [768]
  filter_kinds: [clause]
  ks: [100]
  batch_sizes: [16]
  seeds: [0]
  sweeps: {pubmed: [all5]}
  arms:
    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [$np]}}
YAML
  } > $cfg/suites.yaml
  t0=$(date +%s)
  $PIN $PY -m bench.cli run --config-dir $cfg --dataset pubmed --dim 768 --suite ivf-tune --out "$R" --resume \
    > "$L/logs/run-16384-$np-k100.log" 2>&1
  rc=$?
  echo "$(date -Is) n_lists 16384 n_probe $np (k 100) rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || { echo "$(date -Is) ivf-tune stopped: n_probe $np failed at k 100 too"; exit 1; }
  $T --reached 16384 && break
done
$T | tee "$L/table.txt"
nvidia-smi -q -d CLOCK > "$L/logs/nvidia-smi-clock-end.txt"
echo "$(date -Is) ivf-tune rc=0"
