#!/usr/bin/env bash
# IVF-TUNE PubMed controller (replaces driver.sh mid-run, orchestrator-b's cost guidance): adopts the running
# round's bench process ($1, may be empty), stops any round as soon as one of its cells reaches
# recall_oracle@100 >= 0.95 (records are appended per cell, resume-safe), and above n_probe 256 runs one-point
# rounds (a rebuild is cheaper than an overshoot cell). Then n_lists 16384 the same way. Ends with the table and
# "ivf-tune rc=0" in /scratch/ivf-tune/chain.log (chain-v21's gate).
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

watch() {  # $1 n_lists, $2 the round's flock pid (its child is the bench process)
  local nl=$1 pid=$2
  while kill -0 $pid 2>/dev/null; do
    sleep 10
    if $T --reached $nl; then
      pkill -TERM -P $pid; kill $pid 2>/dev/null
      echo "$(date -Is) n_lists $nl reached 0.95: round stopped early"; return
    fi
  done
}

round() {  # $1 n_lists, rest: the round's n_probe grid
  local nl=$1; shift
  local cfg=$L/config-$nl-$1; mkdir -p $cfg
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
    - {algo: silvertorch, backends: [triton], build: {n_lists: [$nl]}, query: {n_probe: [$(IFS=,; echo "$*")]}}
YAML
  } > $cfg/suites.yaml
  local t0=$(date +%s)
  $PIN $PY -m bench.cli run --config-dir $cfg --dataset pubmed --dim 768 --suite ivf-tune --out "$R" --resume \
    > "$L/logs/run-$nl-$1.log" 2>&1 &
  local pid=$!
  watch $nl $pid
  wait $pid; local rc=$?
  echo "$(date -Is) n_lists $nl n_probe $* rc=$rc s=$(( $(date +%s) - t0 ))"
}

maxdone() { $PY -c "
import json,sys
xs=[json.loads(l)['params']['n_probe'] for l in open('$J') if json.loads(l)['status']=='ok' and json.loads(l)['params']['n_lists']==$1]
print(max(xs, default=0))"; }

tune() {  # $1 n_lists: continue from the last measured point
  local nl=$1
  while ! $T --reached $nl; do
    local np=$(( $(maxdone $nl) * 2 )); [ $np -ge 8 ] || np=8
    [ $np -le $nl ] || { echo "$(date -Is) n_lists $nl: n_probe reached n_lists without 0.95"; return; }
    if [ $np -lt 256 ]; then round $nl $np $(( np * 2 )) $(( np * 4 )); else round $nl $np; fi
  done
}

if [ -n "${1:-}" ]; then watch 4096 $1; while kill -0 $1 2>/dev/null || pgrep -P $1 >/dev/null; do sleep 5; done
  echo "$(date -Is) adopted round (pid $1) ended"; fi
tune 4096
tune 16384
$T | tee "$L/table.txt"
nvidia-smi -q -d CLOCK > "$L/logs/nvidia-smi-clock-end.txt"
echo "$(date -Is) ivf-tune rc=0"
