#!/usr/bin/env bash
# V-ROUTER PubMed: the 9 cells that hit CUDA OOM when the whole `router` suite ran in one process (each arm's 10 M x 768
# index is ~15-17 GiB and memory carried over between arms) rerun one process per (algo, sweep), `--resume` (ok cells
# skipped; keys unchanged), same worktree / tree / code_version, under the flock, the timed flag and st-dloop's yields.
set -u
W=/scratch/wt/v-router-pubmed
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2.5/results
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-router-pubmed-v25
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
cd $W/evaluation
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }
trap 'rm -f /scratch/gpu0.timed' EXIT
for spec in "router c0_mesh" "router c3_journal_reverse" "router all5" "linr_v2 c0_mesh" "linr_v2 all5" \
            "linr_v1_filter_mask c0_mesh" "linr_v1_filter_mask all5"; do
  set -- $spec
  yield_gpu; touch /scratch/gpu0.timed; t0=$(date +%s)
  $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite router --algo $1 --backend triton --filter-kind clause \
    --sweep $2 --seed 0 --out "$R" --resume --interleave >> "$R/_logs/router_pubmed-d768_rerun.log" 2>&1
  rc=$?; rm -f /scratch/gpu0.timed
  echo "$(date -Is) rerun $1 $2 rc=$rc s=$(( $(date +%s) - t0 ))"
done
echo "$(date -Is) rerun done"
