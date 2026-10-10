#!/usr/bin/env bash
# PubMed 10 M d768 tune, timed phase at campaign-v2.11 (library 1d390792; controller 2026-10-10): scratch suite `tune-t`
# (tune_timed_config.py: the 35 v2.10 candidates, bs 1 / 16 / 64, eager + graph), one `bench run` per (n_lists, sweep, filter kind)
# so st-dloop's blocks fit between steps; n_lists ascending, stop file between n_lists. Oracles built at v2.11 into a mirror data
# dir's own gt_d768, then torch.equal against the v2.10 mirror's. Pod a100-x1-b GPU 0, st-dloop first.
# setsid nohup bash driver.sh > /scratch/pubmed-tune-v211/driver.log 2>&1 &
set -u
T=/scratch/wt/tree-v211
D=/scratch/wt/pubmed-tune-timed/docs/artifacts/campaign-v2.11/pubmed-tune-timed
PY=/venvs/v211/bin/python
R=/scratch/campaign-v2.11/results
L=/scratch/pubmed-tune-v211
C=$L/cfg
M=$L/data/pubmed-medcpt
export PYTHONPATH=$T/evaluation:$T/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/pubmed-tune-v211
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $T/evaluation
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('campaign.yaml'))['default']['perf']['code_version'])")
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }
step() { local t0=$(date +%s) c="$*"; "$@"; local rc=$?; echo "$(date -Is) step ${c#$PIN } rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc; }

step $PY $D/tune_timed_config.py /scratch/pubmed-tune-v210/candidates.json $C $M
yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
yield_gpu
step $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite tune-t --config-dir $C/nl1024
step $PY $D/oracle_compare.py /scratch/pubmed-tune-v210/data/pubmed-medcpt/gt_d768 $M/gt_d768
for nl in 1024 4096 16384 65536; do
  [ -e $L/tune.stop ] && { echo "$(date -Is) tune.stop: stopped before n_lists $nl"; break; }
  for sk in all5:clause c0_mesh:bloom c0_mesh:clause c3_journal_reverse:clause; do
    log="$R/_logs/tune-t_pubmed-d768_nl${nl}_${sk/:/_}.log"
    yield_gpu; touch /scratch/gpu0.timed
    { echo "=== clocks at start"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
    step $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite tune-t --config-dir $C/nl$nl --sweep ${sk%:*} \
      --filter-kind ${sk#*:} --out $R --resume >> "$log" 2>&1
    { echo "=== clocks at end"; $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; } >> "$log"
    rm -f /scratch/gpu0.timed
  done
done
echo "$(date -Is) driver done"
