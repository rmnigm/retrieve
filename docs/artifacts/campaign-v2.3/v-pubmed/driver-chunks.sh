#!/usr/bin/env bash
# V-PUBMED filter, SilverTorch chunks (VPUBMED_CHUNKS=st), at campaign-v2.3 on pod a100-x1-b, GPU 0 shared with st-dloop (it has priority): the 90 PubMed
# `filter` cells, as 48 chunks of the real suite (chunks.py: CLI narrowings that keep every
# record key and interleave unit). Per chunk, what `bench campaign` does around a child: wait while
# /scratch/gpu0.st-dloop-wants or /scratch/gpu0.st-dloop-phase exists, prune _parity/ to the chunk's (dataset, dim, algo), a log under _logs/ with
# `nvidia-smi -q -d CLOCK` at start and end, one `bench run … --resume --interleave` under the flock and the timed flag.
# Stops on the first non-zero chunk. VPUBMED_CHUNKS=st setsid nohup bash driver-chunks.sh > /scratch/v-pubmed/driver-chunks-v23.log 2>&1 &
set -u
W=/scratch/wt/v-pubmed-v23
PY=/venvs/retrieve/bin/python
R=/scratch/campaign-v2.3/results
L=/scratch/v-pubmed/filter-v23
export PYTHONPATH=$W/evaluation:$W/retrieve/src RETRIEVE_DATA_ROOT=/data HF_HOME=/scratch/hf
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-pubmed-v23
PIN="flock /scratch/gpu0.lock taskset -c 0-95"
mkdir -p "$R/_logs" "$L" "$TORCHINDUCTOR_CACHE_DIR"
cd $W/evaluation
EXPECT=$($PY -c "import yaml; print(yaml.safe_load(open('campaign.yaml'))['default']['perf']['code_version'])")
yield_gpu() { while [ -e /scratch/gpu0.st-dloop-wants ] || [ -e /scratch/gpu0.st-dloop-phase ]; do sleep 20; done; }   # st-dloop first: per job and for its whole GPU phase

yield_gpu
cv=$($PIN $PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
echo "$(date -Is) code_version $cv (campaign.yaml $EXPECT)"
[ "$cv" = "$EXPECT" ] || { echo "$(date -Is) code_version mismatch, refusing"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$L/clocks.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0.timed' EXIT
clocks() { $PY -c 'from bench import measure; print(measure.clock_report(), end="")'; }

yield_gpu
t0=$(date +%s); $PIN $PY -m bench.cli oracle --dataset pubmed --dim 768 --suite filter > "$L/oracle.log" 2>&1
rc=$?; echo "$(date -Is) oracle rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit $rc

$PY $W/docs/artifacts/campaign-v2.3/v-pubmed/chunks.py 2>/dev/null | grep -- "--algo" | sed 's/^ *[0-9]* cells  //' > "$L/chunks.txt"
n=0
while read -r args; do
  n=$(( n + 1 ))
  algo=$(echo "$args" | awk '{print $2}')
  $PY - "$R/_parity" "$algo" <<'PYEOF'
import shutil, sys
from pathlib import Path
from bench import run
keep = run.parity_group("pubmed", 768, sys.argv[2])
for p in Path(sys.argv[1]).glob("*"):
    if p.name != keep:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
PYEOF
  name="filter_pubmed-d768_chunk$(printf %02d $n).log"
  yield_gpu
  touch /scratch/gpu0.timed
  t0=$(date +%s)
  { echo "=== bench run --dataset pubmed --dim 768 --suite filter $args --out $R --resume --interleave"; echo "=== clocks at start"; clocks; } >> "$R/_logs/$name"
  $PIN $PY -m bench.cli run --dataset pubmed --dim 768 --suite filter $args --out "$R" --resume --interleave >> "$R/_logs/$name" 2>&1 < /dev/null
  rc=$?
  { echo "=== clocks at end"; clocks; } >> "$R/_logs/$name"
  rm -f /scratch/gpu0.timed
  echo "$(date -Is) chunk $n rc=$rc s=$(( $(date +%s) - t0 )) $args"
  [ $rc -eq 0 ] || exit $rc
done < "$L/chunks.txt"
rm -rf "$R/_parity"
$PY -c 'import sys; from pathlib import Path; from bench import records; print("=== aggregated", records.aggregate(Path(sys.argv[1])))' "$R"
echo "$(date -Is) driver done"
