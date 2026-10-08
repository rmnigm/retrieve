#!/usr/bin/env bash
# D3 arXiv: wait for the GPU to be idle, then each suite under its own lock hold (released in between).
D=/scratch/wt/d3-arxiv/docs/artifacts/campaign-v2/d3-arxiv/driver.sh
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "$(date -Is) gpu idle"
for s in bloomwidth bloomwidth-timed; do
  mkdir -p /scratch/arxiv-$s
  t0=$(date +%s)
  flock /scratch/gpu0.lock bash $D $s > /scratch/arxiv-$s/driver.log 2>&1
  rc=$?
  echo "$(date -Is) suite $s rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
  sleep 60   # let v-graph-ids take the lock between suites
done
echo "$(date -Is) d3 done"
