#!/usr/bin/env bash
# kill the k-means test at the controller's 3 GPU-h cap (start 01:44Z -> 04:44Z) if it is still running
until [ "$(date -u +%H%M)" -ge 0444 ] && [ "$(date -u +%Y%m%d)" = 20261013 ]; do
  pgrep -f "run_kmeans.sh /scratch/wt/kmeans-test" >/dev/null || { echo "driver ended $(date -u +%T), watchdog off"; exit 0; }
  sleep 30
done
pgrep -f "run_kmeans.sh /scratch/wt/kmeans-test" >/dev/null || { echo "driver ended, watchdog off"; exit 0; }
echo "CAP-KILL $(date -u +%T): 3 GPU-h reached"; pkill -f "run_kmeans.sh /scratch/wt/kmeans-test"; pkill -f "bench_cached.py"; pkill -f "fit_centroids.py"
