#!/usr/bin/env bash
# ST-IDS compile-time sweep: every (pkg, k, n_probe) in its own process, its own cold TRITON_CACHE_DIR, its own core
# among 96-191 (pod b's timed jobs are pinned to 0-95). CPU only; SWEEP_PKGS picks the packages. Usage: compile_sweep.sh <out_dir> <pkgs_dir>
set -u
OUT=$1 PKGS=$2 ART=$(dirname "$0"); mkdir -p "$OUT"
core=96
for pkg in ${SWEEP_PKGS:-retrieve_v22 retrieve}; do
  for k in 100 1000; do
    for np in 24 64 128 256 512 1024 2048 4096; do
      D=$(mktemp -d -p "$OUT")
      ( CUDA_VISIBLE_DEVICES="" TRITON_CACHE_DIR=$D PYTHONPATH=$PKGS:/scratch/wt/st-ids/retrieve/src \
        timeout 3h taskset -c $core /venvs/retrieve/bin/python "$ART/compile_time.py" --pkg $pkg --k $k --n-probe $np \
        > "$OUT/${pkg}_k${k}_np${np}.json" 2> "$OUT/${pkg}_k${k}_np${np}.err"; echo "rc=$?" >> "$OUT/${pkg}_k${k}_np${np}.err"; rm -rf "$D" ) &
      core=$((core + 1))
    done
  done
done
wait
