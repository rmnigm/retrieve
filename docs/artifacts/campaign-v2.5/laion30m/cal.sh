#!/usr/bin/env bash
# V-LAION30 calibration, its own results tree $L/cal (artifacts, never a campaign tree): the oracle blobs of the grid's
# sweeps; SilverTorch triton n_lists 16384 x n_probe {24, 256, 1024, 4096} (4096 = the 25 % cap) on c0_domain + tags4,
# quality only; then one timed cell per arm (V1, V2, SilverTorch n_probe 24; c0_domain, bs 16, k 100) for the rate.
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$L/cal/config
mkdir -p "$C"
cp config/laion30m.yaml "$C/"
{ cat config/suites.yaml; cat <<'YAML'

laion-cal:                     # V-LAION30 calibration scratch suites (docs/artifacts/campaign-v2.5/laion30m)
  perf: false
  datasets: [laion30m]
  dims: [256]
  filter_kinds: [clause]
  ks: [100]
  batch_sizes: [16]
  seeds: [0]
  sweeps: {laion30m: [c0_domain, tags4]}
  arms:
    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [24, 256, 1024, 4096]}}

laion-cal-timed:
  datasets: [laion30m]
  dims: [256]
  filter_kinds: [clause]
  ks: [100]
  batch_sizes: [16]
  seeds: [0]
  sweeps: {laion30m: [c0_domain]}
  arms:
    - {algo: linr_v1_filter_mask, backends: [triton]}
    - {algo: linr_v2, backends: [triton]}
    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [24]}}
YAML
} > "$C/suites.yaml"

run oracle oracle --dataset laion30m --suite laion30m --dim 256
run quality run --config-dir "$C" --dataset laion30m --dim 256 --suite laion-cal --out "$L/cal/quality" --resume
$PY "$HERE/cal_table.py" "$L/cal/quality" | tee "$L/cal/table-quality.md"
run timed run --config-dir "$C" --dataset laion30m --dim 256 --suite laion-cal-timed --out "$L/cal/timed" --resume
$PY "$HERE/cal_table.py" "$L/cal/timed" | tee "$L/cal/table-timed.md"
echo "$(date -Is) cal rc=0"
