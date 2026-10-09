#!/usr/bin/env bash
# V-LAION30 v2.9: IVF vs exact at recall 0.95 inside one leg (controller GO): scratch suite `laion30m-x` (config dir $C, written
# here), per sweep (tags4 first, then c0_domain) one --interleave process of V2 + SilverTorch n_lists 16384 n_probe 4096 (tags4's
# n95; c0_domain's cap, 0.903) grouped by [algo, n_lists], then V1 alone: V1 + V2 + SilverTorch at 30 M do not fit one A100-80GB.
# bs 1 + 16, k 100, seed 0, clause, triton. Pod d GPU 0; Hub campaign-v2.9/laion30m-x.
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$L/x-config
R=/scratch/campaign-v29/laion30m-x
mkdir -p "$C" "$R/logs"
cp config/laion30m.yaml "$C/"
{ cat config/suites.yaml; cat <<'YAML'

laion30m-x:                    # V-LAION30 v2.9: IVF vs exact at 0.95 inside one leg (docs/artifacts/campaign-v2.9/laion30m/x.sh)
  datasets: [laion30m]
  dims: [256]
  filter_kinds: [clause]
  ks: [100]
  batch_sizes: [1, 16]
  seeds: [0]
  sweeps:
    laion30m: [c0_domain, tags4]
  interleave:
    - {by: [algo, n_lists]}       # run with --algo linr_v2 --algo silvertorch: V1 + V2 + SilverTorch at 30 M do not fit one GPU
  arms:
    - {algo: linr_v1_filter_mask, backends: [triton]}
    - {algo: linr_v2, backends: [triton]}
    - {algo: silvertorch, backends: [triton], build: {n_lists: [16384]}, query: {n_probe: [4096]}}
YAML
} > "$C/suites.yaml"
for sw in tags4 c0_domain; do
  run "x-v2st-$sw" run --config-dir "$C" --dataset laion30m --dim 256 --suite laion30m-x --sweep $sw \
    --algo linr_v2 --algo silvertorch --interleave --out "$R" --resume
  run "x-v1-$sw" run --config-dir "$C" --dataset laion30m --dim 256 --suite laion30m-x --sweep $sw \
    --algo linr_v1_filter_mask --out "$R" --resume
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/x-*.log "$R/logs/"
$PY "$W/docs/artifacts/campaign-v2.5/laion30m/cal_table.py" "$R" | tee "$R/table.md"
run upload-x upload --results "$R" --path-in-repo campaign-v2.9/laion30m-x --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-x.log"
echo "$(date -Is) x rc=0"
