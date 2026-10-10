#!/usr/bin/env bash
# st-dloop's ST-XQ ask (control 2026-10-13-190000000): one profiled 30 M cell set on laion30m d256, bloom partial, c0_domain + tags4,
# n_lists 16384, n_probe 128, bs 16 / 64, at library e16512f5 (dev/v1-topk 8c9b336, this tree), pod d GPU 0, Meta's extension -O3:
# (1) bounds_share.py: rows' rarest-bit bounds, bloom pass rate, sparse share; (2) campaign-v2.10/st-topk/prof.py with official (int32):
# the kernel split, ours vs Meta. Outputs + logs -> Hub artifacts/st-xq-30m-profile.
#   GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash prof.sh > /scratch/laion30m/xq/prof.driver.log 2>&1 &
set -u
GPU=${GPU:-0}
if flock -n /scratch/gpu$GPU.lock true; then echo "$(date -Is) not launched under /scratch/gpu$GPU.lock, refusing"; exit 3; fi
W=/scratch/wt/laion-xq
PY=/venvs/laion-xq/bin/python
L=/scratch/laion30m/xq
R=$L/out
case $GPU in 0) PIN="taskset -c 64-95" ;; 1) PIN="taskset -c 96-127" ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/laion-xq-gpu$GPU HF_HOME=/scratch/hf PYTHONPATH=.:../retrieve/src
mkdir -p "$R/logs" "$TORCHINDUCTOR_CACHE_DIR"
echo "pod-d laion st-xq-30m prof.sh on GPU $GPU ($PIN) since $(date -Is), driver $0" > /scratch/gpu$GPU-holder
trap 'rm -f /scratch/gpu$GPU-holder; kill $SMI 2>/dev/null' EXIT
cd "$W/evaluation"
cv=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
[ "$cv" = "e16512f5b379fc7e2d370cc8a57c68b9854a06d7" ] || { echo "$(date -Is) code_version $cv is not e16512f5, refusing"; exit 2; }
$PY -m bench.cli env > "$R/env.json"
nvidia-smi -i $GPU --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$R/logs/clocks-gpu$GPU.csv" &
SMI=$!
step() {  # NAME SCRIPT ARGS...
  local name=$1; shift
  local t0=$(date +%s)
  $PIN $PY "$@" > "$R/logs/$name.log" 2>&1 < /dev/null
  local rc=$?
  echo "$(date -Is) $name rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || exit $rc
}
A=$W/docs/artifacts
step bounds "$A/campaign-v2.12/st-xq-30m/bounds_share.py" laion30m 256 c0_domain,tags4 16384 "$R/bounds.json"
for sw in c0_domain tags4; do
  step "prof-$sw" "$A/campaign-v2.10/st-topk/prof.py" laion30m 256 bloom $sw 16384 bloom 128 16,64 "$R/prof-$sw.json" official
done
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null  # the trace must stop before the upload hashes it
up=$($PY -m bench.cli upload --results "$R" --path-in-repo artifacts/st-xq-30m-profile --verify 2>&1 | tee "$L/upload.log" | grep -E "MANIFEST|round trip")
echo "$(date -Is) upload: $up"
grep -q "round trip verified" <<< "$up" || exit 4
echo "$(date -Is) prof rc=0"
