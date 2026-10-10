#!/usr/bin/env bash
# Final-pass rebalance runner, pod d GPU 0 (laion; controller 2026-10-10: d1-20 / d1-22 / d1-23 moved from GPU 1 to GPU 0, appended
# after laion's d0 queue): legs.sh with these overrides only: CUDA_VISIBLE_DEVICES=0, TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/final-gpu0,
# --out /scratch/final/gpu0 (the recipe says gpu1), upload path the recipe's. Otherwise as legs.sh: for each LEG_ID, the tag's recipe.yaml gives env and command (run exactly as written);
# the upload is the brief's corrected form (this leg's files staged under <suite>/, n_records + provenance checked on a dry run); pinned to cores 64-95, logs under /scratch/final/logs/<leg>.log, wall time and the manifest sha appended to
# $NOTE as '<leg> done ...' lines (d-run's runner matches them). Stops on the first non-zero rc (reported, not worked around). Launch under the GPU 0 lock:
#   NOTE=<chain note> setsid nohup flock -n /scratch/gpu0.lock bash legs.sh d0-01-... d0-02-... > LOG 2>&1 &
set -u
: "${NOTE:?NOTE=the worker chain note}"
if flock -n /scratch/gpu0.lock true; then echo "$(date -Is) not launched under /scratch/gpu0.lock, refusing"; exit 3; fi
T=/scratch/wt/final
LOGD=/scratch/final/logs
mkdir -p "$LOGD"
cd "$T/evaluation"
[ "$(git -C $T rev-parse HEAD:retrieve/src/retrieve)" = e16512f5b379fc7e2d370cc8a57c68b9854a06d7 ] || { echo "tree is not campaign-v2.12"; exit 2; }
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 1000 >> "$LOGD/clocks-gpu0.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null; rm -f /scratch/gpu0-holder' EXIT
for leg in "$@"; do
  echo "pod-d laion final leg $leg on GPU 0 (taskset -c 64-95) since $(date -Is), driver $0" > /scratch/gpu0-holder
  spec=$(python3 - "$T/docs/artifacts/campaign-final/recipe.yaml" "$leg" <<'PY'
import shlex, sys, yaml
leg = next(l for l in yaml.safe_load(open(sys.argv[1]))["legs"] if l["id"] == sys.argv[2])
assert leg["id"].split("-")[0:2] in (["d1", "20"], ["d1", "22"], ["d1", "23"]), leg["id"]
env = {**leg["env"], "CUDA_VISIBLE_DEVICES": "0", "TORCHINDUCTOR_CACHE_DIR": "/scratch/inductor/final-gpu0"}
for k, v in env.items():
    print(f"export {k}={shlex.quote(str(v))}")
assert leg["command"].count("--out /scratch/final/gpu1 ") == 1, leg["command"]
print(f"CMD={shlex.quote(leg['command'].replace('--out /scratch/final/gpu1 ', '--out /scratch/final/gpu0 '))}")
import re
print(f"UPP={shlex.quote(leg['upload'])}")
print(f"SUITE={shlex.quote(leg['suite'])}")
stem = leg["dataset"] + "-d" + re.search(r"--dim (\d+)", leg["command"]).group(1)
print(f"STEM={shlex.quote(stem)}")
PY
) || { echo "$(date -Is) $leg: recipe parse failed or not a laion / GPU 0 leg"; exit 5; }
  eval "$spec"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
  t0=$(date +%s)
  echo "=== $CMD" >> "$LOGD/$leg.log"
  taskset -c 64-95 bash -c "$CMD" >> "$LOGD/$leg.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date -Is) $leg run rc=$rc s=$(( $(date +%s) - t0 ))"
  [ $rc -eq 0 ] || { printf -- '- %s FAILED rc=%s (run) after %ss\n' "$leg" "$rc" "$(( $(date +%s) - t0 ))" >> "$NOTE"; exit $rc; }
  # corrected upload (brief CORRECTION, 15:20Z): this leg's suite files hardlinked into $U/<suite>/, then upload $U
  U=$(mktemp -d /scratch/final/upload.XXXXXX); mkdir -p "$U/$SUITE"
  cp -al /scratch/final/gpu0/$SUITE/$STEM* "$U/$SUITE/" || { echo "$(date -Is) $leg: nothing to stage"; exit 6; }
  nrec=$(cat "$U/$SUITE/$STEM.jsonl" | wc -l)
  dry=$("$VENV/bin/python" -m bench.cli upload --results "$U" --path-in-repo "$UPP" --dry-run 2>&1 | tee -a "$LOGD/$leg.upload.log")
  grep -q " $nrec record(s) " <<< "$dry" && grep -q e16512f5 <<< "$dry" || {
    printf -- '- %s upload check FAILED (staged %s records; manifest or provenance mismatch)\n' "$leg" "$nrec" >> "$NOTE"; exit 7; }
  up=$("$VENV/bin/python" -m bench.cli upload --results "$U" --path-in-repo "$UPP" --verify 2>&1 | tee -a "$LOGD/$leg.upload.log" | grep -E "MANIFEST|round trip|results.parquet")
  ls "$U/results.parquet" > /dev/null 2>&1 || up="$up (no results.parquet)"
  rm -rf "$U"
  sha=$(sed -n 's/.*MANIFEST.json sha256 \([0-9a-f]*\).*/\1/p' <<< "$up")
  echo "$(date -Is) $leg upload $UPP: $up"
  grep -q "round trip verified" <<< "$up" || { printf -- '- %s upload FAILED after %ss\n' "$leg" "$(( $(date +%s) - t0 ))" >> "$NOTE"; exit 4; }
  printf -- '- %s done %s, %ss, `%s` %s records, MANIFEST %s\n' "$leg" "$(date -u +%H:%MZ)" "$(( $(date +%s) - t0 ))" "$UPP" "$nrec" "$sha" >> "$NOTE"
done
echo "$(date -Is) legs rc=0"
