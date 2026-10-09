#!/usr/bin/env bash
# V-LAION30 v2.9 part 2 (st.sh was stopped after part 1 for x.sh, controller): laion30m-synth SilverTorch, one process per sweep
# (4 builds, 12 cells), pod d GPU 0 -> campaign-v2.9/laion30m-synth-laion30m-synth. Rerun after a crash: --resume skips finished cells.
. "$(dirname "$(readlink -f "$0")")/common.sh"
R=/scratch/campaign-v29/laion30m-synth-laion30m-synth
mkdir -p "$R/logs"
for sw in p01 p02 p05 p1; do
  run "synth-st-$sw" run --dataset laion30m-synth --dim 256 --suite laion30m-synth --sweep $sw --algo silvertorch --out "$R" --resume
done
cp -a "$L/logs/clocks-gpu$GPU.csv" "$L"/logs/synth-st-*.log "$R/logs/"
$PY "$W/docs/artifacts/campaign-v2.5/laion30m/cal_table.py" "$R" | tee "$R/table.md"
run upload-synth upload --results "$R" --path-in-repo campaign-v2.9/laion30m-synth-laion30m-synth --verify
grep -E "MANIFEST|round trip" "$L/logs/upload-synth.log"
echo "$(date -Is) synth-st rc=0"
