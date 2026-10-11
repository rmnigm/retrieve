#!/usr/bin/env bash
# Night queue item 2 (C1 at 0.8 M at one tag): goodreads-synth V1 + V2 triton clause at p 0.001 / 0.003 / 0.01 / 0.03 / 0.05 / 0.1 / 0.2 / 0.5 / 1, bs 1 + 16,
# graph + eager, seed 0, k {100, 1000} (k 100 only where N·p < 4,000, as the synth suite), at campaign-v2.9 on pod d GPU 1, one process.
# The suite `v1v2` lives in a scratch config dir. Hub campaign-v2.9/goodreads-synth-v1v2.
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v29/goodreads-synth-v1v2.driver.log 2>&1 &
export TAG=campaign-v2.9 LEG=goodreads-synth-v1v2 REPO=/scratch/wt/v29-axsynth PY=/venvs/d-run-v29-ax/bin/python GPU=${GPU:-1}
DS=goodreads-synth SUITE=v1v2
R=/scratch/campaign-v29/$DS-$SUITE
CFG=/scratch/v29/v1v2-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG"
printf 'v1v2:\n  datasets: [goodreads-synth]\n  dims: [128]\n  filter_kinds: [clause]\n  ks: [100, 1000]\n  batch_sizes: [1, 16]\n  seeds: [0]\n  sweeps:\n    goodreads-synth: [p0001, p0003, p001, p003, p005, p01, p02, p05, p1]\n  ks_by_sweep:\n    goodreads-synth: {p0001: [100], p0003: [100]}\n  interleave:\n    - {by: algo, values: [linr_v1_filter_mask, linr_v2]}\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [triton]}\n    - {algo: linr_v2, backends: [triton]}\n' >> "$CFG/suites.yaml"
step check check --dataset $DS --dim 128
step oracle oracle --dataset $DS --suite $SUITE --dim 128 --config-dir "$CFG"
NARROW="--config-dir $CFG" stream $DS $SUITE "linr_v1_filter_mask linr_v2|triton"
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["status"]=="ok" for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d128.jsonl")
echo "$(date -Is) ok / not ok: $n"
[ "$n" = "18 0" ] || { note pod-d "$LEG: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.9/$DS-$SUITE
msg="$LEG (18 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.9/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
