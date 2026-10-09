#!/usr/bin/env bash
# Night queue item 1 (C1 + F1 at 30 M, low / mid rates): laion30m-synth V1 + V2 triton at p 0.001 / 0.003 / 0.01 / 0.03 / 0.05, bs 1 + 16, graph + eager,
# seed 0, k 100, at campaign-v2.9 on pod d GPU 1, one process (H-ARMFREE). The rates are in the 30 M synth attrs (SYNTH-TRIM's ten); the suite
# `v1v2-low` lives in a scratch config dir. laion30m's oracle blobs are not moved (laion's GPU-0 job reads the same gt_d256).
# Hub campaign-v2.9/laion30m-synth-v1v2-low.
#   GPU=1 setsid nohup flock -n /scratch/gpu1.lock bash driver.sh > /scratch/v29/laion30m-synth-v1v2-low.driver.log 2>&1 &
export TAG=campaign-v2.9 LEG=laion30m-synth-v1v2-low REPO=/scratch/wt/v29-axsynth PY=/venvs/d-run-v29-ax/bin/python GPU=${GPU:-1}
DS=laion30m-synth SUITE=v1v2-low
R=/scratch/campaign-v29/$DS-$SUITE
CFG=/scratch/v29/v1v2-low-config
. "$(dirname "$(readlink -f "$0")")/../../campaign-v2.5/pod-d/common.sh"
rm -rf "$CFG" && cp -r "$REPO/evaluation/config" "$CFG"
printf 'v1v2-low:\n  datasets: [laion30m-synth]\n  dims: [256]\n  filter_kinds: [clause]\n  ks: [100]\n  batch_sizes: [1, 16]\n  seeds: [0]\n  sweeps:\n    laion30m-synth: [p0001, p0003, p001, p003, p005]\n  interleave:\n    - {by: algo, values: [linr_v1_filter_mask, linr_v2]}\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [triton]}\n    - {algo: linr_v2, backends: [triton]}\n' >> "$CFG/suites.yaml"
step check check --dataset $DS --dim 256
step oracle oracle --dataset $DS --suite $SUITE --dim 256 --config-dir "$CFG"
NARROW="--config-dir $CFG" stream $DS $SUITE "linr_v1_filter_mask linr_v2|triton"
n=$($PY -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; print(sum(r["status"]=="ok" for r in rs), sum(r["status"]!="ok" for r in rs))' "$R/$SUITE/$DS-d256.jsonl")
echo "$(date -Is) ok / not ok: $n"
[ "$n" = "10 0" ] || { note pod-d "$LEG: ok / not-ok $n; driver stopped"; exit 6; }
upload campaign-v2.9/$DS-$SUITE
msg="$LEG (10 cells) done on pod d GPU $GPU at $cv; Hub campaign-v2.9/$DS-$SUITE: $UP"
note pod-d "$msg"; note control "$msg"
finish
