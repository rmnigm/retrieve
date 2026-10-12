#!/usr/bin/env bash
# V-AX-CORR, claims first (controller 2026-10-10 09:00): only the arms whose recall the cluster-correlated pass set moves,
# seed 0: V3 (C2's pre-score recall), SilverTorch triton clause over its n_probe sweep (C6, the local-pass-rate take),
# postfilter. 42 of the suite's 198 cells; V1 / V2 (exact, cost set by the pass count alone), bloom SilverTorch and
# seeds 1-2 decide no claim here. One tree, uploaded as campaign-v2.5/arxiv-corr-synth-synth (a narrowed leg).
#   mkdir -p /scratch/v25 && GPU=0 setsid nohup flock -n /scratch/gpu0.lock bash claims.sh > /scratch/v25/v-ax-corr.driver.log 2>&1 &
export GPU=${GPU:-0} LEG=v-ax-corr DS=arxiv-corr-synth
R=/scratch/campaign-v25/$DS-synth
. "$(dirname "$(readlink -f "$0")")/../pod-d/common.sh"
old_oracles $DS
step check check --dataset $DS --dim 128
step oracle oracle --dataset $DS --suite synth --dim 128
NARROW="--seed 0" stream $DS synth "linr_v3|triton"
NARROW="--seed 0 --filter-kind clause" stream $DS synth "silvertorch|triton"
NARROW="--seed 0" stream $DS synth "postfilter|torch"
upload campaign-v2.5/$DS-synth
msg="v-ax-corr (claims-first, 42 cells: V3, SilverTorch triton clause, postfilter, seed 0) done on pod d GPU $GPU at $cv; Hub campaign-v2.5/$DS-synth: $UP"
note pod-d "$msg"; note control "$msg"
finish
