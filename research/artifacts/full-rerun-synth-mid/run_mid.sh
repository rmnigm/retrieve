#!/usr/bin/env bash
# SYNTH-MID (trimmed GO): stage 3 -> stage 1 -> picks (pick_mid.py, CPU) -> stage 2; final-freeze, /venvs/fr, seed 0, pod e GPU 0 under the lock
set -u
M=/scratch/fr/synth-mid C=/scratch/fr/synth-mid/config L=/scratch/fr/repair/legrun.sh
say(){ echo "$(date -u +%FT%TZ) synth-mid $*"; }
say "stage 3: probe sweep"
CFG=$C bash $L synth-mid-probesweep-pode 0 timed fr-synth-mid-probesweep-d256 laion30m-synth; say "stage 3 rc=$?"
say "stage 1: curves (bloom, k 100, 4096 / 16384) + V3 pools"
CFG=$C bash $L synth-mid-curve-pode 0 quality fr-synth-mid-curve-d256,fr-synth-mid-v3pool-d256 laion10m-synth laion30m-synth; rc=$?; say "stage 1 rc=$rc"
[ $rc = 0 ] || { say "stage 1 not clean: stage 2 not run"; exit 1; }
until [ -f $M/plan/pick_mid.py ]; do sleep 60; done
cd /scratch/wt/fr/evaluation
/venvs/fr/bin/python $M/plan/pick_mid.py /scratch/fr/out/synth-mid-curve-pode-s0/results.parquet > $M/plan/suites-synth-mid-timed.yaml 2> $M/plan/pick_mid.log || { say "pick_mid FAILED"; exit 1; }
cat $M/plan/suites-synth-mid-timed.yaml >> $C/suites.yaml
RETRIEVE_DATA_ROOT=/data /venvs/fr/bin/python ../research/artifacts/full-rerun/check_suites.py $C $C/suites.yaml --only fr-synth-mid-timed-d256 > $M/plan/check_timed.log 2>&1 || { say "check_suites FAILED on the timed suite"; exit 1; }
say "stage 2: timed"
CFG=$C bash $L synth-mid-timed-pode 0 timed fr-synth-mid-timed-d256 laion10m-synth laion30m-synth; say "stage 2 rc=$?"
say "SYNTH-MID-LEGS-DONE"
