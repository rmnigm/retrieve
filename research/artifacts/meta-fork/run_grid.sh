#!/usr/bin/env bash
# One gate_fork block per line of the spec file: DATASET DIM MODE SWEEPS N_LISTS NPROBE BSS [extra args...]
set -u
A=/scratch/wt/meta-fork/research/artifacts/meta-fork; L=${GRID:-/scratch/meta-fork/grid}; mkdir -p $L
cd /scratch/wt/meta-fork-build/evaluation
while read -r ds dim mode sweeps nl np bss extra; do
  [ -z "$ds" ] && continue
  tag="${ds}_d${dim}_${mode}_$(echo $sweeps | tr -c "a-z0-9" "_")nl${nl}_np${np}_$(echo $extra | tr -c 'a-z0-9' '_')"
  [ -s $L/$tag.json ] && continue
  timeout 900 $A/gpu.sh 1 $L/$tag.log /venvs/meta-fork/bin/python $A/gate_fork.py $ds $dim $mode $sweeps $nl $np $bss $L/$tag.json --k 100,1000 --graph $extra > /dev/null
  echo "$(date -Is) $tag rc=$? $(ls $L/$tag.json 2>/dev/null)"
done < "$1"
