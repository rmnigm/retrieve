#!/usr/bin/env bash
# FULL-RERUN's META-FORK grid at the tag (T3: X parity and speed, adapted against Meta's code as published), seed 0, on the pod's
# own datasets, from the head grid's spec (research/artifacts/meta-fork/grids/head-32c0d61.txt): every line twice, at
# --score-path int32 (the bit-exact gate) and fp16 (Meta's default, the study's score path); arms fork and upstream only (the
# Triton reimplementation is out of the study). Each block under the GPU lock and a 900 s limit, as run_grid.sh; to fr/metafork-<pod>.
# usage: metafork_grid.sh pod1|pode     env as queue.sh
set -u
POD=$1 FR=${FR:-/scratch/fr} PY=$VENV/bin/python LOCK=${LOCK:-/scratch/gpu0.lock}
A=$TREE/research/artifacts/meta-fork O=$FR/out/metafork-$POD; mkdir -p "$O"; [ -f "$O/.done" ] && exit 0
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf RETRIEVE_DATA_ROOT=/data TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/fr-metafork
cd "$TREE/evaluation"
if [ "$POD" = pod1 ]; then KEEP="pubmed"; else KEEP="goodreads arxiv"; fi
while read -r ds dim mode sweeps nl np bss extra; do
  [ -z "$ds" ] || [[ " $KEEP " != *" $ds "* ]] && continue
  for sp in int32 fp16; do
    tag="${ds}_d${dim}_${mode}_$(echo "$sweeps" | tr -c 'a-z0-9' '_')nl${nl}_np${np}_${sp}_$(echo "$extra" | tr -c 'a-z0-9' '_')"
    [ -s "$O/$tag.json" ] && continue
    timeout 900 flock "$LOCK" "$PY" "$A/gate_fork.py" "$ds" "$dim" "$mode" "$sweeps" "$nl" "$np" "$bss" "$O/$tag.json" \
      --k 100,1000 --graph --arms fork,upstream --score-path "$sp" $extra > "$O/$tag.log" 2>&1 || echo "$tag rc=$?" >> "$O/FAILED"
  done
done < "$A/grids/head-32c0d61.txt"
"$PY" -m bench.cli upload --results "$O" --path-in-repo "fr/metafork-$POD" --verify > "$O/upload.log" 2>&1 || echo upload >> "$O/FAILED"
[ -s "$O/FAILED" ] || touch "$O/.done"
