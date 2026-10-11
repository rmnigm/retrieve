#!/usr/bin/env bash
# Grid item 9 (tail leg T9, README.md): arXiv at d768 next to the staged d128 / d256, same items, filters and queries.
# Usage: dsweep.sh encode DATA_ROOT SCRATCH      (CPU prep + one GPU encode; from evaluation/ of the leg tree)
#        dsweep.sh config CFG_DIR                (after prep_config.sh: arxiv.yaml gains dim 768 -> ../arxiv-d768/content)
set -euo pipefail
case $1 in
encode)
  DATA=$2 S=$3                       # DATA = the box's /data (evaluation/data -> it); S = scratch for the raw text
  STAGED=$DATA/arxiv-papers OUT=$DATA/arxiv-d768
  # download -> convert -> prep -> encode_text -> encode_queries -> attrs, all into scratch, encoded once at the native 768
  uv run eval-data arxiv all --output-dir "$S/arxiv-rebuild" --truncate-dims 768 --batch-size 512
  # the rebuilt rows must be the staged rows, or every d768 vector is misaligned: stop here if not
  for f in item_id_map.json heldout.parquet; do
    cmp -s "$S/arxiv-rebuild/$f" "$STAGED/$f" || { echo "ABORT: rebuilt $f differs from the staged one (upstream drift)"; exit 1; }
  done
  mkdir -p "$OUT"
  for f in "$STAGED"/*; do case $(basename "$f") in content*|gt_*) ;; *) ln -sfn "$f" "$OUT/";; esac; done
  ln -sfn "$S/arxiv-rebuild/content" "$OUT/content"   # the staged attrs / eval_split stay; only the vectors are new
  ls -la "$OUT/content"
  ;;
config)
  Y=$2/arxiv.yaml
  sed -i 's/^dims: \[64, 128, 256\]/dims: [64, 128, 256, 768]/; s#^content_dir: {64: content_d64, 128: content_d128, 256: content}#content_dir: {64: content_d64, 128: content_d128, 256: content, 768: ../arxiv-d768/content}#' "$Y"
  grep -q '768: ../arxiv-d768/content' "$Y" && grep -q '^dims: \[64, 128, 256, 768\]' "$Y" || { echo "ABORT: arxiv.yaml not patched"; exit 1; }
  ;;
esac
