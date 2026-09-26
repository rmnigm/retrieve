#!/usr/bin/env bash
# Stage every `hub` row of cleanup.tsv out of the commit before H1's `git rm`, one directory
# per Hub subtree, then publish each with `bench upload` (dry run unless PUBLISH=1).
#   docs/artifacts/h1-results-storage/stage_hub.sh [STAGE_DIR]
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
SRC=8c493f2  # staging before H1: the last commit that tracks the files
STAGE=${1:-/scratch/tmp/h1-hub}
TSV=$REPO/docs/artifacts/h1-results-storage/cleanup.tsv
rm -rf "$STAGE" && mkdir -p "$STAGE"
tail -n +2 "$TSV" | awk -F'\t' '$2=="hub"{print $1"\t"$3}' | while IFS=$'\t' read -r path hub; do
  mkdir -p "$STAGE/$(dirname "$hub")"
  git -C "$REPO" show "$SRC:$path" > "$STAGE/$hub"
done
subtrees=(d1-a $(cd "$STAGE" && ls -d artifacts/*))
for sub in "${subtrees[@]}"; do
  args=(--results "$STAGE/$sub" --path-in-repo "$sub")
  if [[ ${PUBLISH:-0} == 1 ]]; then args+=(--verify); else args+=(--dry-run); fi
  (cd "$REPO/evaluation" && uv run --no-sync python -m bench.cli upload "${args[@]}") \
    | grep -E '^[0-9]+ files|CITABLE|MANIFEST.json sha256|round trip|done:'
done
