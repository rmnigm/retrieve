#!/usr/bin/env bash
# A config dir for one NUMBERS leg: the tree's dataset yamls + this plan's suites (+ the picked timed suites once they exist).
# Usage: prep_config.sh TREE OUTDIR   (TREE = the worktree the leg runs from)
set -eu
TREE=$1 OUT=$2 HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for f in "$TREE"/evaluation/config/*.yaml; do [ "$(basename "$f")" = suites.yaml ] || cp "$f" "$OUT/"; done
cat "$HERE/suites.yaml" > "$OUT/suites.yaml"
[ -f "$HERE/suites-timed.yaml" ] && cat "$HERE/suites-timed.yaml" >> "$OUT/suites.yaml"
echo "$OUT/suites.yaml: $(grep -c '^[a-z0-9-]*:' "$OUT/suites.yaml") top-level blocks"
