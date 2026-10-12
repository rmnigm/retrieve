#!/usr/bin/env bash
# The campaign-v2.1 library as package `retrieve_v21` (op namespace `retrieve_v21::`), importable beside the
# working tree's `retrieve` so before/after run in one process: make_v21_pkg.sh <repo> <out_dir>
set -eu
repo=$1 out=$2
rm -rf "$out/retrieve_v21" && mkdir -p "$out"
git -C "$repo" archive campaign-v2.1 retrieve/src/retrieve | tar -x -C "$out" --strip-components=2
mv "$out/retrieve" "$out/retrieve_v21"
grep -rlE '\bretrieve(\.|::)' "$out/retrieve_v21" --include=*.py | xargs sed -i -E \
  -e 's/(from|import) retrieve(\.| )/\1 retrieve_v21\2/g' -e 's/"retrieve::/"retrieve_v21::/g' \
  -e 's/torch\.ops\.retrieve\b/torch.ops.retrieve_v21/g'
