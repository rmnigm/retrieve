#!/usr/bin/env bash
# A tagged library as a renamed package (op namespace renamed too), importable beside the working tree's
# `retrieve` so before/after run in one process: make_pkg.sh <repo> <tag> <name> <out_dir>
# Quoted module paths are renamed too: `interfaces.DISPATCH` and `ops.__getattr__` import the backends by
# string, and before this rule a renamed package's modules dispatched to the current tree's ops.
set -eu
repo=$1 tag=$2 name=$3 out=$4
rm -rf "$out/$name" && mkdir -p "$out"
tmp=$(mktemp -d); git -C "$repo" archive "$tag" retrieve/src/retrieve | tar -x -C "$tmp" --strip-components=2
mv "$tmp/retrieve" "$out/$name" && rmdir "$tmp"
grep -rlE '\bretrieve(\.|::)' "$out/$name" --include=*.py | xargs sed -i -E \
  -e "s/(from|import) retrieve(\.| )/\1 $name\2/g" -e "s/\"retrieve::/\"$name::/g" \
  -e "s/torch\.ops\.retrieve\b/torch.ops.$name/g" -e "s/([\"'])retrieve\./\1$name./g"
