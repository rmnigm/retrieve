#!/usr/bin/env bash
# Night-queue item 8, phase 2: the timed frontier candidates (tune-frontier-candidates.json, picked from phase 1's recall) on YFCC 10 M at
# campaign-v2.9: tune_config.py's `tune` suite, bs {1, 16}, eager + graph, --skip-quality (recall from phase 1, same code and cells).
LEG=tune-v29
TAG=${TAG:-campaign-v2.9}
REPO=${REPO:-/scratch/wt/tree-v29}
PY=${PY:-/venvs/v29/bin/python}
. "$(dirname "$(readlink -f "$0")")/common.sh"
C=$LOG/tune-config-timed
$PY "$HERE/tune_config.py" "$C" "$HERE/tune-frontier-candidates.json" || exit 1
for ds in yfcc10m yfcc10m-synth; do
  step "t-$ds" run --config-dir "$C" --dataset $ds --dim 192 --suite tune --seed 0 --skip-quality --resume --out "$R"
done
finish
