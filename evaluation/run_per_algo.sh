#!/usr/bin/env bash
# Run every algo in <config> as a separate Python process.
#
# Usage: run_per_algo.sh <config> [extra flags forwarded to `evaluate`]
#
# Writes one JSON per algo to `<cfg.output>/<algo>.json`. Each invocation
# is a fresh Python process, so torch.compile / Triton autotune /
# CUDA-graph state from one algo doesn't leak into the next.
set -euo pipefail

cd "$(dirname "$0")"

cfg="${1:?usage: run_per_algo.sh <config> [extra flags...]}"
shift

mapfile -t info < <(uv run python -c "
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
c = {k: v for k, v in c.items() if not k.startswith('_')}
print(c['output'])
for a in c['algorithms']:
    print(a)
" "$cfg")

out_dir="${info[0]}"
algos=("${info[@]:1}")

mkdir -p "$out_dir"
for algo in "${algos[@]}"; do
    echo "=== $(date -u +%FT%TZ) $algo ==="
    uv run evaluate \
        --config "$cfg" \
        --algo "$algo" \
        --output "$out_dir/$algo.json" \
        "$@" \
    || echo "warn: $algo exited rc=$?"
done
