#!/usr/bin/env bash
# Run the benchmark for every per-dim YAML in a suite, then merge the per-config
# JSONs into a single file with a `dim` field on each row.
#
# Usage (run from /workspace/retrieve/evaluation):
#   scripts/sweep_dims.sh 500m   # → results/sweep/500m-all.json
#   scripts/sweep_dims.sh 5b     # → results/sweep/5b-all.json
#   scripts/sweep_dims.sh all    # both
#
# `dim` is parsed from the YAML filename (`<suite>-d<dim>.yaml`). Per-config
# outputs are kept under results/sweep/<config>.json so reruns are cheap when
# only one dim changed.

set -euo pipefail

usage() {
    echo "usage: $0 {500m|5b|all}" >&2
    exit 2
}

suite="${1:-}"
[[ -n "$suite" ]] || usage

run_suite() {
    local s="$1"
    local merged="results/sweep/${s}-all.json"
    mkdir -p results/sweep

    local parts=()
    shopt -s nullglob
    local cfgs=(conf/${s}-d*.yaml)
    shopt -u nullglob
    if [[ ${#cfgs[@]} -eq 0 ]]; then
        echo "no configs matched conf/${s}-d*.yaml" >&2
        return 1
    fi

    for cfg in "${cfgs[@]}"; do
        local name dim out tagged
        name="$(basename "$cfg" .yaml)"
        # filename shape: <suite>-d<dim>[-...].yaml → strip up to "-d", then up to next "-" or "."
        dim="${name#*-d}"
        dim="${dim%%-*}"
        out="results/sweep/${name}.json"
        tagged="results/sweep/${name}.tagged.json"

        echo "=== $cfg → $out (dim=$dim) ===" >&2
        uv run benchmark --config "$cfg" --output "$out"

        # Per-file tag pass: inject dim/config so the merged JSON is self-describing.
        jq --argjson dim "$dim" --arg cfg "$name" \
            '[.[] | . + {dim: $dim, config: $cfg}]' "$out" > "$tagged"
        parts+=("$tagged")
    done

    jq -s 'add' "${parts[@]}" > "$merged"
    echo "wrote $merged ($(jq 'length' "$merged") rows)" >&2
}

case "$suite" in
    500m|5b) run_suite "$suite" ;;
    all)     run_suite 500m; run_suite 5b ;;
    *)       usage ;;
esac
