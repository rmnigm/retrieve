#!/usr/bin/env bash
# Roadmap D1-E: pubmed `filter`, the targeted rerun, on the restaged slice (artifacts/pubmed-restage).
#   d1e_driver.sh check   encode_queries, bench check, the V1 c0_mesh identity cell vs the d1/pubmed record
#   d1e_driver.sh run     the three narrow runs, sequential, rc per command
# From the main checkout's evaluation/, /venvs/retrieve, one GPU; logs under /scratch/d1e.
set -uo pipefail
cd /workspace/retrieve/evaluation
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1e HF_HOME=/scratch/hf
PY=/venvs/retrieve/bin/python
L=/scratch/d1e
OLD=${D1_PUBMED:-/scratch/h2-hub/pubmed/filter/pubmed-d768.jsonl}
mkdir -p $L
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/driver.log; }

cv=$($PY -c "from bench.measure import code_version; print(code_version())")
log "code_version $cv"
[[ $cv == c0e42d1* ]] || { log "ABORT: code_version is not c0e42d1"; exit 2; }

case "${1:-}" in
check)
  $PY -m eval_datasets.etl.pubmed encode_queries > $L/encode_queries.log 2>&1
  log "encode_queries rc=$?"
  $PY -m bench.cli check --dataset pubmed > $L/bench-check.log 2>&1
  rc=$?; log "bench check rc=$rc"; [[ $rc == 0 ]] || exit 1
  $PY -m bench.cli run --dataset pubmed --dim 768 --suite filter --algo linr_v1_filter_mask \
    --backend triton --filter-kind clause --sweep c0_mesh --seed 0 --mode eager --skip-perf \
    --out /scratch/d1e-check --config-dir config > $L/check-cell.log 2>&1
  log "identity cell rc=$?"
  $PY - "$OLD" /scratch/d1e-check/filter/pubmed-d768.jsonl <<'EOF' 2>&1 | tee -a $L/driver.log
import json, sys
def cell(path):
    for line in open(path):
        r = json.loads(line)
        if (r["algo"], r["backend"], r["filter_kind"], r["sweep"], r["seed"]) == (
            "linr_v1_filter_mask", "triton", "clause", "c0_mesh", 0) and r["quality"]:
            return r
old, new = cell(sys.argv[1]), cell(sys.argv[2])
diff = {f"{s}.{m}": (old["quality"][s][m], v) for s in ("oracle", "heldout")
        for m, v in new["quality"][s].items() if old["quality"][s][m] != v}
print("held-out recall@100", new["quality"]["heldout"]["recall@100"],
      "oracle recall@100", new["quality"]["oracle"]["recall@100"])
print("IDENTITY", "EQUAL" if not diff else f"DIFFERS on {len(diff)} metrics: {diff}")
sys.exit(1 if diff else 0)
EOF
  rc=$?; log "identity check rc=$rc"; exit $rc ;;
run)
  R="--dataset pubmed --dim 768 --suite filter --out /workspace/retrieve/evaluation/results --config-dir config --resume"
  $PY -m bench.cli run $R --algo silvertorch --backend triton > $L/silvertorch-triton.log 2>&1
  log "silvertorch/triton rc=$?"
  $PY -m bench.cli run $R --algo linr_v3 --backend triton > $L/linr_v3.log 2>&1
  log "linr_v3 rc=$?"
  $PY -m bench.cli run $R --algo linr_v2 --backend triton --filter-kind bloom --sweep c0c2 --seed 0 \
    > $L/linr_v2-bloom-c0c2.log 2>&1
  log "linr_v2 bloom c0c2 rc=$?" ;;
*) echo "usage: $0 check|run"; exit 2 ;;
esac
