#!/usr/bin/env bash
# Final pass, pod d staging (recipe stage a100-x2-d; controller 2026-10-13: d-run owns it, laion waits for the done line): at the tag campaign-v2.12,
# the recipe's worktree /scratch/wt/final + the -O3 venv /venvs/final-o3, then every pre-tag oracle blob of pod d's datasets moved aside
# (move_old_oracles.py: moved, not deleted), `bench check` per dataset, and `bench oracle` once per (dataset, suite) that pod d's d0 / d1 legs run,
# on GPU 1. Writes the PubMed manifest sha and the done line to .chains/campaign-final/.
#   flock /scratch/gpu1.lock bash stage-a100-x2-d.sh
set -u
TAG=campaign-v2.12; WT=/scratch/wt/final; V=/venvs/final-o3; PY=$V/bin/python
NOTE=/workspace/retrieve/.chains/campaign-final/$(date -u +%Y-%m-%d-%H%M%S000)-d-run-stage-a100-x2-d.md
log() { echo "$(date -Is) $*"; }
git -C /workspace/retrieve fetch -q origin --tags
git -C /workspace/retrieve rev-parse -q --verify "refs/tags/$TAG" > /dev/null || { log "no tag $TAG yet, refusing"; exit 2; }
LIB=$(git -C /workspace/retrieve rev-parse "$TAG:retrieve/src/retrieve")
[ -d $WT ] || git -C /workspace/retrieve worktree add -q --detach $WT $TAG
mkdir -p $WT/articles; [ -e $WT/evaluation/data ] || ln -s /data $WT/evaluation/data
[ "$(git -C $WT rev-parse HEAD:retrieve/src/retrieve)" = "$LIB" ] || { log "worktree is not at $TAG's library, refusing"; exit 2; }
[ -x $PY ] || (cd $WT && UV_PROJECT_ENVIRONMENT=$V taskset -c 0-63 bash scripts/build_official_o3.sh) > /scratch/final-stage-build.log 2>&1
cd $WT/evaluation
$PY -c 'import retrieve, sys; sys.exit(0 if retrieve.__file__.startswith("/scratch/wt/final/") else 1)' || { log "retrieve not imported from $WT, refusing"; exit 2; }
CV=$($PY -m bench.cli env | $PY -c 'import json,sys; print(json.load(sys.stdin)["code_version"])')
[ "$CV" = "$LIB" ] || { log "code_version $CV != $LIB, refusing"; exit 2; }
SO=$($PY -m bench.cli env | $PY -c 'import json,sys; b=json.load(sys.stdin).get("official_build") or {}; print(b.get("so_sha256"), b.get("nvcc_append_flags"))')
log "tree $WT at $TAG, library $LIB, official $SO"
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/scratch/hf TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/final-stage-gpu1
PIN="taskset -c 96-127"
$PY ../docs/artifacts/campaign-v2/v-pod1-run/move_old_oracles.py "$LIB" goodreads goodreads-synth arxiv arxiv-synth arxiv-corr-synth yfcc10m yfcc10m-synth pubmed laion30m laion30m-synth \
  || { log "move_old_oracles failed"; exit 3; }
for dd in goodreads:128 goodreads-synth:128 arxiv:128 arxiv-synth:128 arxiv-corr-synth:128 yfcc10m:192 yfcc10m-synth:192 pubmed:768 laion30m:256 laion30m-synth:256; do
  $PY -m bench.cli check --dataset ${dd%%:*} --dim ${dd#*:} || { log "check ${dd%%:*} failed"; exit 4; }
done
for p in arxiv:filter:128 goodreads:filter:128 yfcc10m:filter:192 pubmed:filter:768 goodreads-synth:synth:128 arxiv-synth:synth:128 yfcc10m-synth:synth:192 \
         arxiv-corr-synth:synth:128 laion30m:laion30m:256 laion30m:laion30m-x:256 goodreads:deep:128 arxiv:deep:128 yfcc10m:deep:192 pubmed:deep:768 \
         laion30m:laion30m-bs1:256 laion30m-synth:laion30m-synth:256 goodreads:h2h:128 arxiv:h2h:128 laion30m:c7-scorepath:256; do
  IFS=: read -r ds su dm <<< "$p"; t0=$(date +%s)
  $PIN $PY -m bench.cli oracle --dataset $ds --suite $su --dim $dm > /scratch/final-stage-oracle-$ds-$su.log 2>&1 < /dev/null
  rc=$?; log "oracle $ds/$su rc=$rc s=$(( $(date +%s) - t0 ))"; [ $rc -eq 0 ] || exit 5
done
PM=$(cut -c1-64 /scratch/v211/pubmed-manifest.sha 2>/dev/null)
mkdir -p "$(dirname "$NOTE")"
printf -- '---\nchain: "campaign-final"\nbranch: "d-run"\ncreated: "%s"\n---\n\n# stage-a100-x2-d done (d-run)\n\n**Never print secrets** (AGENTS.md rule 9).\n\n- Tree `%s` at `%s`, library `%s`; venv `%s` (official `%s`).\n- Pre-tag oracle blobs moved aside (`gt_*/before-%s/`); `bench check` ok on goodreads, goodreads-synth, arxiv, arxiv-synth, arxiv-corr-synth, yfcc10m, yfcc10m-synth, pubmed, laion30m, laion30m-synth; `bench oracle` rebuilt for every (dataset, suite) of the d0 / d1 legs.\n- PubMed (`/data/pubmed-medcpt`) file manifest sha256 `%s` (`/scratch/v211/pubmed-manifest.txt`).\n\n**DONE: pod d staged at %s; laion may start d0 legs (after the GO).**\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WT" "$TAG" "$LIB" "$V" "$SO" "${LIB:0:8}" "$PM" "$TAG" > "$NOTE"
log "stage done; note $NOTE"
