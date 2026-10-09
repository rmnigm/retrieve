#!/usr/bin/env bash
# pod b, claims first (controller 2026-10-10 09:00): the remaining PubMed filter chunks at seed 0 only (V3 and postfilter,
# 8 chunks) at campaign-v2.3, in the v2.3 worktree and tree; seeds 1-2 go to F-REPRO; V-SEEDS arXiv deferred. Then stop:
# V3-BITS-PUBMED waits for dev/v3-bits-pubmed's merge, V-ROUTER PubMed for the goodreads threshold.
set -u
export VPUBMED_CHUNKS=nost VPUBMED_SEEDS=0
export VPUBMED_DONE=/scratch/campaign-v2.1/results/filter/pubmed-d768.jsonl:/scratch/campaign-v2.2/results/filter/pubmed-d768.jsonl:/scratch/campaign-v2.3/results/filter/pubmed-d768.jsonl
bash /scratch/wt/v-pubmed-v23/docs/artifacts/campaign-v2.3/v-pubmed/driver-chunks.sh >> /scratch/v-pubmed/driver-chunks-v23.log 2>&1
echo "$(date -Is) pubmed seed-0 rc=$?"
