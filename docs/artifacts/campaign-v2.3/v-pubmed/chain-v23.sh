#!/usr/bin/env bash
# pod b, GO campaign-v2.3 (orchestrator-b 2026-10-09 20:40): after the running v2.2 chunk (bench 267715) exits: end the v2.2
# driver (148057, SIGSTOPped) and its sampler; PubMed blobs at 0d23c615 set aside; at v2.3 (/scratch/campaign-v2.3/results):
# the remaining PubMed SilverTorch chunks (not ok at v2.1 / v2.2), oracle compare, D3 PubMed bloomwidth-timed, compare, the
# remaining non-SilverTorch chunks, then V-SEEDS arXiv (arXiv f01255f1 blobs set aside, compared after). Stops on any
# non-zero step or oracle difference.
set -u
V23=/scratch/wt/v-pubmed-v23/docs/artifacts/campaign-v2.3
SWAP="/venvs/retrieve/bin/python $V23/v-pubmed/oracle_swap.py"
V21=f01255f106214ec0f540352d0e2a5cd90b84b6a1
V22=0d23c615a60c6d170fbf3e79c66999406bd5006c
AX=/data/arxiv-papers/gt_d128
export VPUBMED_DONE=/scratch/campaign-v2.1/results/filter/pubmed-d768.jsonl:/scratch/campaign-v2.2/results/filter/pubmed-d768.jsonl
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
while kill -0 267715 2>/dev/null; do sleep 5; done
rm -f /scratch/gpu0.timed
kill -KILL 148057; kill 161991 2>/dev/null
echo "$(date -Is) v2.2 driver 148057 ended after its running chunk (bench 267715 exited); campaign-v2.3 legs next"
step $SWAP aside $V22
mkdir -p /scratch/v-pubmed/filter-v23 /scratch/d3-pubmed-timed-v23
VPUBMED_CHUNKS=st step bash $V23/v-pubmed/driver-chunks.sh >> /scratch/v-pubmed/driver-chunks-v23.log 2>&1
step $SWAP compare $V22
step bash $V23/d3-pubmed-timed/driver.sh >> /scratch/d3-pubmed-timed-v23/driver.log 2>&1
step $SWAP compare $V21
VPUBMED_CHUNKS=nost step bash $V23/v-pubmed/driver-chunks.sh >> /scratch/v-pubmed/driver-chunks-v23.log 2>&1
step $SWAP aside $V21 $AX
step bash /scratch/wt/v-seeds-ax/docs/artifacts/campaign-v2.1/v-seeds-ax/driver.sh >> /scratch/v-seeds-ax/driver.log 2>&1
step $SWAP compare $V21 $AX
echo "$(date -Is) chain-v23 done"
