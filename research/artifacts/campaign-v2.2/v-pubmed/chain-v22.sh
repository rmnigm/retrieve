#!/usr/bin/env bash
# pod b, GO campaign-v2.2 (orchestrator-b 2026-10-09 16:10): after the running v2.1 chunk (bench 142586) exits: end the v2.1
# driver (140573, SIGSTOPped) and its sampler; PubMed oracles at f01255f1 set aside; PubMed SilverTorch at v2.2
# (VPUBMED_CHUNKS=st, /scratch/campaign-v2.2/results); oracle compare; D3 PubMed bloomwidth-timed at v2.2; compare; v2.1
# blobs restored; the rest of PubMed non-SilverTorch at v2.1 (VPUBMED_CHUNKS=nost, --resume); then V-SEEDS arXiv at v2.2
# (arXiv f01255f1 blobs set aside, compared after). Stops on any non-zero step or oracle difference.
set -u
V22=/scratch/wt/v-pubmed-v22/docs/artifacts/campaign-v2.2
SWAP="/venvs/retrieve/bin/python $V22/v-pubmed/oracle_swap.py"
OLD=f01255f106214ec0f540352d0e2a5cd90b84b6a1
AX=/data/arxiv-papers/gt_d128
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
while kill -0 142586 2>/dev/null; do sleep 5; done
rm -f /scratch/gpu0.timed
kill -KILL 140573; kill 141384 2>/dev/null
echo "$(date -Is) v2.1 driver 140573 ended after its running chunk (bench 142586 exited); campaign-v2.2 legs next"
step $SWAP aside $OLD
mkdir -p /scratch/v-pubmed/filter-v22 /scratch/d3-pubmed-timed-v22
VPUBMED_CHUNKS=st step bash $V22/v-pubmed/driver-chunks.sh >> /scratch/v-pubmed/driver-chunks-v22.log 2>&1
step $SWAP compare $OLD
step bash $V22/d3-pubmed-timed/driver.sh >> /scratch/d3-pubmed-timed-v22/driver.log 2>&1
step $SWAP compare $OLD
step $SWAP restore $OLD
VPUBMED_CHUNKS=nost step bash /scratch/wt/v-pubmed/docs/artifacts/campaign-v2/v-pubmed/driver-chunks.sh >> /scratch/v-pubmed/driver-chunks.log 2>&1
step $SWAP aside $OLD $AX
step bash /scratch/wt/v-seeds-ax/docs/artifacts/campaign-v2.1/v-seeds-ax/driver.sh >> /scratch/v-seeds-ax/driver.log 2>&1
step $SWAP compare $OLD $AX
echo "$(date -Is) chain-v22 done"
