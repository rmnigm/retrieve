#!/usr/bin/env bash
# pod b: C7 PubMed at v2.8 (-O3 official build): the 641ec3b8 PubMed oracle blobs set aside, the leg, a torch.equal compare.
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/c7-pubmed-v28/docs/artifacts/campaign-v2.8/c7-pubmed/oracle_swap.py"
V27=641ec3b8e09034647f426ca7f68204a1f2866536
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step $SWAP aside $V27
step bash /scratch/wt/c7-pubmed-v28/docs/artifacts/campaign-v2.8/c7-pubmed/driver.sh >> /scratch/c7-pubmed-v28/driver.log 2>&1
step $SWAP compare $V27
echo "$(date -Is) chain-c7-v28 done"
