#!/usr/bin/env bash
# pod b: C7 PubMed at v2.9 (ST-WIDE) vs Meta -O3: the 78cfbc72 PubMed oracle blobs set aside, the leg, a torch.equal compare.
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/c7-pubmed-v29/docs/artifacts/campaign-v2.9/c7-pubmed/oracle_swap.py"
V28=78cfbc721f0c7ab62831832b3791e1f30dc7357c
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step $SWAP aside $V28
step bash /scratch/wt/c7-pubmed-v29/docs/artifacts/campaign-v2.9/c7-pubmed/driver.sh >> /scratch/c7-pubmed-v29/driver.log 2>&1
step $SWAP compare $V28
echo "$(date -Is) chain-c7-v29 done"
