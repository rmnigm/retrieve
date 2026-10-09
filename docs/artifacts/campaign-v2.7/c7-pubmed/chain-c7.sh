#!/usr/bin/env bash
# pod b: C7 PubMed at v2.7: the 472f2fc6 PubMed oracle blobs set aside, the leg, then a torch.equal compare (stops on DIFFER).
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/c7-pubmed-v27/docs/artifacts/campaign-v2.7/c7-pubmed/oracle_swap.py"
V25=472f2fc68c697179b463d3b5a6b19ade194e6e2f
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step $SWAP aside $V25
step bash /scratch/wt/c7-pubmed-v27/docs/artifacts/campaign-v2.7/c7-pubmed/driver.sh >> /scratch/c7-pubmed/driver.log 2>&1
step $SWAP compare $V25
echo "$(date -Is) chain-c7 done"
