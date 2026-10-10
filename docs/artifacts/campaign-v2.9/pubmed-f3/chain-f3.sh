#!/usr/bin/env bash
# pod b: PubMed F3 / C6 panel at v2.9: the leg (all5 / c3_journal_reverse k-1000 oracles build fresh at v2.9), then a torch.equal
# compare against the 472f2fc6 blobs set aside earlier (/scratch/oracle-472f2fc6/pubmed-medcpt).
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/pubmed-f3-v29/docs/artifacts/campaign-v2.9/pubmed-f3/oracle_swap.py"
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step bash /scratch/wt/pubmed-f3-v29/docs/artifacts/campaign-v2.9/pubmed-f3/driver.sh >> /scratch/pubmed-f3-v29/driver.log 2>&1
step $SWAP compare 472f2fc68c697179b463d3b5a6b19ade194e6e2f
echo "$(date -Is) chain-f3 done"
