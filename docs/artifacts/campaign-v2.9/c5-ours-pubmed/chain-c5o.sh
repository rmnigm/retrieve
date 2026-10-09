#!/usr/bin/env bash
# pod b: C5 ours PubMed at v2.9: the leg (its k-100 c0_mesh oracle builds fresh at v2.9), then a torch.equal compare against the
# 472f2fc6 k-100 blob set aside by the v2.7 C7 chain (/scratch/oracle-472f2fc6/pubmed-medcpt).
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/c5-ours-v29/docs/artifacts/campaign-v2.9/c5-ours-pubmed/oracle_swap.py"
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step bash /scratch/wt/c5-ours-v29/docs/artifacts/campaign-v2.9/c5-ours-pubmed/driver.sh >> /scratch/c5-ours-v29/driver.log 2>&1
step $SWAP compare 472f2fc68c697179b463d3b5a6b19ade194e6e2f
echo "$(date -Is) chain-c5o done"
