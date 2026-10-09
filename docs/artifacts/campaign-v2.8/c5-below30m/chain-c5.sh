#!/usr/bin/env bash
# pod b: C5 below 30 M at v2.8: the 472f2fc6 goodreads / arXiv d128 oracle blobs set aside, the leg, torch.equal compares.
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/c5-below30m-v28/docs/artifacts/campaign-v2.8/c5-below30m/oracle_swap.py"
V25=472f2fc68c697179b463d3b5a6b19ade194e6e2f
GR=/data/goodreads-work-id/gt_d128
AX=/data/arxiv-papers/gt_d128
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
step $SWAP aside $V25 $GR
step $SWAP aside $V25 $AX
step bash /scratch/wt/c5-below30m-v28/docs/artifacts/campaign-v2.8/c5-below30m/driver.sh >> /scratch/c5-below30m/driver.log 2>&1
step $SWAP compare $V25 $GR
step $SWAP compare $V25 $AX
echo "$(date -Is) chain-c5 done"
