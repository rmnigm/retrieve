#!/usr/bin/env bash
# pod b: V-ROUTER PubMed keep/kill leg (24 cells, controller's go) right after V3-BITS-PUBMED (chain-v3bits) ends with "done".
set -u
P="chain-v3bit""s.sh"
while pgrep -f "$P" >/dev/null; do sleep 20; done
grep -q "chain-v3bits done" /scratch/v3-bits-pubmed/chain.log || { echo "$(date -Is) V3-BITS-PUBMED did not end cleanly: stop"; exit 1; }
bash /scratch/wt/v-router-pubmed/docs/artifacts/campaign-v2.5/v-router-pubmed/driver.sh >> /scratch/v-router-pubmed/driver.log 2>&1
echo "$(date -Is) router rc=$?"
