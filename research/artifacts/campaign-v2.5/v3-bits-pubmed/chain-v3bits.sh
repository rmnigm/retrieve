#!/usr/bin/env bash
# pod b: after chain g (the 8 seed-0 PubMed chunks at v2.3) ends rc 0: set the v2.3 (1258a63e) PubMed oracle blobs aside,
# run V3-BITS-PUBMED at campaign-v2.5 (472f2fc6; oracles rebuilt), compare with torch.equal. Stops on any failure / DIFFER.
set -u
SWAP="/venvs/retrieve/bin/python /scratch/wt/v-pubmed-v23/docs/artifacts/campaign-v2.3/v-pubmed/oracle_swap.py"
V23=1258a63e7ebd170d7270111b5206b4ea0b8ca7f9
step() { "$@"; local rc=$?; echo "$(date -Is) step ${*:1:4} rc=$rc"; [ $rc -eq 0 ] || exit $rc; }
P="g/chain-claim""s.sh"
while pgrep -f "$P" >/dev/null; do sleep 20; done
grep -q "pubmed seed-0 rc=0" /scratch/v-pubmed/g/chain-claims.log || { echo "$(date -Is) chain g did not end rc=0: stop"; exit 1; }
step $SWAP aside $V23
step bash /scratch/wt/v3-bits-run/docs/artifacts/campaign-v2.5/v3-bits-pubmed/driver.sh >> /scratch/v3-bits-pubmed/driver.log 2>&1
step $SWAP compare $V23
echo "$(date -Is) chain-v3bits done"
