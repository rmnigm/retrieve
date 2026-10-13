#!/bin/bash
# Copy pod 1's /data/laion30m (files the k-means test reads; not the synth attrs) to pod e, then compare sha256 on both sides.
set -u
S=/data/laion30m D=rp-a100-x1-e:/data/laion30m X=/scratch/numbers/p1/bloom/xfer
F="content_d256 gt_d256 item_attrs_narrow.pt clause_is_reverse_narrow.pt eval_split.parquet heldout.parquet prep_log.json vocab.json"
ssh -o BatchMode=yes rp-a100-x1-e "mkdir -p /data/laion30m"
( cd $S && rsync -a --partial $F $D/ ) > $X/rsync.log 2>&1 || { echo "rsync failed" >> $X/rsync.log; exit 1; }
( cd $S && find $F -type f | sort | xargs sha256sum ) > $X/sha-pod1.txt
ssh -o BatchMode=yes rp-a100-x1-e "cd /data/laion30m && find $F -type f | sort | xargs sha256sum" > $X/sha-pode.txt
cmp -s $X/sha-pod1.txt $X/sha-pode.txt && echo "COPY OK $(wc -l < $X/sha-pod1.txt) files sha256-equal" > $X/result.txt || echo "SHA MISMATCH" > $X/result.txt
