#!/bin/bash
# B2 k 1000 moved to pod e (controller): once pod 1's k 100 upload is logged, hold A2-bloom so queue_b2r's k 1000 calls (and its superset upload) skip.
until grep -q "A2-bloom k100 uploaded" /scratch/numbers/p1/logs/driver.log; do sleep 30; done
touch /scratch/numbers/p1/hold-A2-bloom
echo "$(date -Is) A2-bloom: k 1000 moved to pod e; hold-A2-bloom set (k 1000 calls skip on pod 1)" >> /scratch/numbers/p1/logs/driver.log
