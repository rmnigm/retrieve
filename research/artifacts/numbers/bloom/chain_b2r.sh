#!/bin/bash
# Waits for queue_b2.sh to exit (it stops after pubmed's OOM-failed cells), then starts queue_b2r.sh on the same frozen plan.
while pgrep -f "bash queue_b2.sh" > /dev/null; do sleep 30; done
echo "$(date -Is) A2-bloom: queue_b2 ended; restarting as queue_b2r (per-dataset OOM split)" >> /scratch/numbers/p1/logs/driver.log
exec bash /scratch/numbers/p1/bloom/queue_b2r.sh /scratch/numbers/p1/plan-2b3f662
