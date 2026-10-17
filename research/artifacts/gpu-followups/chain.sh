#!/bin/bash
# GPU queue: run a (non-LAION), run a (LAION), then whatever later stages are appended to /scratch/gpu-followups/queue.d/
cd /scratch/wt/laion-inputs/evaluation
A=/scratch/wt/laion-inputs/research/artifacts/gpu-followups/run_a.sh
CELLS="goodreads:gf-a-d128 goodreads-synth:gf-a-d128 arxiv:gf-a-d128 arxiv-synth:gf-a-d128 pubmed:gf-a-d768 pubmed-synth:gf-a-d768" \
  OUT=/scratch/gpu-followups/a/out bash $A > /scratch/gpu-followups/a/run-a-1.log 2>&1
touch /scratch/gpu-followups/a/STAGE1-DONE
CELLS="laion1m:gf-a-d256 laion3m:gf-a-d256 laion10m:gf-a-d256 laion30m:gf-a-d256 laion30m-synth:gf-a-d256" \
  OUT=/scratch/gpu-followups/a/out bash $A > /scratch/gpu-followups/a/run-a-2.log 2>&1
touch /scratch/gpu-followups/a/STAGE2-DONE
mkdir -p /scratch/gpu-followups/queue.d
# later stages: each executable dropped into queue.d runs in name order, once
while true; do
  next=$(ls /scratch/gpu-followups/queue.d/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then
    [ -f /scratch/gpu-followups/queue.d/STOP ] && break
    sleep 10; continue
  fi
  mv "$next" "$next.running"
  bash "$next.running" > "${next%.sh}.log" 2>&1
  mv "$next.running" "$next.done"
done
