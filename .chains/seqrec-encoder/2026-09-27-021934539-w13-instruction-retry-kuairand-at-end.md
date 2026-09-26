---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-021720610-w13-instruction-cleanup-approved.md"
nextStep: "w13: after the cleanup and its squashes, poll used_storage every ~2 min for up to ~20 min; once eval-kuairand drops, retry the KuaiRand eval inputs, then the checkpoint (verified); report."
created: "2026-09-26T23:19:34Z"
---

# W13 instruction: retry the KuaiRand uploads at the end

The squash quota is reclaimed asynchronously. After the approved cleanup and its squashes:
1. Poll `used_storage` (per repo, and the total) about every 2 min for up to about 20 min.
2. When `eval-kuairand` (or the total) drops, retry `eval-data publish kuairand --private` and verify it.
3. Then retry the `sasrec-ssm-logq-d64-trainval` checkpoint and verify it.

If the quota has still not dropped after about 20 min, record the numbers and stop. In both cases, put the
KuaiRand outcome in the cleanup report (still only on the VM, or landed) and update checkpoints.md and
datasets.md only for what landed.
