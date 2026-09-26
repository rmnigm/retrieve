---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-021934539-w13-instruction-retry-kuairand-at-end.md"
nextStep: "w13: do NOT upload the KuaiRand checkpoint at all; retry only the KuaiRand eval inputs after the quota drops; document the checkpoint as not kept."
created: "2026-09-26T23:20:12Z"
---

# W13 instruction: drop the KuaiRand checkpoint entirely (user)

User, 2026-09-27: "fuck the big kuairand checkpoint, having space for evals is more important. drop it altogether".
- **Never upload `sasrec-ssm-logq-d64-trainval`.** Remove the local symlink under `/data/kuairand/checkpoints/`
  so that nothing picks it up. Do not delete `/scratch/ckpt/*`: the VM is being deleted anyway.
- The retry at the end covers **only** `eval-data publish kuairand --private`.
- **Docs, on dev/hstu-hub:**
  - checkpoints.md: no KuaiRand row;
  - datasets.md § kuairand: the d64 refit checkpoint was not kept (the user chose eval space);
    retraining it is `train run` with the flags in `docs/artifacts/seqrec-encoder/k64-refit-sasrec-ssm-logq/command.sh`,
    about 20 min on the H100.
  - The results in validation.md stay as they are, since they are measured records.
