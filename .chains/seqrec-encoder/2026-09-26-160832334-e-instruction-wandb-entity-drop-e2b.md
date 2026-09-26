---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-160718058-e-instruction-split-e2.md"
nextStep: "e-runs: restart E2a now with W&B on (WANDB_ENTITY=pinkmeme); drop E2b; after E2a, wait for the orchestrator's go on E2c (logQ, code in progress)."
created: "2026-09-26T13:08:32Z"
---

# E-runs instruction: W&B entity pinkmeme; E2b dropped; E2c (logQ) follows

User, 2026-09-26: the W&B username is **pinkmeme**. They also said: "maybe just do ssm with logq,
rather than trying to understand too deep".

- **Restart E2a now** with `WANDB_ENTITY=pinkmeme` exported for the job, plus
  `wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=e2a-yambda-d64-hstu-ssm-uniform`.
  E2a is only minutes in, so kill it and relaunch; do not keep two jobs on the GPU. The config is
  otherwise unchanged (hstu body, sampled softmax, `inbatch_negatives=0`).
- **E2b (hstu + gBCE) is dropped.**
- **E2c:** HSTU + sampled softmax with in-batch 4096 + uniform 8192 **and logQ**. A fix agent is
  adding `logq` to the trainer now. When E2a finishes, write its note and wait for the
  orchestrator's go: a new chain note naming the merged commit to pull into `dev/hstu-runs`.
- **Decision:** the higher val ndcg@10 of E2a and E2c goes to E3.
- Record the W&B URLs. Never print the key.
