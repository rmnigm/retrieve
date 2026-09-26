---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-173224012-e-e2c-result.md"
nextStep: "Orchestrator decides: add E1c (sasrec d64 published body + sampled softmax in-batch 4096 + uniform 8192 + logQ, ~11 s/epoch, <=25 min) after E2b, before the decision note? e-runs will not launch it without a go note."
created: "2026-09-26T14:41:44Z"
---

# Proposal (from the user, in the e-runs session): sasrec body + sampled softmax with logQ

User, 2026-09-26: "with logq, maybe the issue is the architecture? i think normal transformer but
with that loss will actually be good also."

Why the current runs cannot answer it: the loss and the body are confounded. E1 is sasrec without
logQ (val 0.0740); E2c is HSTU with logQ (val 0.0921). Gate B, sasrec with gBCE, reached 0.0913
val (published 0.0920). So HSTU+logQ is level with sasrec+gBCE on val so far, not clearly ahead.

Run (E1c): E1's command.sh plus `logq=true`, run id e1c-yambda-d64-sasrec-ssm-logq, W&B on.
Cost ~11 s/epoch, 100 epochs max = ~20-25 min, the cheapest run in the set. Selection would stay
the highest val ndcg@10 across E2a/E2b/E2c/E1c for E3. Not launched; E2b holds the GPU.
