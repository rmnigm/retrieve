---
chain: "seqrec-encoder"
branch: "w1-encoder"
parent: "2026-09-26-140500000-w1-instruction-no-env-dumps.md"
nextStep: "W1: find the collapse mechanism, then run per-position Gate B."
created: "2026-09-26T12:00:00Z"
---

# W1 instruction: shared-K Gate B collapsed, diagnose first

The first Gate B (one shared [256] negative vector per step) ended at 11:16:22Z, but W1's own log-grep wait never woke it. The orchestrator found the GPU idle and sent:

> Result: test ndcg@10 0.0160, recall@100 0.0377, coverage@10 1.6e-5. Near-zero coverage means every user gets the same items: that is a collapse, not just slower learning. Find the mechanism and write it down before launching anything else. Candidates: negatives frozen; gBCE alpha/beta with a shared K; positives colliding with shared negatives; misaligned loss mask/targets; the eval path (normalize, get_output_embeddings). Then run the per-position (published) sampling as Gate B proper, within 0.002 of 0.0813 / 0.1489. Use a watcher that exits when the training PID exits, not one that greps the log.

Outcome: W1 commit e8bb54e (gBCE per-position); the mechanism is in the report note.
