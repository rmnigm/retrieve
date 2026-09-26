---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-162615552-e-go-e2c-logq.md"
nextStep: "e-runs: stop E2a now, score its best checkpoint on test, write its note; merge dev/hstu (e64ad4b); run E2c (logQ); then E2b (hstu + per-position gBCE); decision note; E3."
created: "2026-09-26T13:56:23Z"
---

# E-runs instruction: stop E2a now, run E2c, then bring back E2b

User, 2026-09-26: "maybe stop the current then, see how e2c performs, then maybe do e2b".

**Why:** at epoch 55, E2a's val is 0.0781 / 0.1433 against Gate B's (gSASRec gBCE) 0.0855 / 0.158 at the
same epoch. It was level until about epoch 17 and has trailed since, so it will not reach the bar.

1. **Stop E2a now.** Kill its PID and make sure the GPU is free. Score its current `best_model.pt` on test
   with the gate_a / E0 re-score path. Write the E2a note: "stopped at epoch N by user decision", best val
   (epoch), test, and the W&B URL. Mark it clearly as not run to completion.
2. **Merge dev/hstu (e64ad4b)** into dev/hstu-runs, re-sync the venv (`uv sync`, not rp-sync), then
   run **E2c** exactly as in the go note (`2026-09-26-162615552-e-go-e2c-logq.md`).
3. **After E2c, run E2b** (reinstated): the HSTU body, `loss=gbce normalize=false num_negatives=256
   gbce_t=0.75`, same epochs, patience and eval cadence, W&B on, run id `e2b-yambda-d64-hstu-gbce`.
   - **Exception:** if E2c beats the yambda d64 bar on test (0.0846 **and** 0.1563), write the decision note
     and go straight to E3 instead; I'll say whether to still run E2b.
   - **Early-cut rule for E2c and E2b:** if by epoch 40 the val ndcg@10 is more than 0.006 below Gate B's
     at the same epoch, stop it, score its best checkpoint on test, and note it. This saves budget for E3.
4. **Decision note:** the highest val ndcg@10 among E2a (partial), E2c and E2b goes to E3.
