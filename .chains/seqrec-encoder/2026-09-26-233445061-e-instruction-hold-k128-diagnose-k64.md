---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-222146590-e-instruction-k128-resume-every.md"
nextStep: "e-runs: do NOT start k128. Write the k64 result note, then the bounded diagnosis below, then a diagnosis note whose nextStep is your recommendation for k128. Then idle."
created: "2026-09-26T20:34:45Z"
---

# E-runs instruction: hold k128; diagnose KuaiRand d64 first

R-k64: best val ndcg@10 0.0232 / R@100 0.0104 at **epoch 3**, then declining, with early stop at 13. **Test 0.0046 / 0.0016**,
5× below val. Before spending ~5 h on d128, find out what is going on. State mechanisms with evidence. Bounded:
at most ~45 min, GPU use only for evals.

1. **Protocol:** compare val and test target structure (targets per row: distribution and median; history lengths;
   time span), and how NDCG@10 and R@100 behave with ~246 targets (the R@100 ceiling, the ideal-DCG normalization).
   Is the 5× gap expected from the protocol alone?
2. **Calibration:** a **most-popular baseline** (rank items by train click count, the same for every user) scored
   on val and on test through the same `evaluate` path. A few minutes. If our model is near or below popularity,
   say so plainly.
3. **The curve:** the train loss per epoch, the val ndcg@10 / R@100 / coverage per epoch, and whether the decline
   after epoch 3 comes with collapsing coverage (popularity drift) or with rising coverage (overfitting to the
   tail). Use the logQ/popularity relationship where it helps.
4. **Context:** any A100 gSASRec KuaiRand number in `/workspace/retrieve/.chains/e4-kuairand/` (trajectory only, as far
   as the orchestrator knows).
5. **Recommendation for k128,** exactly one of:
   - (a) run as planned;
   - (b) run with one named change (for example lr, temperature, eval cadence), with the evidence for it;
   - (c) do not run until X.

   The orchestrator decides with the user.

Artifacts: `docs/artifacts/seqrec-encoder/k64-diagnosis/` (small scripts and JSON). Push dev/hstu-runs after the
commit. No code changes to `evaluation/`.
