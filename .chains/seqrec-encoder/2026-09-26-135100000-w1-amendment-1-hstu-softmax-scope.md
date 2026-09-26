---
chain: "seqrec-encoder"
branch: "w1-encoder"
parent: "2026-09-26-134930000-w1-brief-encoder-trainer-rewrite.md"
nextStep: "W1: apply the amendment and continue with Gate A/B."
created: "2026-09-26T10:51:00Z"
---

# W1 amendment 1: 12 h budget, softmax HSTU only (sent mid-run)

User: ~12 h total budget, one best-believed combination, "I believe in HSTU, but not in
attention without softmax". Scope is now smaller:

- **Blocks: only `sasrec` and `hstu`.** Delete `LlamaBlock` and `attn_bias` from the plan.
- **`HSTUBlock` keeps softmax.** `n = norm(x)`; `U,V,Q,K = SiLU(f1(n)).chunk(4)`;
  `A = softmax(QK^T/sqrt(hd) + rab + mask)` via `F.scaled_dot_product_attention` with the
  bias as a float additive mask; `y = f2(norm(A V) * U)`; residual. No pointwise SiLU/n
  attention, no flag for it. `rab` = relative position buckets + log-scaled time-delta buckets
  (`use_time`). Pre-norm RMSNorm is fine; dropout on attention output.
- **Loss:** `gbce` (Gate B only) and `sampled_softmax` with normalized embeddings (A1),
  temperature 0.05, candidates = M in-batch positives + K uniform, accidental hits masked.
  **Drop logQ** (A2) entirely: no config field, no code.
- **Drop row-wise Adagrad and the bf16 table** (A5): not in this budget. Plain fused AdamW.
- `torch.compile` on the dense body stays (measure it once in Gate B or right after:
  epoch time with and without; keep it only if faster).
- Loss test: sampled-softmax vs direct `F.cross_entropy` on a tiny case (no logQ case).
- Priority: get Gate A + Gate B + a working `hstu` path merged fast; the experiments that
  follow are yambda-500m first. Hand back as soon as Gate B is done.
