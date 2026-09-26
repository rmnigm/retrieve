---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-162059107-logq-f1-f2-decided.md"
nextStep: "After E3 frees the GPU, dispatch an opus efficiency worker: profile one HSTU train step with torch.profiler, then apply the top 1-2 overhead fixes, with interleaved before/after epoch times and a val-curve equality check; review gate, then merge."
created: "2026-09-26T14:11:31Z"
---

# Pending: training-overhead step (user asked, scheduled after E3)

User, 2026-09-26: "do we use torch.compile btw or some other overhead reduce tools? maybe also a good
idea to later see if we can implement/test them".

## What is in use now
- `torch.compile` on the dense body only (blocks, final norm, out_proj; default mode, no CUDA graphs).
- bf16 autocast, TF32, fused AdamW, GPU-resident batches, SDPA.
- The loss, negative sampling and embedding lookup run eager.
- Compile was measured once, on gSASRec (~18% faster); it was never measured on HSTU.

## Symptom
E2a (`2026-09-26-165853693-e-e2a-result.md`) ran at 39-42 s/epoch against a 15-25 s prediction.
GPU utilisation was 99% at ~235 W and the training thread at 100% CPU. This suggests launch/CPU
overhead, but it is unprofiled.

## Candidate fixes, in order
1. A profile first.
2. CUDA graphs on the body (`mode="reduce-overhead"`; fixed [B, L] input).
3. Compile the loss path: pad P, or mark it dynamic.
4. Remove host syncs (`.item()` in logging, data-dependent shapes).
5. A fused HSTU-bias attention kernel, only if attention dominates.

## Proof required (contract §5, performance claim)
- The prediction is written before measuring.
- Before/after measured interleaved in one process, with sm_mhz and the unstable flag.
- A short run's val curve matches, so results are unchanged.
- Review gate before merge.
