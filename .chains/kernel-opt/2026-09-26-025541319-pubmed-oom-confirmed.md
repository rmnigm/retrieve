---
chain: "kernel-opt"
branch: "main"
parent: "2026-09-26-024752103-kernel-opt-done-handoff.md"
nextStep: "Decide whether the Triton kernels should take non-power-of-two D (pad the tl.arange extent to the next power of two with a mask): it blocks SilverTorch triton, LiNR V2 and V3 on pubmed (D=768) for E5."
created: "2026-09-26T03:00:00Z"
---

# PubMed SilverTorch build: OOM fix confirmed (commit 2d6eb34)

Staging's harness (8a6096b) + dev/kernel-opt's library, E2's failing cell (pubmed d768 clause c0_mesh).
- Build completes on triton and official. Allocated peak ~38 GiB over a 30.1 GiB fp32 table (quantize +0.5 GiB beyond its 7.2 GiB int8 output); nvidia-smi 75 GB reserved = allocator cache, 44.5 GB steady.
- official: both n_probe cells end to end (build 23 s, recall_oracle@100 0.664 / 0.709, held-out 0.842 / 0.880, n=8428).
- **New blocker**: triton stops at the first forward, `D=768 must be a power of two` (probe scorer tl.arange(0, D)). Same bound hits fused_masked_knn_topk (V2, V3 stage 2) and OPORP at default k_bits=D (W=12). Previously hidden behind the OOM; before Phase 2 it would have been a Triton CompilationError.
Artifacts: docs/artifacts/kernel-opt/pubmed/.
