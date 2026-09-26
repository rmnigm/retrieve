---
chain: "reproducibility-paper"
branch: "main"
parent: "2026-09-15-160000000-f1-f3-record-reframe-deviations-and-provenance.md"
nextStep: "None from this note. When D1 numbers pass their gate, replace each pre-v2 figure below with the v2 measurement of the same claim."
created: "2026-09-26T01:16:00Z"
---

# Fidelity restore: pre-v2 evidence per claim (plan P §B.1-B.2)

Written 2026-09-26 during the docs reorganisation. The first note in this chain kept each claim's paper numbers but dropped the column saying what our earlier measurements showed per claim. Restored here from `docs/plans/reproducibility-paper.md` §B.1-B.2. Every figure below comes from the pre-v2 harness. None is citable (rule 2), and they are kept only as the expected direction.

| claim | pre-v2 evidence | gap the plan named |
|---|---|---|
| S5 | SilverTorch Pareto on arXiv 3M, K <= 400, n_probe sweep, two layouts; no external baseline | Faiss-GPU/CPU IVFFlat and HNSW at matched recall; K in {1000, 2048, 4096}; 20M catalog |
| S6 | V4 (int8 exhaustive) recall@100 0.954-0.971 vs the fp32 oracle; SilverTorch <= 0.91 | recall-vs-n_probe asymptote at K=2048; global vs per-row scale ablation |
| S7 | no standalone filter microbenchmark | `BloomFilter` vs `ExactAttributeFilter` vs CPU posting lists, bs 1-1024, m_bits 512-2048 |
| S8 | m_bits fixed at 1024, FPR never reported | FPR vs m_bits; memory vs `[N, C, A_max]` int64 attrs |
| S9 | kernel-level bloom-vs-none rows only (CUDA handoff §13, CuTe §5) | the full-mask vs partial-mask cell |
| S10 | deep-sweep P ~ 415k at 1664 lists / 256 probes on 3M | probed fraction vs n_lists / n_probe table |
| S12 | K <= 1000 measured | K = 4096 / 10,000 |
| L1 | arXiv C0 (5-30 % pass): V2 0.85 ms vs V1 1.42 ms; Goodreads likewise. This contradicts the paper's "V1 faster at high pass rate" | pass-rate sweep 0.01-100 %, p95, 15M catalog |
| L2 | C5 (~4 % pass): V2 0.81 vs V1 1.43 ms; at B=16 V2 amortises poorly (0.68 ms/query vs V1 0.45) | pass rates << 1 %, p95 |
| L3 | V3 P_c sweep on Goodreads (2k-32k = 0.25-4 %); arXiv C0 V3 1.31 ms vs V1 1.42 at recall 0.68 | bit-width sweep {64, 128, 256, 512} |
| L4 / L5 | ceiling 5.37M items | synthetic 1B x 64 1-bit (8 GB); 240M x 128 fp16 (61 GB) |
| L7 | Triton vs torch backend speedup, numbers only in result JSONs | torch eager vs compiled vs Triton |
| L8 | SilverTorch recall 0.88 -> 0.76 from C0 to C5 while V2 stays ~1.0 | liquidity as "queries with < K candidates in probed clusters" vs n_probe |

The plan's last column also costed G10 (the scale ladder) at 1-2 weeks and at least 200 GB of disk.
