---
title: claims
created: 2026-10-08
updated: 2026-10-08
type: summary
tags: [paper]
sources: [arXiv:2407.13218v3, arXiv:2511.14881v5, docs/validation.md, docs/roadmap.md]
---

# The papers' quantitative claims, and what tests them

Every quantitative claim of the two papers we reproduce, with its section
and figure or table, then the claim-to-evidence matrix C1-C7 that drives
the campaign ([decisions](../decisions.md#campaign-v2-user-2026-10-08):
claims drive cells). Read from the arXiv versions — LiNR
**arXiv:2407.13218v3** (CIKM '24) and SilverTorch **arXiv:2511.14881v5**
(SIGIR '26) — because `articles/` is gitignored and absent on a fresh
box; where our pandoc copies number a table differently, the arXiv
number is given and the difference noted. Nothing in the "ours" column is
citable until its gate is green in [validation](../validation.md).

Testable here means: measurable on public data, one A100, torch-importable
code. The production setups (LinkedIn's 15.5 M jobs, Meta's 10 M / 80 M
pools, HSTU user towers, OverArch, Value Model, A/B tests, TCO) are not
reproducible and are listed so the paper can say so.

## LiNR (arXiv:2407.13218v3)

| id | claim | where | testable here |
|---|---|---|---|
| L-1 | Full-scan serving on GPU with latency as low as 4 ms, indexes from 15 M to 1 B entries | §1 | partly: our latencies at 0.8-10 M; scale above 10 M is out (decisions) |
| L-2 | Native TF/PyTorch boolean masking and indexing cause a **100×** latency increase, hence the custom filter kernel | §6 "Custom filtering kernel" | yes: **C3** (torch backends vs Triton vs postfilter) |
| L-3 | High-pass set (geo + company-reverse, ~1.7 M of 15.5 M pass, ~11 %), d128 fp16, recall label@2000: PyTorch-V1 **4.8 / 4.9 ms** (avg / p95) at B=1, **22.8 / 23.1** at B=16; PyTorch-V2 **14.6 / 47.8** at B=1; TF-V1 6.3 / 6.9 (B=1), 34.8 / 36.6 (B=16); TF-V2 6.9 / 14.4; recall@2k **0.688** for all | §5.3.1, Table 3 | the setup is not; the *shape* is: **C1** |
| L-4 | V1 is faster than V2 at high pass rate (native slicing/copying is slow for large matrices) | §5.3.1 | yes: **C1** |
| L-5 | Low-pass set (adds an exact title clause; max 1.2 M passing, most queries thousands): PyTorch-V2 **1.9 / 2.1 ms** at B=1, **21.4 / 21.9** at B=16; TF-V2 3.4 / 4.5 (B=1), 14.2 / 14.8 (B=16); V2 is superior at low pass rate, and V2 has no recall drop | §5.3.2, Table 4 (`bench report`'s `PAPER_REPORTED` cites it as "tab.5" from our pandoc copy; arXiv v3 numbers it Table 4 — check against `articles/linr.md` on a box that has it) | the setup is not; the crossover is: **C1** |
| L-6 | V3 (Sign-OPORP 1-bit, 512 bits): retaining **1 %** of items for full-precision rescoring gives **~10 %** further latency improvement at **nearly parity** recall; recall@2000 and p95 vs filter size 0.5-2.5 % | §5.3.1, Figure 7 | yes: **C2** |
| L-7 | 1-bit quantization of 1 B × 64 fp16 embeddings: 16× memory reduction, 120 GB → **7.5 GB**, top-50 M from 1 B on one A100 at **97.6 ms** p95, 21 GB HBM | §3.2, §5.3.2 | no (scale); the memory ratio is arithmetic |
| L-8 | Plain KNN with ABM (V1, V2) handles up to **240 M** × 128 fp16 embeddings for top-2k at one query on one A100 | §5.3.2 | no (scale; decisions keep ≤ 10 M) |
| L-9 | Live update: no measurable latency impact at 0/300/600 updates/s (218/215/217 QPS at B=1; 4.57-4.64 ms avg) | §5.3.3, Table 5 | no: live updates are out of scope (backlog G-e) |
| L-10 | Hit Rate@400 gains: Hadamard MLP +10.21 %, MoL with clusters up to +23.67 % (multi-embedding) | §5.1, Table 1 | no (model quality, internal data) |
| L-11 | A/B: +7 % professional interactions, +3 % daily unique professional interactors, −20 % skipped updates; live updates +6 % | §5.2 Table 2, §6 | no |
| L-12 | All reported latencies use separately implemented custom ops (no fused masked-matmul kernel) "for self-consistency"; fusing filtering with matmul and top-k "improves serving speed" | §6 | framing for **C1**/**C3**: our V1/V2 are fused Triton, which the paper itself predicts changes the V1/V2 picture |

## SilverTorch (arXiv:2511.14881v5)

| id | claim | where | testable here |
|---|---|---|---|
| S-1 | End to end, 80 M pool, 2× A100-40G, 24 probes, top-k 1024, 200 ms P99 budget: **1210 QPS**, **23.7×** the CPU baseline (51 QPS), 3.5-6.7× the GPU baselines | §6.1.1, Figure 5a, abstract | no (production pipeline); the paper's operating point (n_probe 24) is reused as ours |
| S-2 | 10 M pool, 1 GPU: **3802 QPS**, 165.3× over CPU, 20.8× over GPU baselines; QPS scales 3.1× from 80 M to 10 M | §6.1.1, Figure 5b | no (pipeline); scaling with N is **C6** in our terms |
| S-3 | Co-design (ProbeThenFilter) gives **~17-25 %** QPS over FilterThenProbe end to end | §6.1.1, Figure 5 | the kernel-level form is **C5** |
| S-4 | Cost efficiency **20.9×** (retrieval) and **13.35×** (with OverArch, 771 QPS) over CPU; 3.56× / 2.27× over the best GPU baseline | §6.1.2, Table 2 | no (TCO) |
| S-5 | P99 ~**15 ms** regardless of traffic; 15.3 ms at 32 probes, 10 QPS (11.4× CPU, 1.6× GPU); ANN search ~2 ms under high traffic; network/transform 18.9 % of the GPU baseline's latency | §6.1.3, Figure 6 | no (pipeline) |
| S-6 | Int8 fused IVF, 20 M × 128, B=16, top-k 2048: **2.2-14.7×** lower latency than Faiss-GPU at recall 0.35-0.92; top-k 4096: **31.3-51×** vs HNSW, **4.6-49.2×** vs Faiss-CPU | §6.2.1, Figure 7 | Faiss/HNSW are out (decisions); the recall-latency curve is **C6** |
| S-7 | Int8 quantization: SilverTorch **cannot reach 0.95 recall**; "no measurable recall loss at 64 probes and top-2048"; global min/max scale | §6.2.1, §4.2 | yes: **C6** (recall ceiling vs n_probe, N, pass rate) |
| S-8 | Bloom index on 40 M items, 5,000 real queries, 6 features × 10 values, 5 hashes: **291-523×** faster than the CPU inverted index, **12.6-42.7×** than the GPU forward index; latency constant from 512 to 1024 bits | §6.2.2, Figure 8a | partly: bloom phase time vs N and m_bits (**C4**); the inverted/forward-index baselines are not reimplemented |
| S-9 | Bloom FPR **6.98** at 512 bits, **0.067** at 1024 (units not stated; read as %); memory 2.56 / 3.84 / 5 / 10 GB at 512 / 768 / 1024 / 2048 bits (40 M items); inverted index 19.8 GB (1.98× the 2048-bit bloom, 4.21× the 1024-bit one); heuristic bits = max values × hashes × 3 = 1800 | §6.2.2, Figure 8b | yes on real attributes: **C4** |
| S-10 | Reducing bloom bits 1024 → 768 raises FPR **0.00173 % → 3.89 %** with no end-to-end recall loss | §6.2.4 | the FPR half: **C4**. Inconsistent with S-9 (0.067 at 1024; 768 between 6.98 and 0.067): a question for the authors (roadmap "Needs the user") |
| S-11 | Co-design at 20 M × 128: scratch **35.6 MB → 18.2 MB** at probe 32 (bloom scratch 18.2 → 0.14 MB), latency **1.55 → 0.72 ms**; **1.79-2.15×** latency improvement on average | §6.2.3, Figure 9 | yes: **C5** |
| S-12 | Co-design at 81 M items, 9,000 clusters, 256 probes: only 2.3 M items (2.8 %) processed, **30×** less filtering compute and scratch | §4.3 | the ratio is arithmetic on cluster sizes; checkable on our indexes |
| S-13 | Probe-then-filter and filter-then-probe give identical recall for a given number of probes | §4.3, §7 | yes: bit-exact by construction (library gate, [validation](../validation.md)) |
| S-14 | The transposed bloom layout lets one 64-bit AND match 64 items; 40 M items → 625,000 partitions | §4.1, Figure 4 | the layout is ours too (TF-1); its N-scaling is **C4** |
| S-15 | Recall with OverArch + Value Model: E-Task +2.4-35.5 %, C-Task +1.12-3 %; QPS 1210 → 771 | §6.2.4, Table 3 | no (models) |
| S-16 | Selective filters may need more probes ("an inherent property of IVF-based search"; recommendation filters are broad) | §7 | yes: **C6** on the synth axis (n95 vs p) |

## Claim-to-evidence matrix

"Exhibit" names the paper's planned tables and figures (T1 claims, T2
real-filter headline, T3 official vs Triton, F1 latency vs pass rate, F2
recall vs pass rate, F3 Pareto, F4a bloom FPR/memory, F4b co-design).
"Reusable records" are judged by the reuse rule
([decisions](../decisions.md#campaign-v2-user-2026-10-08)) from the record
inventory ([artifact](../artifacts/campaign-v2/README.md)); "status so
far" is read from [validation](../validation.md) and is NOT CITABLE.

| id | claim (original) | exhibit | cells (suite · datasets · arms · params) | reusable records | missing (roadmap step) | status so far |
|---|---|---|---|---|---|---|
| C1 | LiNR V1 beats V2 at high pass rate, V2 wins at low (L-3, L-4, L-5) | F1, T1 row | `synth` · goodreads, arXiv, YFCC · V1, V2 triton, bs {1, 16}, k {100, 1000}, 7 (YFCC 5) pass rates, 3 seeds, V1-vs-V2 interleaved | none (no controlled pass-rate axis exists) | V-PILOT, V-AX-SYNTH, V-YFCC | not located; arXiv V2 0.905 ms < V1 1.520 ms at bs 1 hints that fused V2 may win everywhere (L-12's own prediction) |
| C2 | V3 retaining ~1 % gives ~10 % speedup at near-parity recall (L-6) | F2, T1 row | `synth` V3 pool {1 %, 5 %} of passing; `deep` V3 pool {0.5, 1, 2, 5, 10 %} on goodreads, arXiv, YFCC | none: D1's arXiv `deep` V3 cells ran absolute pools (2k-32k), which no v2 cell (`candidate_pool_frac`) keys to ([reuse](../artifacts/campaign-v2/README.md#reuse-entries)) | V-PILOT, V-AX-SYNTH, V-GR-DEEP, V-YFCC | at pool 5000, V3 recall 0.50-0.88 and slower than V2 on arXiv (1.39 vs 0.91 ms): likely not reproduced. Our V3 runs 128 bits at D 128 against LiNR's 512 ([LN-8](reproduction-deviations.md#4-linr-v1v4--deviations)), a gap in the recall half of the claim |
| C3 | Custom filter kernel ~100× faster than native masking/indexing (L-2) | T1 row, T2 | `synth` V1/V2 torch (eager and `torch.compile`) at p {0.01, 0.1, 1.0} vs triton, goodreads and arXiv (YFCC/PubMed only if those two leave C3 ambiguous); `filter` postfilter α {1, 8}; SilverTorch torch reference at n_probe 24 | none: D1-a's torch-vs-triton ratios are on gSASRec goodreads and pre-license-fix arXiv (history) | V-PILOT, V-AX-SYNTH, V-GR-FILTER | 4.2× (V1), 7.3× (V2), 10.2× (V3) at d128: same direction, far smaller |
| C4 | Bloom FPR ~0.067 % at 1024 bits, memory, 291-523× vs inverted index, transposed layout flat in N (S-8, S-9, S-10, S-14) | F4a | `bloomwidth` · goodreads, arXiv, PubMed `c0_mesh` · triton and official bloom · m_bits 64-2048 × k_hash {3, 5}, quality-only + one timed point per width at bs 16 | none at the new widths; `bloom_fp_rate` at 1024 on existing bloom cells (0 false positives seen) | D3 | 0 FP at 1024 bits so the sweep goes lower; transposed-index scaling reproduced with Meta's code (2.0× at 0.8 M, 6.1× at 3 M), not validated |
| C5 | Co-design 1.79-2.15× lower latency than full mask then IVF (S-3, S-11) | F4b | `codesign` · arXiv, goodreads · official bloom, `bloom_path` {partial, full}, n_probe {8, 32, 128}, 3 sweeps, 3 seeds, interleaved | none: arXiv 60/60 at `c0e42d1`, 58 unstable, not interleaved — fails the timing half of the rule | V-CODESIGN | arXiv leg run but unstable; goodreads not run |
| C6 | Int8 IVF with the filter in the probe kernel: high QPS, recall ceiling < 0.95 at production settings, more probes for selective filters (S-6, S-7, S-16) | F3, T2, F2 | `deep` (goodreads, arXiv, YFCC), `synth` SilverTorch clause n_probe {24, n95, 4·n95} + bloom at n95, `filter` n_probe {24, n95}, the PubMed `n95` probe | in the manifest: none; arXiv `deep` SilverTorch quality at `72e5a90` passes the rule on 162 cells (perf on 51) but its groups lack v2's `n_probe` 16 / 64, and a manifest `match` cannot name params ([reuse](../artifacts/campaign-v2/README.md#reuse-entries)) | V-GR-DEEP, V-YFCC, V-SEEDS, V-PUBMED | recall at n_probe 24 falls with N: 0.84 (3 M), 0.55 (YFCC), 0.67 (PubMed); Triton < 1 ms at bs 1 up to 10 M d192 |
| C7 (ours) | Our Triton reimplementation matches official int32 bit-exactly and is faster end to end, Meta's scorer faster in isolation | T3, lessons | `h2h` · goodreads E1c, arXiv · triton, official fp16, official int32 · none + bloom, n_probe 24, interleaved, 5 repeats, `--profile`; L6 parity gate at d 64-768 | none (b3 and kernel-opt h2h are history, superseded) | CV2-LIB (L6), H2H-FINAL | b3 "citable, contested"; after CSR/transposed bloom, kernel-only bloom 0.90× (goodreads) / 1.24× (arXiv), not validated |

Claims marked "no" above enter T1 as *untestable* with the reason, so the
table covers the papers whole. Every planned cell of the
[roadmap](../roadmap.md) maps to one of C1-C7; a cell that maps to none
leaves it.
