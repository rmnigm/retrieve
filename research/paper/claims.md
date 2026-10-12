---
title: claims
created: 2026-10-08
updated: 2026-10-11
type: summary
tags: [paper]
sources: [arXiv:2407.13218v3, arXiv:2511.14881v5, research/validation.md, research/roadmap.md]
---

# The papers' quantitative claims, and what tests them

Every quantitative claim of the two papers we reproduce, with its section
and figure or table, then the claim-to-evidence matrix C1-C7, re-stated for
the study: Meta's official SilverTorch code, adapted by us, against LiNR
V1-V3 ([decisions](../decisions.md#goal-and-scheduling)); our Triton
SilverTorch reimplementation is not in it. The matrix drives
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
| S-6 | Int8 fused IVF, 20 M × 128, B=16, top-k 2048: **2.2-14.7×** lower latency than Faiss-GPU at recall 0.35-0.92; top-k 4096: **31.3-51×** vs HNSW, **4.6-49.2×** vs Faiss-CPU | §6.2.1, Figure 7 | Faiss/HNSW are out (model-based retrieval: standalone ANN services, not graph ops; decisions); the recall-latency curve is **C6** |
| S-7 | Int8 quantization: SilverTorch **cannot reach 0.95 recall**; "no measurable recall loss at 64 probes and top-2048"; global min/max scale | §6.2.1, §4.2 | yes: **C6** (recall ceiling vs n_probe, N, pass rate) |
| S-8 | Bloom index on 40 M items, 5,000 real queries, 6 features × 10 values, 5 hashes: **291-523×** faster than the CPU inverted index, **12.6-42.7×** than the GPU forward index; latency constant from 512 to 1024 bits | §6.2.2, Figure 8a | partly: bloom phase time vs N and m_bits (**C4**); the inverted/forward-index baselines are not reimplemented |
| S-9 | Bloom FPR **6.98** at 512 bits, **0.067** at 1024 (units not stated; read as %); memory 2.56 / 3.84 / 5 / 10 GB at 512 / 768 / 1024 / 2048 bits (40 M items); inverted index 19.8 GB (1.98× the 2048-bit bloom, 4.21× the 1024-bit one); heuristic bits = max values × hashes × 3 = 1800 | §6.2.2, Figure 8b | yes on real attributes: **C4** |
| S-10 | Reducing bloom bits 1024 → 768 raises FPR **0.00173 % → 3.89 %** with no end-to-end recall loss | §6.2.4 | the FPR half: **C4**. Inconsistent with S-9 (0.067 at 1024; 768 between 6.98 and 0.067): a question for the authors (roadmap "Needs the user") |
| S-11 | Co-design at 20 M × 128: scratch **35.6 MB → 18.2 MB** at probe 32 (bloom scratch 18.2 → 0.14 MB), latency **1.55 → 0.72 ms**; **1.79-2.15×** latency improvement on average | §6.2.3, Figure 9 | yes: **C5** |
| S-12 | Co-design at 81 M items, 9,000 clusters, 256 probes: only 2.3 M items (2.8 %) processed, **30×** less filtering compute and scratch | §4.3 | the ratio is arithmetic on cluster sizes; checkable on our indexes |
| S-13 | Probe-then-filter and filter-then-probe give identical recall for a given number of probes | §4.3, §7 | yes: bit-exact by construction (library gate, [validation](../validation.md)) |
| S-14 | The transposed bloom layout lets one 64-bit AND match 64 items; 40 M items → 625,000 partitions | §4.1, Figure 4 | not on the study's arm: Meta's released code builds a per-2048-doc bundle bloom, not this layout ([OF-1](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)); our transposed layout was in the Triton reimplementation, out of the study. Stated as a paper-vs-code finding |
| S-15 | Recall with OverArch + Value Model: E-Task +2.4-35.5 %, C-Task +1.12-3 %; QPS 1210 → 771 | §6.2.4, Table 3 | no (models) |
| S-16 | Selective filters may need more probes ("an inherent property of IVF-based search"; recommendation filters are broad) | §7 | yes: **C6** on the synth axis (n95 vs p) |

## Claim-to-evidence matrix

"Exhibit" names the paper's planned tables and figures (T1 claims, T2
real-filter headline, T3 Meta as published vs adapted, F1 latency vs pass
rate, F2 recall vs pass rate, F3 Pareto, F4a bloom FPR/memory, F4b
co-design). Each claim is re-stated for the study and marked **keep**
(unchanged), **change** (its SilverTorch half moves to the adapted code, or
its subject changes) or **drop**. "Runs" names what feeds it: LiNR from the
final pass (`campaign-final`), SilverTorch from the adapted backend's
measurement legs (roadmap
[NUMBERS](../roadmap.md#2-numbers-adapted-silvertorch-over-the-study-grid));
the full rerun of everything at the last repo version comes last and
regenerates the final tables.
Records of our Triton SilverTorch feed no claim. "Status so far" is read
from [validation](../validation.md) (the final pass's EXHIBITS and D1-G rows,
the META-FORK gates) and is **NOT CITABLE**; no number here is new.

| id | claim (original) | for this study | decision | why | runs | exhibit | status so far (NOT CITABLE) |
|---|---|---|---|---|---|---|---|
| C1 | LiNR V1 beats V2 at high pass rate, V2 wins at low (L-3, L-4, L-5) | unchanged | **keep** | LiNR only; untouched by the reframe | final pass: `synth` V1 / V2 Triton at 0.8-30 M, bs {1, 16}, seeds 0-2 on the crossover cells (`seeds-c1`) | F1, T1 | changed at campaign-v2.12: the bs 16 crossover is near p 0.017-0.026 at 0.8 / 3 / 10 / 30 M, V2's bs 16 edge at most 13 %; at bs 1 the crossover is 0.76-0.98 |
| C2 | V3 retaining ~1 % gives ~10 % speedup at near-parity recall (L-6) | unchanged; NUMBERS item 7 adds V3 as a pool curve against SilverTorch at matched recall | **keep** | LiNR only | final pass: `synth` / `deep` V3 pools, `v3bits`, V3-BITS-PUBMED; NUMBERS | F2, T1 | **not comparable as stated**: LiNR's recall is end-task recall with an unpublished protocol. Under our metric (share of the exact filtered top-k returned) a 1 % pool reaches 0.89 (median, real filters) and parity needs 20 %, where V3 is no faster than V2 ([results](results.md#claim-verification)); our V3 runs at most D bits against LiNR's 512 at d128 ([LN-8](reproduction-deviations.md#4-linr-v1v3--deviations)) |
| C3 | Custom filter kernel ~100× faster than native masking / indexing (L-2) | LiNR's custom filter kernels (Triton) against native torch masking (eager and `torch.compile`) | **change** | the SilverTorch torch-reference arm leaves with the Triton reimplementation; LiNR's torch arms carry the claim alone | final pass: V1 / V2 torch vs Triton on `synth`, C3 coverage on real filters, the postfilter | T1, T2 | does not hold: same direction, far smaller than ~100× |
| C4 | Bloom FPR ~0.067 % at 1024 bits, memory, 291-523× vs inverted index, transposed layout flat in N (S-8, S-9, S-10, S-14) | the adapted SilverTorch bloom's FPR, memory and recall against width (configured width C12, recheck C13), beside Meta's bloom as published; the transposed-layout half is not testable on the study's arm (S-14) | **change** | Meta's published bloom ignores `m_bits` (OF-10) and admits bundle false positives (OF-12); the adaptation changes both, so the width curve must come from the adapted code | META-FORK gate R; NUMBERS `bloomwidth`; Meta-as-published D3 records for the "as published" column | F4a | Meta's published bloom: false positives in the results at 30 M (OF-12), width not followed (OF-10); the adapted bloom: gate R running |
| C5 | Co-design 1.79-2.15× lower latency than full mask then IVF (S-3, S-11) | partial (co-designed) vs full mask on the adapted backend, beside Meta's code as published | **change** | the Triton "ours" half (C5-OURS) leaves; the adapted code builds the partial mask inside its payload kernels (C8), so the published code's host-bound partial path is no longer the only answer | NUMBERS `codesign` (adapted and published, interleaved) incl. 30 M | F4b | Meta's code as published (`-O3`): full / partial above 1 at 30 M (the co-design wins), reversed below 30 M; the adapted code not yet timed |
| C6 | Int8 IVF with the filter in the probe kernel: high QPS, recall ceiling < 0.95 at production settings, more probes for selective filters (S-6, S-7, S-16) | the same, on the adapted code, reported at the tuned point per dataset size with per-filter probe curves (NUMBERS items 1-3) | **change** (arm only) | the int32 path is bit-exact with Meta's published one, so recall cannot differ by implementation; the numbers still come from the adapted rerun | NUMBERS `deep`, `synth`, `filter`, ceiling checks, the LAION N-sweep | F3, T2, F2 | holds (recall ceiling below 0.95 at the cap at 10 M; more probes for selective filters), on the Triton records; to be re-read on the adapted code |
| C7 (ours) | Our Triton reimplementation matches official int32 bit-exactly and is faster end to end; Meta's scorer faster in isolation | Meta's official code, adapted, is bit-exact with the code as published (int32, fp16) except where a change alters the bloom on purpose, captures in a CUDA graph, and is faster than the published code (`-O3`) end to end | **change** | the subject was our Triton reimplementation, which is out of the study; the comparison that remains is Meta as published vs adapted ([page](meta-published-vs-adapted.md)) | META-FORK gates (X, R, S, capture, compile); NUMBERS `h2h` (adapted vs published, interleaved, both score paths) | T3 | X: every cell of the ≤ 10 M gate grid `torch.equal` (hub-index `artifacts/meta-fork/head-82fafb6`); faster than the published `-O3` code on that grid; review pending |

**The study's own comparison.** SilverTorch (adapted) against LiNR is not a
claim of either paper; it is what the exhibits show side by side: T2 (real
filters), F1 / F2 (pass rate), F3 (recall-latency Pareto at the tuned point
per dataset size, [decisions](../decisions.md#campaign-v2-user-2026-10-08)),
with paired CIs from NUMBERS item 5 (interleaved SilverTorch vs LiNR).

Claims marked "no" above enter T1 as *untestable* with the reason, so the
table covers the papers whole. Every planned cell of the
[roadmap](../roadmap.md) maps to one of C1-C7 or to the study's own
comparison; a cell that maps to none leaves it.
