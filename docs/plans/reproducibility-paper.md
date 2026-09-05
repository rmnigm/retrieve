# Reproducibility paper plan — torchretrieve vs. SilverTorch / LiNR

> **Status:** planning document, 2026-09-05. Research + gap audit only; no code changed.
> Companion agents own the benchmark-harness rewrite and the large-dataset search — this
> document only states what the paper *needs* from them (§B.6, §B.7).
> Sources: local — `articles/silvertorch.md`, `articles/linr.md`, `docs/thesis/main.tex`
> (ch. 4–6), `docs/system/{filtering,architecture,evaluation}.md`, `docs/plans/00-roadmap.md` §4,
> `docs/plans/future-work-and-research.md`, `docs/plans/cuda-silvertorch-handoff.md` §13,
> `docs/plans/cute-dsl-scorer.md` §5, `evaluation/config/**`, `evaluation/results/**`.
> Web sources are cited inline; anything I could not verify is marked **[unverified]**.
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) — G2 is roadmap Phase B, G3/G4/G7/G8 are
> delivered by Phase D1, G5/G6 are D2/D3, G1/G9 are F1/F3, G10–G16 are Phase G. The venue decision
> (ECIR 2027 Reproducibility, paper 19 Oct 2026) is recorded in roadmap §0.

## 0. Executive summary

1. **The premise "industry papers with no public code" is half wrong.** Meta released
   [meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch) (Apache-2.0, repo created
   2025-10-07, last push 2026-07-23, no tagged releases, 29 stars per the GitHub API on 2026-09-05).
   The README advertises only the Bloom index, but the tree ships
   `ops/csrc/fused_kmean_ann_cuda.cu` with the schema
   `fused_kmean_ann(cluster_offsets, cluster_ids, cluster_length, embeddings, queries, ..., Tensor? filtering_bit_mask, int divisor_for_int8, Tensor? filtering_bit_index, Tensor? per_embedding_scale)`
   (int8 with dp4a, optional 1-bit filter mask) plus `fused_kmean_ann_with_partial_masks` — i.e.
   Algorithm 1's Phase 2+3 kernels — and `fresh_index_post_processing`. LiNR has no public code
   (searched 2026-09-05; only the arXiv/ACM PDFs at https://arxiv.org/abs/2407.13218).
   Consequence: the paper is a **reproducibility** study (different team, different setup, different
   data) for both, *and* can include a **replicability** check of SilverTorch's kernels by running the
   authors' ops on our public data. Reviewers will ask why not; do it (§B.3 G2).
2. **Best-fit venue: ECIR 2027 Reproducibility track** (abstract 12 Oct 2026, paper 19 Oct 2026,
   12 LNCS pages, "a successful reproduction of the work is not a requirement"), with **SIGIR 2027
   Reproducibility** (≈Feb 2027 by the 2026 proxy, 9 ACM pages) as the chained resubmission target
   and RecSys 2027 as the recsys-only fallback. KDD has no reproducibility track and its research
   track excludes "purely replicative work". Reasoning in §A.2.
3. **The thesis reframes badly for a repro paper:** `main.tex` calls QuantizedIVF "a new algorithm
   proposed in this work" (lines 242, 254, 462, 898). The paper must call it what
   `docs/system/architecture.md` already calls it — SilverTorch's Algorithm 1 — and list deviations.
4. **Six things no reviewer will let pass:** (i) no QPS / P95 / P99 anywhere (papers report QPS
   under a 200 ms P99 budget, avg+p95); (ii) single seed, no CIs; (iii) no external baseline
   (Faiss-GPU/CPU, HNSW, cuVS) — every SilverTorch speedup is "vs Faiss"; (iv) scale ≤ 5.4M vs
   10M–80M / 240M / 1B; (v) the official SilverTorch kernels not run; (vi) Goodreads d128/d256
   filter sweeps still on the stale oracle (`00-roadmap.md` §2 — publication blocker).
5. Effort to an ECIR-grade paper with the existing A100: ~4 weeks of which ~10 GPU-days
   (§B.3 tiers P0). SIGIR-grade adds the scale ladder and three more baselines (~3 more weeks).

---

## Part A — Venue research

### A.1 Tracks, formats, deadlines (checked 2026-09-05)

| Venue | Repro/resource track? | What it wants | Pages / format / anonymity | Deadlines | Artifacts & badges |
|---|---|---|---|---|---|
| **ECIR 2027** (Southampton, 21–25 Mar 2027) — [CfP](https://www.ecir2027.co.uk/call-for-reproducibility-papers) | Yes, Reproducibility track | "replicability (different team, same experimental setup) and reproducibility (different team, different experimental setup)"; repeatability (same authors) rejected; "A successful reproduction of the work is not a requirement, but it is crucial to provide a precise and rigid evaluation of the process"; must explain motivation for choosing the papers, the process, difficulties, lessons | up to 12 pages + refs, Springer LNCS, appendices count; double-blind | abstract **12 Oct 2026**, paper **19 Oct 2026**, notification 7 Dec 2026 | code via anonymous repo or EasyChair upload, reviewers need access; criteria **Reliability / Impact / Novelty / Availability** |
| ECIR 2027 Resource track — [CfP](https://www.ecir2027.co.uk/call-for-resource-papers) | Yes (for the library itself, later) | "Software tools and frameworks that support experimentation, evaluation, ..., benchmarking, reproducibility" | 12 pages LNCS; single-blind | submission 2 Nov 2026, notification 7 Dec 2026 | permanent URL, open licence, maintenance plan |
| **SIGIR 2027** | CfP not published yet **[unverified]**; proxy = [SIGIR 2026 Reproducibility track](https://sigir2026.org/en-AU/pages/submissions/reproducibility-track) | "repeat, reproduce, generalise, or analyse prior work"; preference for reproducibility over replicability; "not on reproducibility badging but on generating new research insights with existing approaches"; analyse which assumptions held and which did not, error modes, unexpected conclusions | at most 9 pages incl. everything except references, ACM `sigconf`; **double-anonymous** in 2026 (2025 was non-anonymous — [SIGIR 2025 call](https://sigir2025.dei.unipd.it/call-res-repro-papers.html)) | 2026 cycle: abstract 5 Feb, paper 12 Feb, notification 2 Apr, camera-ready 29 Apr (AoE) | "publicly share all the resources"; criteria **Contribution / Motivation / Soundness / Quality of reproduction artefacts**; post-acceptance [ACM SIGIR badging](https://sigir.org/general-information/acm-sigir-artifact-badging/) via OpenReview within 3 years, README "from a freshly installed OS", pinned deps, Docker suggested |
| **RecSys 2026** (Minneapolis, 28 Sep–2 Oct 2026) — [CfP](https://recsys.acm.org/recsys26/call/) | Yes, Reproducibility track | reproducibility/replicability papers "across different domains, using different datasets, or comparing against alternative baselines" + resource papers; methodology/reflective papers dropped in 2026; original authors excluded | repro: 8 pages + 2 ref pages ACM; resource: 4 pages; **mutually anonymous** | 2026 cycle: abstract 28 Apr, paper 5 May, notification 9 Jul, camera-ready 27 Jul; 2027 dates not announced **[unverified]** | artifacts must be available to reviewers or desk-reject; criteria novelty of findings, impact, technical rigor, "Artifacts Availability: is the submitted work repeatable?" |
| **KDD 2027** (San Jose, Aug 2027) — [Research](https://kdd2027.kdd.org/research-track-call-for-papers/), [D&B](https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/) | No repro track; D&B track = datasets/benchmarks/tools | Research track: "purely replicative work" out of scope; D&B: new benchmarks and benchmarking tools | 8 content pages + unlimited refs/appendix, ACM; double-blind | abstract 19 Jul 2026, paper 26 Jul 2026 (**passed**), notification 14 Nov 2026 | "Artifacts Available" ACM badge with DOI-hosted artifacts |
| CIKM 2026 — [Resource](https://cikm2026.diag.uniroma1.it/resource-papers/) | Resource track only | datasets, benchmarks, software | 4 pages + refs, ACM; single-blind | abstract 30 May / paper 6 Jun 2026 (**passed**) | GitHub code; DOI for data (Zenodo) |
| WSDM 2027 — [CfP](https://www.wsdm-conference.org/2027/cffp.html) | None (main + new Findings track) | — | 9 pages + refs | abstract 17 Aug 2026 (**passed**) | code sharing "encouraged" |
| TheWebConf 2027 — [CfP](https://www2027.thewebconf.org/research-track-papers/) | None; closest is "Evaluation, Human Computation, and Resources" research track | — | 8 content / 12 total, ACM | abstract 18 Oct, paper 25 Oct 2026 | "Artifacts Available" badge |

ACM badge semantics (Artifacts Available / Evaluated–Functional / Evaluated–Reusable / Results
Reproduced / Results Replicated):
https://www.acm.org/publications/policies/artifact-review-and-badging-current. Only KDD, WWW and the
SIGIR post-publication process award them; ECIR (Springer) has no badge, so "Availability" is a
review criterion instead.

### A.2 Which venue, and why

**Primary: ECIR 2027 Reproducibility.** (a) Its CfP is the only one that says in writing that a
failed reproduction is acceptable — we *will* report failures (LiNR's V1-faster-than-V2 claim
inverts under a fused compaction kernel; the Triton bloom path did not show the paper's transposed-index
win until the CUDA backend's memory-level-parallelism fix, `cuda-silvertorch-handoff.md` §13).
(b) 12 LNCS pages fit two claims tables and a deviations table; 9 ACM pages do not comfortably.
(c) ECIR's efficiency community is the audience — ECIR 2026 accepted "Practical, Efficient, In-Memory
Inverted Indexes" and "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'"
([accepted list](https://ecir2026.eu/programme/accepted-papers); it listed 4 reproducibility papers
when read through a proxy on 2026-09-05, while a conference post claims 9 **[unverified]**).
(d) The timing chains: ECIR notification 7 Dec 2026 → SIGIR 2027 deadline ≈ early Feb 2027 →
RecSys 2027 ≈ May 2027. Nothing is lost by trying ECIR first with the P0 cut (§B.3).
**Risk:** 6 weeks from today; P0 needs ~10 A100-days. If the GPU is not continuously available in
September, skip straight to SIGIR 2027 and use the extra months for P1.

**SIGIR 2027** is the canonical home if SilverTorch was indeed a SIGIR'26 paper
(`future-work-and-research.md` says so; the arXiv listing does not — **[unverified]**): "reproduce
last year's SIGIR systems paper on public data" is the archetypal SIGIR repro submission. It weighs
"new research insights" more than fidelity, so the generalisation axes (backends, embedding sources,
selectivity) must carry the paper there. Double-anonymous: the PyPI name `torchretrieve` and the
GitHub org identify the author — submit through an anonymised mirror and name the package only in
camera-ready.

**RecSys 2027** only if the story is narrowed to recommendation (drop arXiv, add a large recsys
dataset). Its reviewers (Ferrari Dacrema tradition) will scrutinise the evaluation protocol
(splits, oracle definition, baseline tuning) more than kernels.

**KDD:** no fit. The library + harness could later go to KDD D&B or ECIR Resource as a separate
4–12-page resource paper; do not mix that into the reproducibility paper.

### A.3 What well-regarded reproducibility papers look like

| Paper | Structural lesson for us |
|---|---|
| Ferrari Dacrema et al., RecSys'19 "Are We Really Making Much Progress?" ([arXiv](https://arxiv.org/abs/1907.06902), [code](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)); TOIS'21 extension ([arXiv](https://arxiv.org/abs/1911.07698)) | Explicit *reproducibility criteria* applied to each target (code available? runs? datasets public?); baselines tuned with the same budget as the proposed method; per-claim verdicts; the meta-finding is the contribution |
| Rendle, Zhang, Koren 2019 "On the Difficulty of Evaluating Baselines" ([arXiv](https://arxiv.org/abs/1905.01395)) | An under-tuned baseline invalidates a speedup claim. Faiss/cuVS `nprobe`, HNSW `ef`, our `n_probe`/`P_c` must be tuned on the same recall grid |
| Kamphuis, de Vries, Boytsov, Lin, ECIR'20 "Which BM25 Do You Mean?" ([preprint](https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf)) | Enumerate *variants* of the "same" method formally, then test whether they differ significantly. We have bloom vs exact predicate, global vs per-row int8 scale, row-wise vs transposed bloom index, fused vs two-kernel — a "which SilverTorch do you mean" table |
| Lin, SIGIR Forum'18 "The Neural Hype and Comparisons Against Weak Baselines" ([ACM](https://dl.acm.org/doi/10.1145/3308774.3308781)); Ma, Sun, Pradeep, Lin 2021 "A Replication Study of Dense Passage Retriever" ([arXiv](https://arxiv.org/abs/2104.05740)) | Independent re-implementation inside a maintained toolkit; report *both* "claims largely verified" and "original under-reported the baseline"; the toolkit (Pyserini) is the durable artifact — same role as `torchretrieve` |
| Sun et al., RecSys'20 "Are We Evaluating Rigorously?" ([DOI](https://doi.org/10.1145/3383313.3412489), [DaisyRec](https://github.com/AmazingDD/daisyRec)) | Enumerate evaluation-protocol factors (split, negative sampling, tuning, metric) and report sensitivity to them |
| Campagnano, Mallia, Silvestri, SIGIR'25 "Unveiling DIME: Reproducibility, Generalizability, and Formal Analysis" ([PDF](https://www.pinecone.io/research/SIGIR25a.pdf)) | Paper is organised strictly by RQ: RQ1 theory, RQ2 reproducibility, RQ3 generalisability (new models/datasets), RQ4 refinement; one section per RQ |
| Ghosh, David, Chatterjee, SIGIR'26 "Reproduction Beyond Benchmarks: ConstBERT and ColBERT-v2 Across Backends and Query Distributions" ([arXiv](https://arxiv.org/abs/2604.09982)) | Reproduce within 0.05% on the original benchmark, then show 86–97% drops on other query distributions and an 8-point gap from backend parameters — our backend axis (torch/Triton/CUDA/CuTe) and embedding-source axis are the same move |
| Said & Bellogín 2026, "Reproducibility in Recommender Systems: A Survey" (51 RecSys repro-track papers 2020–25) ([arXiv](https://arxiv.org/abs/2607.26074)) | In practice repro papers *extend* rather than strictly replicate (new datasets, baselines, criteria); the track rewards that; dataset diversity is low, so a new large filtered dataset is itself a contribution |
| Fuhr, SIGIR Forum'17 "Some Common Mistakes in IR Evaluation" ([ACM](https://dl.acm.org/doi/10.1145/3190580.3190586)); Hoefler & Belli, SC'15 "Scientific Benchmarking of Parallel Computing Systems" ([ACM](https://dl.acm.org/doi/10.1145/2807591.2807644)) | Don't over-state precision; relative improvements of means are misleading; report CIs, the full distribution (violins), multiple-comparison-aware tests; state whether an effect is deterministic or observed by chance |
| Big-ANN NeurIPS'23 filter track ([results](https://arxiv.org/abs/2409.17424), [site](https://big-ann-benchmarks.com/neurips23.html)); FANNBench, SIGMOD'26 ([arXiv](https://arxiv.org/abs/2508.16263), [code](https://github.com/lmcccccc/FANNBench)); Shi et al. 2025 unified filtered-ANN benchmark ([arXiv](https://arxiv.org/abs/2509.07789)) | Standard filtered-ANN protocol = QPS at recall ∈ {0.90, 0.95, 0.99} on a *selectivity* axis from 0.1% to 100%, plus build time and index memory. Adopt it verbatim so the paper is comparable |

**Reviewer checklist distilled from the above** (each item maps to a gap in §B.3):
research questions phrased around the original claims · a claims table (claim → verdict →
deviation) · explicit deviations with reasons · generalisation to new data/settings · multiple
seeds + CIs + paired tests for "A beats B" · full hardware/software disclosure (GPU, driver, CUDA,
torch, triton, commit, clocks) · honest negative results · strong *public* baselines tuned on the
same recall grid · artifact that runs from a fresh machine with one command · a
reproducibility checklist in the paper.

---

## Part B — Gap analysis

### B.0 What the thesis has today (evidence inventory)

Datasets (`main.tex` tab:datasets): Goodreads 797,043 items / 313,178 test users (gSASRec, 4
clauses incl. one reverse), arXiv 2.99M / 10,000 queries (Nomic-Embed, 4 clauses), Yambda-500M
3.06M / 45,932, Yambda-5B 5.37M / 459,067 (no filters). Sweeps: `d ∈ {64,128,256}`, `K ∈ {100,
500, 1000}` (+200/400 in deep sweeps), `B ∈ {1, 8, 16}`, backends `{triton, torch}` (+`cuda`,
`cute` for SilverTorch in the deep sweep), bloom `m_bits=1024, k_hash=5`, IVF `n_lists ∈ {1664,
8192}`, `n_probe ∈ {4…256}`, single `seed: 0`, `users_limit: 10000`. Metrics: recall/ndcg@K vs a
filtered full-scan fp32 oracle (filtered cells) or held-out interactions (no-filter), `do_bench`
median/p20/p80 over a 4096-query pool, 20 warm-ups, index and peak memory. Hardware: one
A100-SXM4-80GB; torch 2.10.0+cu128, triton 3.6.0 (from the CUDA/CuTe validation records).
Thesis artefacts that survive into the paper: tab:recall_nofilter, tab:pareto_arxiv,
tab:pareto_goodreads, fig:pareto_quality_latency, tab:batch_scaling, fig:batch_scaling,
fig:dim_scaling, tab:memory, fig:selectivity_recall, fig:linr_v3_pc_sweep, fig:silvertorch_pareto,
fig:backend_speedup (ch. 4), plus the kernel-level backend tables in `cuda-silvertorch-handoff.md`
§13 and `cute-dsl-scorer.md` §5/§5.1 (not in the thesis).

### B.1 Claims-coverage table — SilverTorch (`articles/silvertorch.md` §4, §6)

| # | Claim (paper) | Testable here? | Thesis shows | Missing |
|---|---|---|---|---|
| S1 | E2E 80M pool, 24 probes, top-k 1024, 12-way multi-embedding queries, AND/OR/NOT over 6 features: **1210 QPS under 200 ms P99**, 23.7× vs Faiss-CPU+inverted index, 3.5–6.7× vs Faiss-GPU+forward-index shards; 5 runs averaged | No (multi-server client/server, 80M, 2-GPU shard, proprietary queries) | nothing comparable | Declare out of scope; replace with single-GPU QPS-under-P99 on ≤ 15M (S2 proxy) |
| S2 | 10M pool unsharded: **3802 QPS**, 165.3× CPU, 20.8× GPU baseline | Partially — 10M is reachable (real 10M from the dataset search, or `synth_arxiv` 15M) | latency only, max 5.37M | QPS + P99 in the harness; a Faiss-GPU IVFFlat + attribute-mask baseline on the same box |
| S3 | QPS scales 3.1× from 80M → 10M | Partially | 0.8M / 3M / 5.4M latencies, different datasets | a same-dataset size ladder 1M → 50M (synthetic) reporting QPS |
| S4 | TCO: 20.9× vs CPU, 3.56× vs best GPU baseline; 771 QPS with OverArch still 13.35× / 2.27× | No (needs S1) | — | out of scope; at most a J/query column (`future-work` I8) |
| S5 | ANN breakdown, 20M×128d, batch 16, top-k 2048: INT8 fused IVF **2.2–14.7× lower latency than Faiss-GPU IVFFlat fp32** at recall 0.35–0.92; at top-k 4096 **31.3–51× vs HNSW, 4.6–49.2× vs Faiss-CPU**; 50 warm-up + 100 test batches | Partially | fig:silvertorch_pareto (arXiv 3M, K ≤ 400, n_probe sweep, two layouts) — no external baseline | Faiss-GPU/CPU IVFFlat and HNSW at matched recall; K ∈ {1000, 2048, 4096}; 20M catalog |
| S6 | INT8 "cannot reach 0.95 recall"; "no recall loss with 64 probes and top-2048" (§4.3) | Yes | tab:pareto_*: V4 (int8 exhaustive) recall@100 0.954–0.971 vs fp32 oracle; QuantizedIVF ≤ 0.91 | recall-vs-n_probe asymptote at K = 2048; global vs per-row scale ablation (official op exposes `per_embedding_scale`) |
| S7 | Bloom index 40M items, 6 features / 10 values avg, 5 hashes: **291–523× vs CPU inverted index, 12.6–42.7× vs GPU forward index**; latency flat from 512 to 1024 bits | Partially (≤ 5.4M; forward index ≈ `ExactAttributeFilter`) | no standalone filter microbenchmark | `BloomFilter` vs `ExactAttributeFilter` vs a CPU posting-list intersection, batch sizes 1–1024, m_bits 512–2048 |
| S8 | FPR 6.98% at 512 bits → 0.067% at 1024; memory 1.2 / 1.8 / 2.4 / 4.7 GB for 512/768/1024/2048 bits vs 19.8 GB inverted index; heuristic bits = max_values × K × 3 | Yes | m_bits fixed at 1024, FPR never reported | measured FPR vs m_bits on Goodreads/arXiv; memory table vs `[N, C, A_max]` int64 attrs; note our `(clause_idx, value)` salting deviation (`filtering.md`) |
| S9 | Co-design (20M, probe 32): scratch 35.6 MB → 18.2 MB, latency 1.55 → 0.72 ms, **1.79–2.15×** across probes | Yes, ablation not run | kernel-level bloom-vs-none rows in handoff §13 / cute §5 | explicit cell: full bloom mask → IVF vs fused partial bloom, latency + `fwd_scratch_mib` vs n_probe |
| S10 | 81M items / 9,000 clusters / 256 probes scores 2.3M items (2.8%), 30× less filtering work | Yes (arithmetic + measured P) | deep-sweep comments (P ≈ 415k at 1664/256 on 3M) | one table of probed fraction vs n_lists/n_probe |
| S11 | OverArch (MoL) + Value Model: +2.4–28.2% E-task, +0.6–1.12% C-task recall; QPS 1210 → 771 | No | — | out of scope (`future-work` R4 is a separate paper) |
| S12 | Tensor-native index; no top-k cap (Faiss-GPU ≤ 2048, CAGRA ≤ 1024); one tuning knob vs HNSW's three | Yes (qualitative) | K ≤ 1000 measured | run K = 4096 / 10,000; cite [Faiss GPU wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU) for the 2048 cap |
| S13 | Transposed bloom index: one 64-bit AND tests 64 items; 1-bit masks not 8-bit bool | Yes | CUDA backend implements it; Triton uses row-wise sigs; handoff §13: kernel-only bloom 2.1× only after the MLP fix | this is the "which SilverTorch" finding — write it up |
| S14 | Fresh index / streaming updates | No | — | out of scope (`live-update-api.md`) |
| S15 | Scale-out to 2 GPUs for 80M | No | — | out of scope |
| S16 | Int8 = global min/max scale at publish | Yes | implemented as global scale (`architecture.md`) | ablation vs per-row (S6) |

### B.2 Claims-coverage table — LiNR (`articles/linr.md` §3.1–3.2, §5.3, §6)

| # | Claim (paper) | Testable? | Thesis shows | Missing |
|---|---|---|---|---|
| L1 | High-pass-rate (15.5M jobs, 25k queries, geo + reverse-company clauses, avg 1.7M pass ≈ 11%, d=128 fp16): PyTorch-V1 bs1 **4.8 ms avg / 4.9 p95**, PyTorch-V2 14.6 / 47.8; bs16 V1 22.8 / 23.1; recall@2k 0.688 for all → **V1 faster than V2 at high pass rate** (native slicing overhead) | Yes — and the thesis **contradicts** it | tab:pareto_arxiv C0 (5–30% pass): V2 0.85 ms vs V1 1.42 ms; Goodreads likewise | the pass-rate sweep 0.01%…100% to locate the crossover; p95; 15M catalog (`synth_arxiv` 15M); explain: our V2 fuses compaction (paper §6 predicts exactly this flip) |
| L2 | Low-pass-rate (extra title clause, most queries pass thousands): PyTorch-V2 bs1 **1.9 / 2.1 ms**, TF-V2 3.4 / 4.5; bs16 TF 14.2 vs PyTorch 21.4 → V2 dominates; per-query parallel matters at bs16 | Yes | C5 (~4% pass): V2 0.81 vs V1 1.43 ms; B=16 V2 amortises poorly (tab:batch_scaling 0.68 ms/query vs V1 0.45) — consistent | pass rates ≪ 1%; p95; bs16 analysis of the per-query-variable-P problem |
| L3 | V3 512-bit Sign-OPORP keeping 1% of items: **~10% further latency gain at near-parity recall** (fig 7) | Yes | fig:linr_v3_pc_sweep (Goodreads, P_c 2k–32k = 0.25–4%); arXiv C0 V3 1.31 ms vs V1 1.42 at recall 0.68 | bit-width sweep {64,128,256,512}; the 1%-retained point; check which bit width the thesis used **[check `linr_v3` params]** |
| L4 | 1B × 64d fp16 (120 GB) → 7.5 GB at 64-bit codes; single-query top-50M on one A100 peaks 21 GB, **97.6 ms p95** | Partially (synthetic 1B×64 codes = 8 GB fits; fp16 source does not) | ceiling 5.37M | R7 ladder with 1-bit only at ≥ 250M; top-k at k=50M is a `torch.topk` stress test |
| L5 | V1/V2 handle up to **240M × 128d fp16** on one A100 for top-2k single query | Partially (61 GB fp16 fits an 80 GB card) | — | one synthetic run; report peak memory and latency |
| L6 | Live update at 0/300/600 upserts/s: no measurable latency impact (218 QPS bs1, 4.57 / 4.79 ms) | No (no upsert API yet) | — | out of scope; cite `live-update-api.md` as future work |
| L7 | Native TF/PyTorch masking/indexing is **100× slower** than the custom CUDA filter kernel | Yes — directly | fig:backend_speedup (Triton vs torch, median) — numbers live only in the results JSONs **[extract]** | report torch-eager vs torch-compiled vs Triton vs CUDA; expect far less than 100× under `torch.compile` — a headline finding either way |
| L8 | Pre-filtering fixes the liquidity problem of post-filtered ANN; A/B +7% interactions, +6% from freshness | Partially | fig:selectivity_recall: QuantizedIVF recall 0.88 → 0.76 from C0 to C5 while V2 stays ≈ 1.0 | quantify liquidity as "queries with < K passing candidates in probed clusters" vs n_probe; A/B is unreproducible |
| L9 | MoL / Hadamard offline gains (+10–23% HR@400); fixed clusters beat trained | No | — | out of scope |
| L10 | All reported numbers use *un-fused* individual ops (§6) | deviation | thesis kernels are fused (`fused_masked_knn_topk`, `clause_compact`) | state it; run an un-fused torch variant as the paper-faithful point |
| L11 | Metrics: avg + p95 latency, recall label@2000 | deviation | median/p20/p80, recall@100/500/1000 vs oracle | add mean, p95, p99 and K = 2000 |

### B.3 Prioritised gap list (effort; A100?)

**P0 — required for any submission (≈ 10 A100-days + 2 weeks of writing/harness work)**

| # | Gap | Effort | GPU |
|---|---|---|---|
| G1 | Reframe QuantizedIVF as SilverTorch Alg. 1; write the deviations table: `(clause_idx,value)` bloom salting; exact-predicate mode (not in paper); global int8 scale; k-means init/iters; row-wise sigs (Triton) vs transposed (CUDA); host `torch.topk` epilogue; no DSL / NOT / nesting; fused V2 compaction; `torch.compile(reduce-overhead)` | 1 day | no |
| G2 | **Run the official SilverTorch ops** (`bloom_index_build/search_batch`, `fused_kmean_ann` int8 with `filtering_bit_mask`, `…_with_partial_masks`) on arXiv/Goodreads with our clusters and codes; check bit-exactness of scores, compare latency at matched `n_probe`; document build friction (Python 3.10–3.13, torch 2.4–2.10, CUDA 12.1/12.4/12.8 per README — our torch 2.10 is in range) | 2–4 days | yes |
| G3 | Harness stats: mean, p95, p99 (p999), per-query latency vectors, closed-loop QPS = B/latency and an open-loop max-QPS-under-200 ms-P99 sweep over B (roadmap 4a.1–3, 4a.7) | harness 1–2 days (other agent); reruns 2 days | yes |
| G4 | Multi-seed × 5 (k-means, OPORP, query pool) on headline cells; bootstrap CIs over queries for recall, over repetitions for latency; paired tests for every "A faster than B" sentence (roadmap 4b.3) | 1 day harness + 1–2 days runs | yes |
| G5 | External baselines at matched recall on the same box: Faiss-GPU `IndexIVFFlat` fp32, Faiss-CPU IVFFlat (64 threads), HNSW (hnswlib/Faiss), brute-force cuBLAS fp16 floor (= LiNR V1 no-filter, already there), K ∈ {100, 1000, 2048}; filtering for Faiss via `IDSelector`/mask | 2–3 days | yes |
| G6 | Bloom FPR + memory vs m_bits ∈ {512, 768, 1024, 2048}; `BloomFilter` vs `ExactAttributeFilter` vs CPU inverted index microbench (S7/S8; `future-work` R8) | 1–2 days | yes |
| G7 | Goodreads d128/d256 filter re-run on the fresh oracle (roadmap §2 blocker) | 1 day | yes |
| G8 | Cross-dataset deep sweeps (`silvertorch` on Goodreads, `linr_v3` on arXiv; roadmap 4b.7) so no figure is single-dataset | 1 day | yes |
| G9 | Provenance in every row (`extra.gpu/torch/commit` exist; add driver, CUDA, triton, clocks, `nvidia-smi -q` dump) and a hardware/software disclosure box | hours | no |

**P1 — needed for a SIGIR-grade paper (≈ 3 more weeks)**

| # | Gap | Effort | GPU |
|---|---|---|---|
| G10 | Scale ladder on one dataset family: real 10M (dataset agent) + `synth_arxiv` 15M/30M/50M for QPS scaling (S2/S3), 240M×128 fp16 V1/V2 stress (L5), 1B×64 1-bit (L4) — `future-work` R7 | 1–2 weeks; ≥ 200 GB disk | yes |
| G11 | Controlled pass-rate sweep (synthetic attribute, 0.01%…100%) for the V1/V2 crossover and the liquidity curve (L1/L2/L8) | 2–3 days | yes |
| G12 | Co-design ablation: full bloom mask → IVF vs fused partial bloom, scratch memory + latency vs n_probe (S9) | 1–2 days | yes |
| G13 | cuVS baselines: IVF-Flat / IVF-PQ and CAGRA with bitset prefilter ([cuVS filtering docs](https://docs.rapids.ai/api/cuvs/nightly/filtering/), [CAGRA](https://docs.rapids.ai/api/cuvs/stable/neighbors/cagra/), [cuvs-bench](https://docs.rapids.ai/api/cuvs/stable/cuvs_bench/)); note CAGRA's top-k cap | 2–3 days; build risk | yes |
| G14 | CPU filtered-graph baselines at matched recall — Filtered-DiskANN ([WWW'23](https://dl.acm.org/doi/10.1145/3543507.3583552), [DiskANN](https://github.com/microsoft/DiskANN)), ACORN ([SIGMOD'24](https://arxiv.org/abs/2403.04871)) — or run FANNBench's harness on one of our datasets and cite its numbers for the rest | 3–5 days | CPU box |
| G15 | V3 bit-width sweep + the "keep 1%" operating point (L3); int8 global-vs-per-row scale and K = 2048/4096 recall ceiling (S6/S16) | 1–2 days | yes |
| G16 | Extended batch grid {1,4,16,64,256,1024} and K = 10 (roadmap 4b.4–5) | 1 day | yes |

**P2 — optional / out of scope for this paper:** live update (L6, weeks), multi-GPU (S15),
Milvus/Qdrant filtered-HNSW end-to-end ([Qdrant](https://qdrant.tech/articles/vector-search-filtering/),
[Milvus bitset](https://milvus.io/docs/bitset.md)) — cite, don't run; energy J/query (I8, 1 day);
GPU IVF-RaBitQ in cuVS ([arXiv 2602.23999](https://arxiv.org/html/2602.23999v1)) as a forward pointer.

### B.4 Scope declarations the paper must make explicitly

Not reproduced, stated in §3 of the paper with a one-line reason each: OverArch scoring and the
in-model Value Model (S11); multi-task 12-head queries and the AND/OR/NOT DSL (S1); scale-out and
TCO (S4, S15); fresh index / live update (S14, L6); MoL/Hadamard similarity models and cluster
training (L9); A/B results and the +6% freshness lift (L8); TorchScript/native serving stack (LiNR
§4.4); LiNR's TensorFlow implementation. What *is* reproduced: SilverTorch's Bloom index, fused
INT8 IVF search and the co-designed Algorithm 1; LiNR's V1 (masking), V2 (explicit pre-filter),
V3 (Sign-OPORP 1-bit + rescoring); added beyond both papers: LiNR-V4 (int8 exhaustive), an exact
fused predicate for SilverTorch, and four backends (torch, Triton, CUDA C++, CuTe DSL) with
bit-exact parity — the last is the paper's "new insight" hook for SIGIR.

### B.5 Framing: recommendation vs. semantic search

LiNR is feed/jobs recommendation (two-tower member/item embeddings, exhaustive search);
SilverTorch is ads/recsys retrieval (User Tower → ANN → OverArch). The thesis mixes recsys
(Goodreads, Yambda; gSASRec user-history queries) with text similarity (arXiv; Nomic-Embed,
item-as-query). Recommended framing: **"attribute-filtered candidate retrieval on GPUs"** with
recommendation as the primary domain and arXiv as the *generalisation* axis (RQ4: do the
conclusions transfer to a text-embedding space with a different attribute/pass-rate distribution
and a different query distribution?). This is exactly the SIGIR'26 ConstBERT/ColBERT-v2 move and
is acceptable at ECIR/SIGIR; at RecSys, drop arXiv. State in the paper that the recall targets
differ by mode (oracle-relative under filters, interaction-relative without) and never mix them in
one plot.

### B.6 What the dataset search must deliver (requirements only)

- Size: ≥ 10M items (paper regime S2), ideally 20–40M (S5/S7/S9 regimes); fits one 80 GB card
  with fp16 + int8 + 1024-bit sigs + scratch → N ≤ ~60M at d = 128.
- Attributes: ≥ 4 categorical clauses of mixed cardinality (SilverTorch's "broad and dense" recsys
  features *and* one high-cardinality one), ≥ 1 clause that is naturally negated (LiNR's reverse
  company clause), multi-valued per item (SilverTorch: 10 values/item avg), so bloom sizing (S8) is
  non-trivial.
- Queries: per-query constraints derived from user context, not random; pass-rate distribution
  spanning 0.1%–50% (Big-ANN/FANNBench convention) with a documented histogram; ≥ 10k queries.
- Ground truth: exact filtered top-K (K up to 2048) computable in fp32 on the box; for the
  no-filter mode, held-out interactions with a temporal split.
- Embeddings: either trainable (two-tower/gSASRec, so d ∈ {64,128,256} are all available) or
  precomputed with a stated encoder; licence must permit redistributing derived embeddings and
  attributes (HF dataset).
- Cheapest known candidate to check first: Yambda-5B's own 9.39M tracks with artist / album /
  duration-bucket / release-year as clauses (metadata already in the release,
  `articles/yambda.md`) — turns the existing unfiltered 5.37M set into a ~9.4M filtered recsys set.
  YFCC-10M (Big-ANN filter track: CLIP, 200k-tag vocabulary, 100k queries with 1–2 tags) is the
  comparability anchor for the filtered-ANN community but is not recsys.

### B.7 Artifact packaging the paper needs

- `torchretrieve` on PyPI pinned to a tagged release matching the paper's commit; git tag `paper-ecir27`.
- Zenodo (DOI) snapshot of repo + `evaluation/results/**/*.json` + configs — required for any ACM
  badge and for ECIR "permanent repository" wording.
- HF datasets: preprocessed item embeddings per d, `item_attrs_narrow.pt`, query embeddings/attrs,
  oracle top-K (`gt_topk_v3_*.pt`) per sweep; HF checkpoints for gSASRec already exist
  (`docs/system/checkpoints.md`). Check Goodreads (UCSD terms), arXiv (Kaggle metadata CC0) and
  Yambda licences before uploading derived data.
- One-command reproduction: `uv sync && just reproduce-paper` → downloads from HF, runs every cell
  in the paper's YAMLs, regenerates every figure from JSON; a `--from-json` mode that rebuilds all
  figures without a GPU. Pinned `uv.lock`, Dockerfile with CUDA 12.8 / torch 2.10 / triton 3.6.
- Anonymised mirror (e.g. anonymous.4open.science) for double-blind submission; no PyPI name in the PDF.
- Reproducibility checklist appendix (hardware, software, seeds, clocks, warm-up, timing method,
  number of repetitions, CI method, dataset versions, licences).

---

## Part C — Proposed paper

### C.1 Title options

1. *Which SilverTorch Do You Mean? Reproducing Industrial GPU Retrieval with Attribute Filtering on Public Data*
2. *Bloom, Probe, Quantize: An Independent Reproduction of SilverTorch and LiNR*
3. *Model-Based Retrieval on GPUs, Reproduced: LiNR and SilverTorch Beyond Proprietary Data*
4. *What Survives at 10M Items? A Reproducibility Study of Two Industrial GPU Retrieval Systems*

### C.2 Abstract skeleton (≈ 180 words)

Context (industrial GPU retrieval with in-model filtering: LiNR CIKM'24, SilverTorch 2025;
proprietary data, partial code) → what we did (independent PyTorch/Triton re-implementation of
both, four backends bit-exact, public datasets 0.8M–{10M+}, oracle-relative protocol, official
SilverTorch kernels as a check) → what held (int8 fused IVF Pareto dominance, ≈2× co-design, bloom
memory/latency, liquidity of post-filtering) → what did not / deviated (V1-vs-V2 crossover flips
with fused compaction; transposed bloom win only with memory-level parallelism; INT8 recall ceiling
numbers; 100× masking claim under `torch.compile`) → generalisation (embedding source, d, batch,
backend, catalog size) → baselines (Faiss GPU/CPU, HNSW, cuVS) → artifact (`torchretrieve`, HF data,
one-command reproduction).

### C.3 Research questions

- **RQ1 (mechanism replicability):** Do SilverTorch's three reported effects — fused INT8 IVF vs
  Faiss-GPU fp32 (S5), Bloom vs forward/inverted index (S7/S8), co-design vs separate filtering (S9) —
  reproduce on public data with an independent implementation, and does the authors' released
  kernel agree with ours (G2)?
- **RQ2 (quality):** Do the INT8 recall ceiling (S6) and LiNR's 1-bit/rescoring trade-off (L3) hold
  against an exact filtered oracle and against held-out interactions?
- **RQ3 (filtering regimes):** Where is LiNR's V1/V2 crossover in pass rate (L1/L2), and how does
  the liquidity problem manifest for IVF+Bloom vs exhaustive pre-filtering (L8) across selectivity?
- **RQ4 (generalisation):** Do the conclusions transfer across embedding sources (sequential-recsys
  vs text), d ∈ {64,128,256}, batch sizes, catalog sizes 0.8M → 50M, and implementation backends
  (torch / Triton / CUDA / CuTe)? Is LiNR's "native masking is 100× slower" (L7) still true?
- **RQ5 (baselines):** At fixed recall {0.90, 0.95, 0.99}, how do the reproduced systems compare
  with Faiss-GPU/CPU IVF, HNSW, cuVS CAGRA/IVF and (CPU) filtered graph indexes?

### C.4 Section plan — ECIR (12 LNCS pages) with the SIGIR 9-page ACM compression noted

| § | Content | Pages (ECIR) | Existing material | New experiments |
|---|---|---|---|---|
| 1 Introduction | why these two papers (industrial, no/partial code, proprietary data, contradictory claims about pre-filtering cost); contributions list; RQs | 1.25 | thesis intro | — |
| 2 Background | LiNR V1–V3, SilverTorch Bloom/INT8 IVF/Alg. 1 in one page; related reproducibility work (§A.3) and filtered-ANN benchmarks | 1.5 (SIGIR: 1) | thesis ch. 2–3, `future-work` §Sources | — |
| 3 Reproduction methodology | what was reimplemented, the deviations table (G1), the official-kernel check (G2), scope declarations (§B.4), the "which SilverTorch" variant table (S13/S16) | 1.5 | `filtering.md`, `architecture.md`, handoff §13 | G2 |
| 4 Experimental setup | datasets (+ the new large one), clauses and pass-rate histograms, oracle protocol, metrics incl. p95/p99/QPS, hardware/software box, seeds/CIs/tests | 1.5 (SIGIR: 1) | thesis ch. 5 | G3, G4, G9 |
| 5.1 RQ1 | Pareto latency-vs-recall with Faiss/HNSW/cuVS overlays (fig:silvertorch_pareto reborn); official vs ours bit-exact + latency; bloom FPR/memory vs bits; co-design ablation | 1.5 | fig 6.7, handoff §13 | G5, G6, G12, G13 |
| 5.2 RQ2 | recall vs n_probe asymptote at K=2048; V3 bits × P_c; no-filter interaction recall (tab:recall_nofilter) | 0.75 | tab 6.1, fig 6.6 | G15 |
| 5.3 RQ3 | pass-rate sweep: V1/V2/V3/IVF latency and liquidity vs selectivity (fig:selectivity_recall with a quantitative x-axis) | 1 | fig 6.5, tab 6.2/6.3 | G11 |
| 5.4 RQ4 | backend figure (torch-eager/compiled/Triton/CUDA/CuTe; L7), d-scaling, batch/QPS, scale ladder | 1.25 | fig 4.1, 6.3, 6.9; cute §5.1 table | G10, G16 |
| 5.5 RQ5 | QPS-at-recall table vs all baselines, single GPU; honest "what we could not compare" | 0.75 | — | G5, G13, G14 |
| 6 Claims table + lessons | one table per paper (S#/L#: reproduced / partially / not / contradicted / out of scope); lessons for authors of industrial systems papers; limitations | 1 | §B.1–B.2 | — |
| 7 Conclusion + artifact + checklist | 0.5 | §B.7 | — |

Figures that survive from the thesis (re-rendered in English, with CIs): 6.1, 6.3, 6.5, 6.6, 6.7,
6.9, 4.1; tables 6.1–6.5 fold into the claims tables. New figures: official-vs-ours, FPR-vs-bits,
co-design ablation, pass-rate crossover, scale ladder, QPS-at-recall baseline table.

### C.5 Minimal cut for the ECIR deadline vs. full SIGIR version

ECIR (19 Oct 2026): P0 only — sections 5.1 without cuVS/graph baselines (Faiss + HNSW suffice),
5.4 without the 240M/1B points, RQ5 as a table with Faiss/HNSW only. SIGIR (≈ Feb 2027): add G10–G16,
lead with the backend/scale generalisation as the "new insight", compress §2 and §4.

## Sources not linked inline above

SilverTorch paper https://arxiv.org/abs/2511.14881 · LiNR https://doi.org/10.1145/3627673.3680091 ·
Big-ANN'23 filter track https://big-ann-benchmarks.com/neurips23.html · VecFlow SIGMOD'25
https://arxiv.org/abs/2506.00812 · Milvus filtered search https://milvus.io/docs/filtered-search.md ·
ACM badging policy https://www.acm.org/publications/policies/artifact-review-and-badging-current ·
SIGIR badging https://sigir.org/general-information/acm-sigir-artifact-badging/ ·
KDD 2027 D&B https://kdd2027.kdd.org/datasets-and-benchmarks-track-call-for-papers/ ·
RecSys 2026 second call (track chairs, contact) https://mailman1.isti.cnr.it/hyperkitty/list/hcitaly@isti.cnr.it/thread/GJSJZJS724LOMJP4ZILMYOCJW6DM55GW/
