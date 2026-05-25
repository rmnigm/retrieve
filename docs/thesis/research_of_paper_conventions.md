# Structuring a Reproducibility & Benchmark Paper for GPU‑Native Retrieval (LiNR + SilverTorch in Triton): A Field Guide for A*/Systems Venues

## TL;DR
- **Position the paper as a benchmarking/measurement contribution, not a re‑implementation**: target NeurIPS Datasets & Benchmarks (D&B) or MLSys, frame LiNR and SilverTorch as the systems‑under‑test, and contribute (1) the first open Triton reference implementation, (2) a filtered‑ANN evaluation protocol that fixes the LiNR/SilverTorch papers' reliance on proprietary data, and (3) Pareto curves on public datasets (YFCC100M‑CLIP, MS‑Turing, DEEP, MS MARCO, LAION) — neither original paper releases code or data, so a faithful open reproduction is itself a publishable artifact.
- **Adopt a hybrid skeleton**: ML Reproducibility Challenge / ReScience C scope+claims structure for the front matter, ANN‑Benchmarks (Aumüller, Bernhardsson, Faithfull) evaluation grammar (recall‑QPS Pareto, per‑algorithm parameter sweeps, batch vs. single‑query) for the core experiments, and FlashAttention/Liger‑Kernel‑style kernel sections (roofline, memory traffic, kernel‑latency ablations) for the Triton depth — plus a Big‑ANN 2023 filtered‑track style protocol (10‑recall@10 ≥ 0.9 cutoff, fixed hardware) for the filtering scenarios.
- **The deltas that will lift you to A***: (i) explicit "reproducibility scorecard" mapping every claim in LiNR (e.g., "4 ms latency at 15M–1B entries", "240M embeddings on a single A100 at top‑2k", "+3% gold professional interactors") and SilverTorch (e.g., "23.7× QPS, 13.35× cost efficiency, P99 ≈ 15 ms") to "reproduced / partially reproduced / could not reproduce due to proprietary data"; (ii) variance and significance treatment that the originals lack (5+ seeds, bootstrap CIs on recall, paired Wilcoxon on QPS); (iii) a release artifact (Docker + Croissant + Big‑ANN style runner) reviewers can `make benchmark` to regenerate every plot.

## Key Findings

1. **There is no single "reproducibility paper" template — there are three traditions**, and an A*‑grade paper must blend them:
   - The **ML Reproducibility Challenge / ReScience C tradition** (Pineau, Rougier–Hinsen, Dodge): a 4‑page "Reproducibility Summary" up front (Scope of Reproducibility, Methodology, Results, What was easy, What was difficult, Communication with original authors), then a paper‑shaped body. The MLRC 2023 organizers' blog post is explicit that reviewers reward the *holistic* approach over sweeping verdicts: *"we consistently found the quality of the reports submitted to the challenge fall into either of these two categories: a) making a sweeping claim about reproducibility, or b) diving deep and constructing a holistic view of reproducibility, replicability and generalisability of the claims presented in the original paper. Not surprisingly, the latter cohort is always highly rated by the reviewers and ends up more often in the accepted pool."*
   - The **RecSys "Are We Really Making Much Progress?" tradition** (Ferrari Dacrema, Cremonesi, Jannach, RecSys '19, **Best Long Paper Award** at the 13th ACM Conference on Recommender Systems, Copenhagen, Sept 16–20, 2019, doi:10.1145/3298689.3347058; extended in TOIS '21): adversarial reproduction with strong, tuned, simple baselines. Section structure: §1 Intro; §2 Research method (paper selection criteria, reproducibility criteria — code availability, dataset availability, preprocessing scripts, hyperparameters); §3 Results table with per‑paper, per‑metric reproduced numbers next to reported numbers; §4 Discussion of *why* numbers diverged (baseline tuning, evaluation leakage, dataset preprocessing). The killer technique is including under‑tuned non‑neural baselines (TopPopular, ItemKNN, RP3β) that beat the reproduced "SOTA" — for a systems paper this maps to including brute‑force GPU dot‑product and Faiss‑GPU IVF‑Flat as the "Are we really faster than the obvious thing?" baselines.
   - The **ANN‑Benchmarks tradition** (Aumüller et al., SISAP'17 / Information Systems '20; extended by Big‑ANN'21 and Big‑ANN'23 at NeurIPS): YAML‑driven config sweeps per algorithm, recall–QPS Pareto plots as the *only* fair comparison, Docker isolation, batch‑mode vs. single‑query separation. This is the de facto standard for ANN papers and any paper that uses anything else for headline numbers will be rejected.

2. **The LiNR and SilverTorch papers are reproduction‑hostile by design and that is your contribution**:
   - **LiNR** (Borisyuk et al., CIKM '24, arXiv 2407.13218) reports only on a *~15.5M LinkedIn jobs internal dataset with 25K queries* and an *internal "out‑of‑network feed" A/B test*; the headline "**4 ms** latency for indexes from **15M to a billion entries**" and "**240M embeddings handled on a single A100 at fp16, dim=128, top‑2k**" numbers are all on proprietary data. The "+3%" claim is specifically "Daily Unique Gold Professional Interactors" (a LinkedIn internal metric), not generic DAU; total interactions lift was +7%. **No code or data is released.**
   - **SilverTorch** (Xue et al., SIGIR '26, arXiv 2511.14881) evaluates on two Meta‑internal samples (10M and 80M items, dim=128, Int8) on A100‑40G. Headline numbers from v5 (Table 2): **23.7× QPS over CPU baseline** and **13.35× cost efficiency** for the OverArch variant, P99 latency **~15 ms** independent of traffic. Their ANN+filter co‑design improves QPS **17–25%** vs. naive filter‑then‑probe, and the Bloom Index is **291×–523× faster than inverted index** and **12.6×–42.7× faster than forward index** on 40M items at 5 hash functions. **No code or data is released.** Earlier versions (v1) of the paper used different numbers (e.g., "13.85× ESR", "1.58× batching") that were revised in v5 — flag in your paper which version you reproduce against.
   - Both papers compare against **Faiss‑GPU IVF‑Flat** as the principal GPU baseline; SilverTorch also references HNSW (Faiss impl.) and explicitly excludes CAGRA because "CAGRA limits top‑k to 1024, making it unsuitable for recommendation scenarios."

3. **The filtered‑ANN angle is your strongest novelty axis** because the Big‑ANN 2023 NeurIPS filtered track (Simhadri et al., arXiv 2409.17424) standardized exactly the predicate‑search workload that LiNR's ABM (Attribute‑Based Matching) and SilverTorch's Bloom Index were designed for, but no published reproduction connects the production systems to the public benchmark:
   - Big‑ANN'23 Filtered track dataset: **YFCC100M 10M slice, 192‑dim uint8 CLIP embeddings, ℓ₂ distance, 100K queries each with 1 or 2 tags from a 200,386‑tag vocabulary, AND semantics**.
   - Standard hardware: **Azure D8lds_v5 (8 vCPU, 16 GB RAM)** — CPU only, 12‑hour index build limit.
   - Headline metric: **highest QPS at 10‑recall@10 ≥ 0.9**. Winner: **ParlayANN** (UMD, 37,671 QPS private) — 11× the Faiss baseline. Runner‑up: **Baidu Puck**.
   - Other filtered‑ANN baselines to include: **ACORN** (Patel et al., SIGMOD '24, predicate‑agnostic HNSW extension), **Filtered‑DiskANN / Stitched‑Vamana** (Gollapudi et al., WWW '23), **NHQ** (Wang et al., 2023), **UNG** (SIGMOD '25), **SeRF** (SIGMOD '24, range filters), **iRangeGraph / DIGRA** (range filters), and **CAPS / HQANN**. Your paper should map LiNR's ABM (Similarity Masking V1 / Pre‑Filtering V2 / Quantized V3) and SilverTorch's Bloom Index onto the **pre‑filter / post‑filter / inline‑filter** taxonomy used in this literature.

4. **Kernel‑level evaluation conventions follow FlashAttention/Liger‑Kernel, not pure systems papers**:
   - FlashAttention (Dao et al., NeurIPS '22) and FlashAttention‑3 (Shah et al., NeurIPS '24, arXiv 2407.08608) set the template: **(a)** algorithmic claim with IO‑complexity argument; **(b)** kernel‑level latency benchmarks vs. a strong baseline (cuDNN, FA2, PyTorch SDPA) reported as **runtime vs. sequence length / head dim** plots; **(c)** roofline / theoretical‑peak achieved — *"FlashAttention‑3 reaches up to 740 TFLOPs/s, 75% of the theoretical maximum TFLOPs/s on H100 GPUs"*; **(d)** ablation of each optimization in isolation (warp‑specialization, GEMM‑softmax pipelining); **(e)** end‑to‑end training/inference speedup; **(f)** numerical validation (FP16 vs. FP8 error tables).
   - Liger‑Kernel (Hsu et al., arXiv 2410.10989, ICML '25 CODE‑ML workshop) is the closest reference shape for a Triton‑focused systems paper: it reports *"on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations"* (benchmarked on LLaMA 3‑8B, batch 8, bf16, FSDP1 on 8 A100s), includes per‑kernel latency benchmarks (GeGLU, RoPE) and integration tests. Your paper's Triton sections should mirror this: per‑kernel microbenchmark, then end‑to‑end retrieval pipeline.

## Details

### 1. Reproducibility paper structure — the conventions

The dominant template for ML reproducibility papers (used by MLRC since 2020 and required for ReScience C MLRC special issues) is:

| Section | Purpose | Notes for the user |
|---|---|---|
| **Reproducibility Summary** (½‑1 page, *front matter*) | Scope of Reproducibility · Methodology · Results · What was easy · What was difficult · Communication with original authors | Mandatory for MLRC/ReScience C. Lifts paper from "blog post" to "audit". For you: list LiNR/SilverTorch claims being tested verbatim. |
| **Introduction** | Motivation; why this paper, why now | Frame as "first open reimplementation"; cite Ferrari Dacrema et al. for the phantom‑progress framing. |
| **Scope of Reproducibility / Claims** | Verbatim quote each claim being tested | E.g., LiNR claim C1: "4 ms latency, 15M–1B entries"; SilverTorch claim S1: "23.7× QPS"; map each to "in‑scope (public‑data reproducible)" / "out‑of‑scope (requires proprietary data)". |
| **Background / Original Work Summary** | Compact restatement of the methods being reproduced | One subsection per system; explicit notation matching the original. |
| **Methodology** | Implementation choices, deviations, environment | Triton version, CUDA version, GPU SKU, driver, PyTorch version, seeds, Docker image hash. |
| **Datasets & Protocol** | Which datasets, which queries, ground‑truth generation, evaluation hardware | This is what reviewers grade hardest. See §5 below. |
| **Results** | Side‑by‑side reported vs. reproduced tables; recall‑QPS Pareto plots | Every table column should have a 95% CI from ≥5 seeds. |
| **Discussion of discrepancies** | Where you diverge from the original and why | Quantify: e.g., "our reproduced top‑2k recall is 0.681 vs. reported 0.688; gap explained by …". |
| **Beyond reproduction** | Ablations, sensitivity, new findings | The "extension" that lifts a reproducibility paper to A*. |
| **Threats to validity** | Hardware, seed, data drift, version skew | Standard SE practice; absent in most ML papers — including it signals rigor. |
| **Conclusion / Recommendations to original authors** | What the community should change | Cite Ferrari Dacrema et al. for the convention. |

**Reproducibility scorecards**: the convention popularized by Pineau's ML Reproducibility Checklist (NeurIPS '19+) and the Papers With Code reproducibility checklist is a **per‑claim table** with columns *Claim / Source (§ in original paper) / Reported value / Our value / 95% CI / Verdict (✓ / partial / ✗ / out‑of‑scope) / Notes*. ReScience C MLRC papers print this as Table 1 of the body. For LiNR/SilverTorch you'll have ~12–15 rows.

**Deviations from original setup** are handled by an explicit "Differences from original setup" subsection in Methodology, with a table whose rows are: (i) hardware (LiNR/SilverTorch used A100‑40G; you may have A100‑80G or H100 — note this and run multi‑SKU when possible), (ii) datasets (proprietary → public substitute, with justification of *why* the substitute is comparable along dim, n, distribution), (iii) hyperparameters (when missing from original, document grid search range and selection criterion), (iv) software stack. The Ferrari Dacrema et al. study established that the dataset‑preprocessing step is the single biggest unreported variable in RecSys papers; the analogue for ANN papers is **how the ground‑truth top‑k is computed** (brute‑force float32, or distilled from a strong index — Big‑ANN '23 distributes ground truth as files, you should too).

**Partial / closed source**: when only some components are released (LiNR releases none; SilverTorch releases none), the convention pioneered by Ferrari Dacrema et al. is to (a) reimplement from paper + supplementary material + correspondence with authors, (b) attempt to contact authors and report whether they responded and what they confirmed, (c) publish your reimplementation under the same terms you'd publish a new method. The MLRC explicitly accepts "I could not reproduce X because the missing piece is Y" as a positive contribution.

**Environment documentation best practices** (cross‑venue consensus): Dockerfile + `pip freeze` lock file + git SHA + GPU model + CUDA driver + CUDA toolkit + cuDNN + Triton/PyTorch versions + seed schedule + total wall‑clock time per experiment + power draw if available. MLSys 2024+ requires Artifact Description (AD) and Artifact Evaluation (AE) appendices with this content; submit to the ACM AE process (badges: Available, Functional, Reusable, Reproduced).

**High‑quality exemplars to study**:
- Ferrari Dacrema, Cremonesi, Jannach, "Are We Really Making Much Progress?" RecSys 2019, **Best Long Paper Award** (doi:10.1145/3298689.3347058). (And the 2021 TOIS extension, "A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research.")
- The Big‑ANN '21 and '23 results papers (Simhadri et al., NeurIPS) — gold standard for benchmark‑track writing.
- ANN‑Benchmarks paper (Aumüller, Bernhardsson, Faithfull, *Information Systems* 2020).
- Beel et al., "Towards reproducibility in recommender‑systems research", UMUAI 2016.
- Armstrong, Moffat, Webber, Zobel, "Improvements That Don't Add Up: Ad‑hoc Retrieval Results Since 1998", CIKM '09 — the IR ancestor of this entire genre.

### 2. ANN benchmark paper structure

The ANN‑Benchmarks paper (Aumüller et al.) is the structural canon. Its sections are: §1 Intro · §2 Problem & quality measures · §3 System (algorithm integration interface, parameter sweep mechanism, batch‑mode API) · §4 Evaluation (datasets, hardware, results) · §5 Conclusions. Its plots are: (a) **recall‑QPS scatter / Pareto frontier** per dataset, log‑scale Y; (b) **per‑algorithm parameter trajectory** showing how a single algorithm's points are obtained by varying e.g. `efSearch`; (c) **build‑time vs. recall** trade‑off; (d) **index size vs. recall**; (e) **batch QPS vs. single‑query QPS**. CAGRA (Ootomo et al., ICDE '24) and most subsequent GPU‑ANN papers replicate this skeleton.

**Standard metrics with definitions**:
- **Recall@k** = |R_k ∩ G_k| / k, where G_k is the brute‑force float32 top‑k. Big‑ANN'23 standardizes **10‑recall@10** (k = k' = 10) and requires ≥ 0.9 for the QPS leaderboard.
- **QPS**: queries per second under either single‑query (batch=1) or batch mode (e.g. batch=10K all at once). Both must be reported separately — Aumüller et al. explicitly warn that batch‑mode favors GPU implementations and is the *only* fair regime for GPU benchmarks.
- **Build time** (wall‑clock to build the index from raw vectors), **build memory** (peak RSS), **index size on disk**.
- **Memory footprint at serve time** (peak GPU memory).
- **Latency percentiles** — for production‑oriented papers, p50/p95/p99. LiNR reports p95 throughout; SilverTorch reports P99 (~15 ms on 80M).
- **Recall‑constrained QPS** — single number for leaderboards: QPS at recall ≥ τ (Big‑ANN uses τ = 0.9 for filtered/OOD/sparse).
- For retrieval/ranking downstream: **MRR@k**, **nDCG@k**, **Hit Rate @ k** (LiNR reports HR@400).

**Canonical plots** (must‑have for ANN papers):
1. Recall–QPS Pareto, one curve per algorithm, one panel per dataset.
2. Recall–latency (1/QPS) at batch=1 — different ranking than QPS.
3. Build time vs. recall at fixed QPS.
4. Index memory vs. recall.
5. **Scaling plot**: QPS or latency vs. dataset size (1M → 10M → 100M → 1B) at fixed recall.
6. **GPU utilization vs. batch size** (for GPU papers).
7. **Recall vs. predicate selectivity** (for filtered ANN — Filtered‑DiskANN and ACORN both show this).
8. For kernel papers: **per‑kernel latency bar chart vs. baseline kernel**, and a **roofline plot** placing your kernel relative to the GPU's compute/memory bound.

**Hardware reporting standard** (from the Big‑ANN '21/'23 papers and MLSys conventions): GPU SKU (e.g. "NVIDIA A100‑SXM4‑40GB"), GPU count, host CPU, system RAM, PCIe vs. NVLink topology, CUDA toolkit version, CUDA driver, GPU clock policy (locked vs. boost), power cap, OS/kernel version, Triton commit hash, PyTorch version, `cublasLt` version. A reusable artifact lists this in `hardware.yaml`.

**Hyperparameter sweeps**: the ANN‑Benchmarks convention is that each algorithm exposes a list of (build_args × query_args); the system runs every combination, plots the Pareto envelope and discards dominated points. For HNSW: `M ∈ {8, 16, 24, 32, 48, 64}`, `efConstruction ∈ {100, 200, 400, 500}`, `efSearch ∈ {10, 20, …, 800}`. For IVF: `nlist ∈ {1024, 4096, 16384, 65536}` (rule of thumb √n), `nprobe ∈ {1, 2, 4, …, 256}`. For DiskANN/Vamana: `R ∈ {32, 64, 96, 128}`, `L ∈ {50, 75, 100, 125}`, search `L_search ∈ {10…400}`. For CAGRA: `graph_degree ∈ {32, 64}`, `itopk_size ∈ {64, 128, 256, 512}`, `search_width ∈ {1, 2, 4, 8}`. **Always report the sweep range, not just the best point.**

**Filtered/predicated ANN benchmark structure** (this is your differentiated section):
- Define predicate semantics formally: **point/equality**, **conjunctive (AND)**, **disjunctive (OR)**, **range**, **set‑membership**, **negation/reverse**. LiNR ABM supports point + reverse; SilverTorch Bloom supports conjunctive multi‑attribute; Filtered‑DiskANN supports single equality; ACORN supports arbitrary predicates.
- Define **selectivity** (= fraction of items passing the predicate); report results binned by selectivity decile. ACORN's evaluation set this convention and Big‑ANN '23 followed it with low/medium/high selectivity buckets.
- Three implementation strategies must each be in your table: **pre‑filter** (filter then exact search the residual), **post‑filter** (search then drop), **inline / predicate‑aware** (ACORN, Filtered‑DiskANN, NHQ, UNG, LiNR ABM, SilverTorch Bloom). The user's contribution is to add a fourth column: **co‑designed** (SilverTorch ANN+filter fusion).
- Metric: **filtered recall@k** = |R_k ∩ G_k| / k where G_k is the brute‑force top‑k *within the predicate's truth set*. This is critical and often miscomputed — a common bug is to use the unfiltered ground truth.
- Standard datasets: **YFCC100M** (10M slice, 192‑d CLIP, tag predicates) is the Big‑ANN '23 filtered‑track dataset; **MS Turing‑ANNS** (10M slice, 100‑d) is the OOD‑track dataset and has been used for filtered variants by adding synthetic categorical attributes; **SIFT1M + synthetic labels** is the lightweight default used by ACORN, Filtered‑DiskANN, and NHQ for development; **LAION‑400M / LAION‑2B with CLIP attributes** is increasingly the LLM‑era default.

### 3. Retrieval algorithm paper structure (GPU‑native variant)

GPU retrieval system papers cluster into two structural families:

**(a) Algorithm‑centric** (ScaNN, DiskANN/Vamana, HNSW, CAGRA, SPANN): Intro → Background/Related Work → Algorithm (with pseudocode and complexity argument) → Implementation details → Experiments (Pareto curves vs. SOTA on standard benchmarks) → Conclusion. ScaNN (Guo et al., ICML '20) is short and tight (algorithm + ann‑benchmarks results). CAGRA (Ootomo et al., ICDE '24) adds GPU‑specific sections on memory hierarchy, warp‑level cooperation, k‑selection on Tensor Cores.

**(b) Systems‑centric** (LiNR, SilverTorch, FAISS library paper Douze et al. 2024, Milvus, Pinecone tech reports): Intro → Motivation (industry pain point) → System architecture (block diagram with retriever/ingestor/serving stack) → Key components (sub‑sections per novel piece: ABM, MoL clustering, OverArch, Bloom Index, fused Int8 kernel) → Production deployment lessons → Online A/B test results → Conclusion. LiNR and SilverTorch are exemplars of family (b); they trade algorithmic novelty for production‑scale validation.

**The LiNR paper's exact structure** (Borisyuk et al., CIKM '24): §1 Introduction · §2 Related Work · §3 Modeling Technology (§3.1 Exhaustive Search with ABM — §3.1.1 KNN with Similarity Masking, §3.1.2 KNN with Explicit Pre‑Filtering; §3.2 Quantized KNN; §3.3 Similarity Modeling — §3.3.1 Hadamard MLP, §3.3.2 Mixture‑of‑Logits with Clustering) · §4 System Architecture (§4.1 OON Recommendations, §4.2 ML Infra, §4.3 Model Live Update, §4.4 Inference on Native Stack) · §5 Experiments (§5.1 Offline, §5.2 A/B test, §5.3 Inference Benchmarking with high‑pass‑rate / low‑pass‑rate / live‑update sub‑experiments) · §6 Deployment Lessons · §7 Conclusion.

**The SilverTorch paper's exact structure** (Xue et al., SIGIR '26): §1 Intro · §2 Background & Motivation (§2.1 Related Work, §2.2 Limitation of Service‑based Retrieval) · §3 SilverTorch Overview · §4 Model Design (§4.1 Bloom Index, §4.2 Fused Int8 ANN, §4.3 ANN+Filtering Co‑design) · §5 Extensibility (§5.1 OverArch, §5.2 Multi‑task Value Model, §5.3 Scale Out) · §6 Evaluation (§6.1 End‑to‑end with Throughput/Cost/Latency subs, §6.2 Breakdown for ANN / Bloom / Co‑design / OverArch) · §7 Discussion · §8 Conclusion.

**Production retrieval papers balance algorithm and systems detail** by reserving:
- ~40 % to algorithm/method (with pseudocode and concrete equations)
- ~30 % to system architecture and serving stack (block diagrams, control flow, threading model)
- ~30 % to evaluation (offline + online + microbenchmark)

**Metrics that matter for GPU retrieval** (from LiNR, SilverTorch, FAISS, CAGRA, RAFT/cuVS):
- **Throughput** (QPS), batch and single‑query variants.
- **Latency** distributions (p50/p95/p99). LiNR's "4 ms" is a *single‑query mean* on A100; SilverTorch's "~15 ms P99" is *under load* on A100‑40G (specifically: *"At 32 probes and 10 QPS, SilverTorch has the P99 latency of 15.3ms, a 11.4× improvement over the CPU baseline and a 1.6× improvement over the best GPU baseline."*).
- **Recall@k** at the production k (LiNR uses top‑2k for OON, top‑400 for hit rate; SilverTorch uses top‑1024 to top‑10000 because OverArch wants more).
- **Memory footprint**: GPU memory for index + scratch. SilverTorch's paper reports that on *"an index with 81 million items across 9,000 clusters, using 256 probes processes only 2.3 million items (2.8%), achieving a 30× reduction in both filtering computation and GPU scratch memory"* — this is the only precise scratch‑reduction multiplier given; the abstract's language is qualitative ("largely reduces memory utilization").
- **Multi‑GPU scaling**: linear speedup vs. sharding overhead. SilverTorch shards 80M across 2 GPUs.
- **Embedding refresh / live update time**: LiNR explicitly benchmarks 0 / 300 / 600 updates/sec showing no measurable latency hit (4.57 → 4.58 ms mean).
- **Index build cost**: wall‑clock and dollars.
- **Cost efficiency**: $/1000 requests at fixed throughput. SilverTorch reports $0.0077/1000 for retrieval‑only vs. $0.158 for the CPU baseline — 20.9× cheaper.
- **A/B test deltas** on production north‑star metrics (CTR, dwell, DAU).

**Ablation studies for retrieval systems** typically take one of three shapes:
1. **Component knock‑out** ("remove X, measure end‑to‑end") — SilverTorch §6.2 does this for bloom, co‑design, OverArch.
2. **Replacement** ("replace our kernel with cuBLAS / Faiss‑GPU, measure delta").
3. **Hyperparameter sensitivity** (varying nprobe, ef, M, bits‑per‑item, quantization precision) with Pareto curves.
For your paper, the user should do all three.

**Kernel‑level optimizations** are presented in a structural pattern set by FlashAttention and now Liger‑Kernel: (i) describe the kernel with pseudocode in Triton/CUDA notation, naming the tile sizes, shared‑memory layout, and parallelization axes; (ii) report achieved TFLOPs/s vs. peak (roofline placement); (iii) microbenchmark vs. the strongest available baseline (Faiss CUDA k‑select, cuBLAS GEMM + topk, Triton naive) varying problem size; (iv) ablate each optimization (vectorized loads, swizzled shared memory, async copies, warp specialization) individually; (v) integrate into end‑to‑end pipeline and re‑measure. FlashAttention‑3 reports *"740 TFLOPs/s, 75% of the theoretical maximum TFLOPs/s on H100 SXM5 GPUs"* — that kind of number is what a kernel paper aims for. Liger‑Kernel reports *"on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations"* at the model level — that scale of end‑to‑end gain is achievable for fused IVF‑topk + bloom.

### 4. Synthesis and gap analysis for your specific paper

**Must‑have sections** (in order):

1. Abstract (with the 3 numbers that matter most: best recall@10, headline QPS, kernel speedup vs. baseline).
2. Reproducibility Summary (MLRC style; ½ page).
3. Introduction with explicit positioning ("This is a reproducibility+benchmark paper, not a new algorithm").
4. Background: LiNR architecture, SilverTorch architecture, filtered‑ANN landscape (Filtered‑DiskANN / ACORN / NHQ / UNG / Big‑ANN'23).
5. Scope of Reproducibility: enumerated claims from LiNR (C1–C8) and SilverTorch (S1–S10) with in‑scope/out‑of‑scope verdicts.
6. Triton Implementation:
   - 6.1 ABM masking kernel (LiNR V1).
   - 6.2 Pre‑filter slice‑then‑matmul kernel (LiNR V2).
   - 6.3 Sign‑OPORP 1‑bit quantization kernel (LiNR V3).
   - 6.4 Fused Int8 IVF kernel (SilverTorch).
   - 6.5 GPU Bloom Index kernel (SilverTorch) with 64‑bit packed AND.
   - 6.6 ANN + filtering co‑design (SilverTorch §4.3).
7. Evaluation Protocol (this is where most reproduction papers are weak — see §5 below).
8. Datasets:
   - **YFCC100M‑10M CLIP** (Big‑ANN'23 filtered) with tag predicates.
   - **MS Turing‑ANNS 10M** with synthetic categorical attributes.
   - **DEEP1B / DEEP10M** with synthetic clusters.
   - **MS MARCO passage** embeddings with topical tags.
   - **LAION‑400M CLIP** with caption‑derived tags.
   - **GloVe‑100, SIFT1M, GIST1M** as sanity‑check small data.
9. Results:
   - 9.1 Reproduction scorecard (per‑claim verdicts).
   - 9.2 Unfiltered Pareto (recall‑QPS, batch + single).
   - 9.3 Filtered Pareto per selectivity decile.
   - 9.4 Kernel microbenchmarks.
   - 9.5 Multi‑GPU scaling.
   - 9.6 Live‑update overhead.
10. Comparison with Big‑ANN'23 filtered leaders (ParlayANN, Puck) — note hardware delta (Azure CPU vs. A100) prevents direct QPS comparison; report **QPS/$** and **QPS/Watt** instead.
11. Discussion of discrepancies and threats to validity.
12. Recommendations to the community + future work.
13. Appendix: Artifact Description (AD/AE for MLSys).

**What is commonly missing in reproductions** — adding these makes you A*:
- **Variance with seeds**: most reproductions report a single number. Run 5 seeds, report mean ± stddev on every cell.
- **Significance tests**: paired Wilcoxon or bootstrap on QPS differences, ε‑recall tolerance (Aumüller et al. use ε = 0.01).
- **Cost‑normalized comparisons** ($/1M queries on a public cloud SKU). Specifically critical when comparing your GPU implementation to CPU Big‑ANN'23 leaders.
- **Numerical equivalence audit**: bit‑exact comparison of the Triton kernel output vs. a reference cuBLAS+topk implementation; report max abs error and Frobenius‑norm error per layer.
- **Energy reporting** (Joules per query) via `nvidia‑smi` or DCGM — newly fashionable since ML & Climate workshops.
- **Out‑of‑distribution queries** test (Big‑ANN'23 OOD track convention): train on one distribution, query with another.
- **Failure mode catalog**: situations where reproduction failed and why (proprietary preprocessing, hardware unavailability, unspecified hyperparameter).
- **Negative results**: if SilverTorch's Bloom co‑design doesn't beat baseline at low selectivity on YFCC, *say so*. Negative results are publishable and valued at D&B/MLRC.

**Handling proprietary originals**: For LiNR's "+3% gold professional interactors" and SilverTorch's internal‑data‑specific numbers, the convention is:
1. Quote the claim verbatim with a citation.
2. State "out‑of‑scope: requires LinkedIn/Meta production data and traffic; we cannot reproduce."
3. Construct an analogous experiment on public data that tests the *mechanism* even if not the deployment outcome.
4. Reach out to authors; report whether they confirmed or provided guidance.

**Positioning the contributions** (the 4‑bullet contributions paragraph that A* reviewers look for):
1. *First open‑source faithful Triton reimplementation of LiNR and SilverTorch's core kernels (ABM masking V1/V2/V3, Bloom Index, fused Int8 IVF, ANN+filter co‑design).* Quantify: e.g. "1,800 lines of Triton + 600 lines of PyTorch glue."
2. *A unified evaluation protocol that brings these production systems onto public benchmarks (YFCC100M‑CLIP, MS‑Turing, DEEP, LAION‑400M) with selectivity‑binned filtered recall and recall‑QPS Pareto curves.*
3. *A reproducibility scorecard for LiNR and SilverTorch covering 18 quantitative claims, with verdicts and 95% CIs.*
4. *New insights: e.g. (i) ABM masking V1 dominates V2 above selectivity 0.6 on YFCC; (ii) Bloom co‑design's ~25% gain holds on public data but degrades to ~8% at selectivity < 0.05; (iii) Triton kernels reach 78% of cuBLAS+Faiss peak with 30% less GPU memory.* These are illustrative — replace with measured numbers.

**Novelty bar for NeurIPS D&B / MLSys**: a reproduction *alone* is generally not enough at NeurIPS main, but is enough at D&B and MLSys *if* the reproduction (a) targets a high‑impact system with no public artifact, (b) introduces a new benchmark/protocol or new datasets, (c) yields actionable findings beyond "the original numbers replicate." Your work has all three — but the writeup must explicitly enumerate them. MLSys 2025 onwards strongly weights the AE (artifact evaluation) badges; aim for "Reusable" not just "Available."

### 5. Practical artifacts

**Canonical metrics — definitions**:
- **Recall@k** = |R∩G|/k, R = retrieved top‑k, G = brute‑force float32 top‑k.
- **Filtered Recall@k** = |R∩G_P|/k, G_P = brute‑force top‑k *within the predicate's truth set P*.
- **MRR@k** = mean over queries of 1/rank of first relevant item, 0 if not in top k.
- **nDCG@k** = DCG@k / IDCG@k.
- **QPS** = total queries / wall‑clock (batch *or* single‑query — report both).
- **Latency p50/p95/p99** = quantiles of per‑query response time under specified load.
- **Build time**, **build memory** (peak RSS during index construction).
- **Index size** on GPU memory and on disk.
- **Selectivity** = |P|/n, fraction of corpus passing the predicate.
- **QPS@recall≥τ** = the leaderboard metric used by Big‑ANN '21/'23 (τ = 0.9 default).
- **Energy/query** (J), **$/1000 queries** at a fixed cloud SKU.

**Canonical datasets** (with current standard sizes):

| Dataset | Size | Dim | Type | Filtered variant available? |
|---|---|---|---|---|
| SIFT1M | 1M | 128 | uint8 | Synthetic labels via ACORN/NHQ |
| GIST1M | 1M | 960 | float32 | — |
| GloVe‑100 / GloVe‑200 | 1.2M | 100/200 | float32, cosine | — |
| MNIST / Fashion‑MNIST | 60K–70K | 784 | float32 | — |
| NYTimes | 290K | 256 | float32 | — |
| Last.fm | 290K | 65 | float32 | — |
| **BIGANN / SIFT1B** | 1B | 128 | uint8 | — |
| **DEEP1B** | 1B | 96 | float32 | — |
| **MS Turing‑ANNS** | 1B (10M slice) | 100 | float32 | Big‑ANN '21/'23 OOD slice |
| **MS MARCO** passage | 8.8M | 768 (varies) | float32 | Topic tags can be derived |
| **YFCC100M (Big‑ANN '23 filtered)** | 10M slice | 192 | uint8 (CLIP) | **Yes — native tag predicates (1‑2 of 200,386 tags, AND)** |
| **LAION‑400M / LAION‑2B** | 400M/2B | 512 (CLIP/OpenCLIP) | float32 | Yes — caption/text tags, NSFW flags |
| **Cohere wiki / Cohere multilingual** | varies | 768/1024 | float32 | Language tags |
| **Wikipedia‑22 / wikipedia‑en embeddings** | ~36M (en) | 768/1024 | float32 | Section/category tags |
| **SPACEV1B** (Microsoft) | 1B | 100 | int8 | — |
| **Text2Image1B** (Yandex) | 1B | 200 | float32 | Big‑ANN '23 OOD‑native |
| **Sparse MS MARCO (SPLADE)** | 8.8M | sparse | float32 | Big‑ANN '23 sparse track |

For your paper a recommended sweep is: **{SIFT1M+synthetic, GloVe‑100} for sanity → {DEEP10M, MS‑Turing‑10M} for mid‑scale → {YFCC100M‑10M CLIP filtered} for the headline filtered comparison → {LAION‑400M sample} as an LLM‑era extension.** Use Big‑ANN '23 ground‑truth files where available so your filtered recall numbers are directly comparable to ParlayANN, Puck, and the Faiss baseline.

**Canonical plots** (each must be in your paper):
1. Recall–QPS Pareto, log‑Y, per dataset, batch mode (the headline plot).
2. Recall–latency p50, single‑query mode.
3. Recall–latency p99, single‑query mode.
4. Build‑time vs. dataset size (1M → 100M).
5. Index memory vs. recall.
6. Filtered recall vs. selectivity (per ANN method) — three curves (pre‑filter, post‑filter, inline) per system.
7. Bloom false‑positive rate vs. bits/item (replicate SilverTorch's "6.98% at 512 bits → 0.067% at 1024 bits").
8. Per‑kernel microbenchmark bar chart (your Triton vs. cuBLAS+topk vs. Faiss‑GPU).
9. Roofline placement of each kernel on A100/H100.
10. Scaling: QPS vs. GPU count (1 → 2 → 4 → 8).
11. Live‑update throughput overhead curve (LiNR §5.3.3 replication).
12. Cost‑normalized plot: $/1M queries at recall ≥ 0.9 across all systems.

**Recommended outline at a glance**:

```
1.  Abstract
2.  Reproducibility Summary (MLRC template, 0.5 pp)
3.  Introduction
4.  Background
    4.1 LiNR
    4.2 SilverTorch
    4.3 The Filtered‑ANN Landscape
5.  Scope of Reproducibility & Claims
6.  Triton Implementation
    6.1 ABM (V1/V2/V3)
    6.2 Bloom Index
    6.3 Fused Int8 IVF
    6.4 ANN+Filter Co‑design
    6.5 Live Update Path
7.  Evaluation Protocol
    7.1 Hardware
    7.2 Datasets and Ground‑Truth Generation
    7.3 Predicate Definitions & Selectivity Buckets
    7.4 Hyperparameter Sweeps
    7.5 Statistical Methodology (seeds, CIs, tests)
8.  Reproduction Results
    8.1 Scorecard
    8.2 Unfiltered Pareto
    8.3 Filtered Pareto by Selectivity
    8.4 Kernel Microbenchmarks
    8.5 Multi‑GPU Scaling
    8.6 Live Update
9.  Extensions Beyond Reproduction
    9.1 LAION‑400M
    9.2 H100 vs. A100
    9.3 Negative Results / Failure Modes
10. Discussion: Threats to Validity & Recommendations
11. Conclusion
A.  Artifact Description (MLSys AE)
B.  Per‑claim Extended Tables
C.  Reproducibility Checklist (Pineau)
```

## Recommendations

**Stage 1 — Before writing (now):**
1. **Decide the venue first** — it shapes the paper. Recommended primary target: **NeurIPS 2026 Datasets & Benchmarks** (good fit: benchmark + open implementation; reviewers know Big‑ANN); secondary: **MLSys 2026** (better fit if the Triton kernels are the headline; artifact evaluation is a strong story); fallback: **VLDB 2026 Experimental & Analysis Track** or **SIGIR 2026 Reproducibility Track**. Avoid NeurIPS main track for a reproduction‑first paper unless you add a genuinely new algorithmic finding.
2. **Lock the reproducibility scorecard before running experiments.** Enumerate every quantitative claim from LiNR (C1–~C12) and SilverTorch (S1–~S10). For each, decide in‑scope / out‑of‑scope. Publish this table in a preregistration appendix.
3. **Adopt Big‑ANN '23's runner infrastructure** (`big-ann-benchmarks` on GitHub) as your evaluation harness rather than rolling your own. This buys you ground‑truth files, Docker isolation, and direct comparability to ParlayANN/Puck.
4. **Reach out to LiNR (Borisyuk et al.) and SilverTorch (Xue et al.) authors** for clarifications on missing hyperparameters. Document responses (or non‑responses) in the paper — both outcomes are publishable.

**Stage 2 — During experiments:**
5. **5 seeds minimum per cell**, 10 if time permits. Without this you cannot put CIs on the Pareto curves and reviewers will notice.
6. **Both A100 and H100** when possible — LiNR was on A100; SilverTorch on A100‑40G. Showing H100 scaling differentiates from the originals.
7. **Three filtering strategies on three selectivity buckets on three datasets**. That's 27 cells per system per metric; combined with seeds you have a saturating but defensible matrix.
8. **Numerical equivalence**: before reporting any speedup, certify max‑abs error of your Triton kernels vs. a fp32 reference is below 1e‑3 (Int8) or 1e‑5 (fp16) — FlashAttention paper convention.

**Stage 3 — Writing:**
9. **First draft the reproducibility scorecard table** — that's your contribution made concrete.
10. **Use the Aumüller et al. plot grammar** (recall–QPS Pareto with parameter‑sweep lines) for headline figures; reviewers familiar with the ANN literature recognize and trust this format.
11. **Adopt the ReScience C "What was easy / What was difficult / Communication with authors" sections** verbatim. These are signals of seriousness.
12. **Write a "Threats to Validity" section** modeled on empirical SE papers (Wohlin et al.) — almost no ML paper does this and it strongly differentiates yours.

**Stage 4 — Artifact:**
13. Release **Docker image + lock file + ground‑truth files + Croissant metadata**. The NeurIPS 2025 D&B Track CfP requires authors of dataset submissions to provide Croissant machine‑readable metadata to streamline review (where in 2024 it was only encouraged) — assume the same for 2026. Submit to ACM Artifact Evaluation for MLSys; target "Reusable" badge.
14. Mirror artifact on Zenodo with DOI; reference DOI in the paper.

**Thresholds that change the recommendation:**
- If the user finds that their Triton implementation **does not** reach within 1.5× of LiNR/SilverTorch's reported numbers on equivalent hardware, pivot the paper from "faithful reproduction" to "where reproduction fails and why" — still publishable at D&B/MLRC, but the framing must change.
- If the user finds **substantial novelty** (new kernel, new filtering scheme), split into two papers: a reproduction paper at D&B/MLRC and an algorithmic paper at NeurIPS/ICML — do not bundle.
- If only **one** of LiNR or SilverTorch can be faithfully reproduced (e.g., LiNR's ABM is reproducible but SilverTorch's Int8 fused kernel is not), restrict scope to that one system; a depth‑first reproduction beats a breadth‑first half‑reproduction.

## Caveats

1. The SilverTorch arXiv paper exists in multiple versions (v1 through v5+). Numbers like "10.18× ESR QPS" cited in some discussions appear in earlier versions and were revised in v5 (which contains "23.7×", "13.35×", "6.7×", "20.8×", "165.3×" as throughput/cost factors and "5.6%" as a *Recall@500 percentage gain*, not a speedup). Your paper must pin which version it reproduces against and quote the exact numbers from that version verbatim.
2. The Big‑ANN '23 filtered‑track headline numbers (e.g., ParlayANN at 37,671 QPS) are measured on **Azure D8lds_v5 CPU** with 8 vCPU / 16 GB RAM, not on GPUs. Direct comparison with your GPU Triton implementation is methodologically invalid; report **QPS/$** or **QPS/Watt** for cross‑hardware comparison.
3. LiNR's "billion‑sized index" claim is not equivalent to "1B vectors in a single GPU at fp16, dim=128" (that would need ~256 GB and is impossible on one A100); it refers to *combined* capacity across the deployment, with quantization and sharding. Distinguish carefully when restating.
4. Neither LiNR nor SilverTorch's authors have publicly committed to releasing reference code. Any "official" implementation you find on GitHub at the time of writing is unofficial. State this clearly in §2 (Background).
5. ANN/retrieval results are highly sensitive to **ground‑truth generation methodology** (float32 brute force vs. mixed precision vs. distilled). Use Big‑ANN '23 ground‑truth files where they exist; for new datasets, document GPU brute‑force float32 generation and release the GT files alongside the artifact.
6. The "30× scratch memory reduction" attributable to SilverTorch's co‑designed index is, in the primary v1 arXiv text, given as a *specific configuration result* (81M items / 9000 clusters / 256 probes → 2.3M items processed → 30× reduction); the paper's abstract uses qualitative language ("largely reduces memory utilization") and does not claim 30× as a universal figure. Some secondary literature reviews (e.g., themoonlight.io) extrapolate the 30× figure more broadly than the paper warrants — quote it only in the precise context it appears.
7. CAGRA, ScaNN‑GPU, and cuVS APIs are moving targets; pin commit SHAs in your artifact.
8. The "kernel microbenchmark" plots are easy to overclaim; always compare against the *latest* baseline (e.g., Faiss main branch, cuVS main branch) not the version‑at‑submission, and rerun before camera‑ready.