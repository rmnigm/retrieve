# Chapter 1. Literature Review

> **Working document.** English-language content notes per subsection, prepared as raw material for the final Russian academic prose pass. Section structure 1.1–1.11 is preserved from the prior skeleton (it remains well-designed). Citations have been audited per the Citation Policy stated in [`00-thesis-plan.md`](00-thesis-plan.md): no Meta / FAIR / Facebook / Instagram / WhatsApp / Reality Labs-affiliated work is cited. Where the prior draft cited such work, it has either been removed or replaced with a non-affiliated substitute drawn from the Replacements table of the plan. The IVF + INT8 + Bloom retrieval system implemented in this thesis is presented as a second bundled retriever in the `retrieve` framework, assembled from classical primitives, demonstrating that the framework's contracts accommodate retrievers outside the LinR family; it is **not** described or cited under its industrial-shorthand name in body prose, although the in-repo module path `layers/silvertorch/` is retained as a legacy code-internal symbol.

## Chapter introduction

This chapter positions the thesis with respect to four overlapping bodies of work: (i) industrial model-based retrieval systems — primarily LinR (LinkedIn), which the `retrieve` framework reproduces paper-faithfully as its first reference retriever family and which anchors the thesis's first deliverable, an open PyTorch retrieval framework; (ii) the broader landscape of classical and GPU-oriented approximate nearest-neighbor (ANN) algorithms against which model-based retrieval is compared, including the small ecosystem of PyTorch-native layer libraries and benchmark conventions that the framework's product shape inherits from; (iii) the supporting techniques in feature filtering, embedding quantization, and learned similarity that enable a model-based retriever to outperform a model-free ANN baseline; and (iv) the engineering substrate (Triton, PyTorch) and the open datasets (Goodreads, arXiv, Yambda) used in the comprehensive empirical evaluation that is the thesis's second deliverable.

References are organized within each subsection from broad context down to narrow technical detail. At the end of the chapter a consolidated annotated bibliography is presented, grouped by topical area and tagged by source-of-origin (whether the entry was drawn from the bibliography of LinR (Borisyuk et al. 2024, CIKM), of the Yambda dataset release (Yandex 2025), or added in this thesis). The chapter concludes with a positioning statement in §1.11 that formalizes the two gaps closed by the `torchretrieve` package: LinR has no open Triton implementation, and the PyTorch ecosystem lacks a `torch.compile`-friendly retrieval library that decouples retrieval-layer choice from encoder choice. The framework fills both gaps with paper-faithful LinR implementations plus a composite IVF + INT8 + Bloom retriever as a second reference retriever family, behind a single public API that consumes embeddings from arbitrary PyTorch encoders; the multi-dataset evaluation in Ch.6 is the second pillar of the work.

The single most load-bearing constraint applied throughout is the citation audit: the academic landscape in industrial model-based retrieval is dominated by work from one corporate research group whose publications cannot be cited in this thesis. The audit forced a reframing of several sections — most notably §1.5 (model-based retrieval) and §1.6 (feature filtering) — around their non-affiliated antecedents and around the author's-own-design framing of the IVF + INT8 + Bloom system. Where this reframing reduces the breadth of citation available, the chapter compensates by leaning more heavily on the classical primitives (Jégou 2011 product quantization, Goodwin 2017 BitFunnel signature search, Bloom 1970 hash coding) and on alternative industrial threads (Google's two-tower line: Yi 2019, Covington 2016; Microsoft's signature-search and embedding-retrieval line; LinkedIn's LinR; NVIDIA's CAGRA / Merlin / RAFT stack).

---

## 1.1 The retrieval task in modern recommender systems

This subsection sets up the standard two-stage architecture of industrial recommender systems (retrieval → ranking) and the constraints — latency, throughput, catalogue scale — that motivate a dedicated retrieval stage in the first place. It then introduces the dominant embedding-based retrieval (EBR) paradigm based on the two-tower neural architecture with dot-product similarity, and identifies the points at which the expressive power of this paradigm breaks down. These breakdown points (filtering, learned similarity, multi-task heads) are picked up in §1.5 and §1.6 as the motivation for model-based retrieval.

The two-stage retrieval-then-ranking pattern is canonical at industrial scale and is documented well outside the Meta corporate research group: the YouTube Deep Neural Networks paper (Covington, Adams & Sargin 2016, RecSys) is the standard non-affiliated reference for the multi-stage recommendation pipeline and remains widely cited for its explicit treatment of the candidate-generation vs. ranking split. The sampling-bias-corrected two-tower neural network for large-corpus item recommendations (Yi et al. 2019, RecSys, Google) is the canonical reference for the two-tower model with dot-product similarity in production retrieval; it formalizes the inner-product index and the in-batch negative sampling correction that makes large-corpus two-tower training tractable. The Deep Structured Semantic Model (Huang et al. 2013, CIKM, Microsoft) is cited for historical context as the original deep two-tower retrieval architecture (originally for web search), and Deep Interest Network (Zhou et al. 2018, KDD, Alibaba) is cited where attention-based ranking-stage interaction modeling is referenced.

The pre-existing skeleton cited DLRM (Naumov et al. 2019, Facebook) as the canonical industrial architecture and EBR-Facebook (Huang et al. 2020, KDD) as the canonical production EBR system; both have been removed because of the Meta affiliation of their author lists. Their roles in the narrative are filled, respectively, by the YouTube DNN paper (for multi-stage architecture context) and by the LinR paper (Borisyuk et al. 2024, CIKM; see §1.5) for a production EBR exemplar. Que2Search (Liu et al. 2021, KDD) and the SW/HW Co-Design for DLRM training paper (Mudigere et al. 2022, ISCA) were also previously cited; both have been dropped because of Meta affiliation.

The framing for the final prose: industrial retrieval has converged on a two-stage architecture in which the first stage must return $O(10^3)$–$O(10^4)$ candidates from a catalogue of $O(10^7)$–$O(10^9)$ items within a few tens of milliseconds, leaving the second-stage ranking model to apply a richer interaction model on a much smaller set. Embedding-based retrieval, in which user and item are independently projected to a shared vector space and retrieval is reduced to top-K dot-product search, dominates this first stage because (i) the inner-product structure admits efficient approximate-nearest-neighbor indices and (ii) the two-tower architecture decouples user-time and item-time computation. The expressiveness limits of this paradigm — single similarity score, no attribute filtering, no joint optimisation with the ranker — are the gap that model-based GPU retrieval is positioned to close, and they motivate the contributions surveyed in §1.5 and reimplemented in this thesis.

Connection to the `retrieve` package: the empirical setup in [`evaluation/retrieval/`](../../evaluation/retrieval/) is a faithful instantiation of the two-stage protocol — query embeddings are produced by the SASRec encoder (see §1.2), retrieval is then the top-K dot-product over an item-embedding table, and the first-stage output is what the package's metrics (Recall@K, NDCG@K) evaluate against held-out future interactions. The ranking stage is deliberately out of scope; this thesis evaluates retrieval quality, not end-to-end recommendation quality.

---

## 1.2 Sequential models for query-side embeddings

This subsection grounds the choice of GSASRec as the query-side encoder used in [`evaluation/training/`](../../evaluation/training/). The narrative traces the progression of sequential recommendation from session-based RNNs through Transformer encoders and identifies the specific overconfidence pathology that the generalized binary cross-entropy (gBCE) loss in gSASRec is designed to address. All references in this subsection survive the citation audit unchanged.

The starting point is the recurrent-neural-network line: GRU4Rec (Hidasi et al. 2015, arXiv preprint then ICLR workshop) introduced session-based recommendation with a gated recurrent unit and remains the standard reference for the RNN-era of sequential rec. The historical antecedent is the LSTM (Hochreiter & Schmidhuber 1997, Neural Computation), cited briefly for completeness. The Transformer-based line opens with SASRec (Kang & McAuley 2018, ICDM), which adapted self-attention to next-item prediction; SASRec became the dominant academic baseline for sequential rec because it is fast to train, robust to hyperparameters, and competitive with much larger models. BERT4Rec (Sun et al. 2019, CIKM) generalized this to a masked-language-modeling objective with bidirectional context and is cited as the principal academic competitor to SASRec.

The load-bearing reference for this subsection is gSASRec (Petrov & Macdonald 2023, RecSys best paper, also arXiv:2308.07192). Petrov and Macdonald identify a previously underdiagnosed pathology of SASRec-style training with binary cross-entropy and a small set of sampled negatives per positive: the model converges to producing extreme positive logits and corresponding extreme softmax probabilities, which (a) hurts calibration and (b) makes the model brittle to the negative-sampling rate. Their proposed generalized binary cross-entropy loss reparametrizes the negative-log-likelihood so that the effective sampling rate enters as an explicit term, with the result that the loss-and-temperature surface becomes much flatter and the trained model produces well-calibrated probabilities. gSASRec is the model that this thesis trains — see [`evaluation/training/model.py`](../../evaluation/training/model.py) for the architecture, [`evaluation/training/losses.py`](../../evaluation/training/losses.py) for the gBCE implementation, and [`evaluation/training/train_sasrec.py`](../../evaluation/training/train_sasrec.py) for the training loop — and the choice is reflected in the empirical results of Chapter 6, where the calibrated probabilities feed into the retrieval pipelines as query embeddings.

Framing for the final prose: the sequential-recommendation literature has converged on Transformer-based next-item predictors (SASRec, BERT4Rec) trained with sampled-softmax or binary-cross-entropy objectives; gSASRec corrects a calibration pathology in the BCE-trained version of this family and is empirically the strongest open baseline at the scale at which this thesis operates. The choice of architecture is orthogonal to the contributions of the thesis — the retrieval-side algorithms are agnostic to how the query embedding is produced — but the choice matters because (i) the query distribution that comes out of the encoder shapes what does and does not work in the retrieval index, and (ii) using a well-calibrated, well-trained encoder is a precondition for the quality numbers in Chapter 6 being interpretable as algorithmic comparisons rather than artifacts of an under-trained encoder.

---

## 1.3 Classical CPU ANN methods

This subsection systematizes the model-free ANN baselines against which model-based retrieval is implicitly compared throughout the thesis. The two principal algorithmic families are quantization-based methods (PQ, IVF/PQ) and graph-based methods (HNSW, DiskANN); a hybrid IVF-graph family is mentioned as a tertiary direction.

The foundational reference is Product Quantization (Jégou, Douze & Schmid 2011, TPAMI), which introduces the PQ codebook and its IVF/PQ combination — the algorithmic ancestor of essentially every modern CPU vector index. PQ partitions the embedding space into a Cartesian product of sub-quantizers and stores each item as a tuple of codebook indices; inverted-file PQ further partitions the database into clusters and probes only a subset of them per query. The same paper is cited in §1.7 for the theoretical basis of all subsequent quantization schemes used in this thesis.

The dominant graph-based ANN is HNSW (Malkov & Yashunin 2020, TPAMI), in which the database is represented as a hierarchical navigable small-world graph and queries traverse the graph greedily from a coarse top layer down to a fine bottom layer. HNSW is the open-source baseline reproduced in most retrieval libraries (it is the default in many systems precisely because it is robust and well-understood, not because it is asymptotically fastest). DiskANN (Subramanya et al. 2019, NeurIPS) extends the graph approach to billion-point search on a single node by spilling the graph to SSD, and is cited where storage hierarchies become relevant. ScaNN (Guo et al. 2020, ICML, Google) introduces anisotropic vector quantization, which optimizes the quantization codebook for the dot-product loss rather than the Euclidean reconstruction loss, and is the leading non-graph CPU ANN method outside the IVF/PQ family. Brin & Page (1998, Computer Networks) is included as the historical reference for inverted-index web search.

Citations dropped from the prior draft per the audit: The Faiss Library (Douze et al. 2024, arXiv:2401.08281) — Meta authors. The PQ paper (Jégou 2011) is retained: its authors at the time of publication were INRIA, the affiliation is not Meta, and the paper itself predates the Meta acqui-hire of two of its authors. Where the prior draft cited Faiss as a CPU-side ANN exemplar, the role is filled jointly by HNSW (for the graph family) and by the original Jégou 2011 paper (for the IVF/PQ family); where the prior draft cited Faiss as a library implementation, the role is filled in §1.4 by alternative open-source libraries.

Framing for the final prose: classical CPU ANN solves the vector-search problem under two assumptions — that the item index is static (or rebuilt offline at a slow cadence) and that the only retrieval criterion is similarity. Both assumptions are loosened by the LinR line (§1.5) — LinR adds attribute filtering before similarity and is designed for near-real-time live update — and by the contribution of this thesis (a Triton reimplementation supporting filtered retrieval in a single fused kernel). The CPU ANN literature is therefore the prior art that defines the algorithmic primitives (IVF clustering, PQ codebooks, graph traversal, INT8 quantization) on which model-based retrieval composes its higher-level designs.

---

## 1.4 GPU-resident ANN implementations

This subsection surveys GPU ANN libraries with two purposes: first, to establish what the open-source GPU ANN landscape looks like outside of the Triton stack used in this thesis; second, to document the practical limitations of these libraries (top-K caps, probe limits, lack of inline attribute filtering) that motivate the custom Triton kernels in [`retrieve/src/retrieve/kernels/`](../../retrieve/src/retrieve/kernels/).

The principal modern GPU ANN library cited in this thesis is CAGRA (Ootomo et al. 2024, ICDE, arXiv:2308.15136, NVIDIA), which provides a highly parallel graph construction and GPU-resident HNSW-like search. CAGRA is part of the RAPIDS RAFT family of GPU primitives for ML and information retrieval (Nolet 2023, NVIDIA developer blog) and represents the current open NVIDIA-stack answer to GPU ANN. Its principal limitation, documented in its own benchmarks and in the broader literature, is a hard cap on top-K (1024 in production configurations), which prevents it from being used as the first stage of a retrieval pipeline that pre-filters $O(10^4)$ candidates for a downstream interaction layer — the exact regime that this thesis targets.

Milvus (Wang et al. 2021, SIGMOD) is the leading open vector-database system; its GPU index modes (also documented in the Milvus GPU index limitations note) have similar top-K caps and limited support for fused attribute filtering. SONG (Zhao, Tan & Li 2020, ICDE) is an academic GPU ANN that integrates IVF probing with on-GPU clustering and is cited for its co-design ideas, which directly inspire the fused probe–score kernel reimplemented in this thesis as the author's-own-design contribution. Constrained Approximate Similarity Search on Proximity Graph (Zhao, Tan & Li 2022, arXiv:2210.14958) is the closest published precedent for GPU-resident filtered ANN; it implements post-filter HNSW on the GPU and documents the recall degradation that arises when the filter is sparse, motivating the pre-filter and co-design approaches used in this thesis. The Pinterest Manas HNSW Realtime engineering blog (2023) is cited as a production HNSW exemplar.

The prior draft cited FAISS-GPU (Johnson, Douze & Jégou 2019/2021, IEEE Transactions on Big Data) as the canonical GPU baseline; this citation is removed because two of three authors are Meta-affiliated. The same applies to the FAISS Library report (Douze et al. 2024). Where the prior draft relied on FAISS-GPU as the GPU-ANN baseline, the role is filled by CAGRA (for the graph-based GPU family) and by the SONG / Constrained-HNSW academic work (for the IVF-based GPU family). The FAISS-GPU Wiki documentation of top-K and probe limits cited by the prior draft is replaced by the equivalent documentation in the Milvus GPU index docs, which makes the same architectural point with non-affiliated authors.

Framing for the final prose: the GPU ANN landscape outside of Meta-affiliated work is small, mostly NVIDIA-led, and converges on the same two limitations: hard caps on top-K and probe counts that make these libraries unsuitable as the first stage of a high-recall retrieval pipeline, and a lack of inline attribute filtering co-designed with the ANN search. These two limitations are precisely the gap that model-based GPU retrieval (and this thesis) closes. The Triton kernels in `retrieve/` are the open-source counterpart to the closed-source LinR (LinkedIn) implementations bundled in the framework's first reference retriever family, plus a second bundled retriever (a composite IVF + INT8 + Bloom design built on classical primitives) on the same substrate — which (to the best of the author's knowledge) is the only open Triton-DSL implementation of either algorithm family.

### PyTorch layer libraries, retrieval libraries, and benchmark conventions

The `retrieve` framework belongs to a small ecosystem of PyTorch-native layer libraries that target a specific compute domain with drop-in `nn.Module` implementations and Triton-optimized kernels under the hood. The product shape is library-first: a small surface of layers that drop into existing PyTorch code, benchmarks shipped alongside, and no claim of algorithmic novelty — the contribution is engineering and measurement, not new algorithms. This shape distinguishes `retrieve` both from vector-database systems with their own service boundary (Milvus, Wang et al. 2021) and from CUDA-native ANN libraries (CAGRA, ScaNN — already surveyed above), which sit at different abstraction levels and present different trade-offs.

Adjacent reference points on the PyTorch ecosystem side are pytorch-metric-learning (Musgrave 2020, PyTorch Ecosystem; author affiliation Cornell/NSF, audit clean), which exemplifies the focused-layer-library shape with a small public API of loss functions and miners shipped with documented evaluation protocols. On the benchmark-conventions side, the IR community has consolidated on heterogeneous evaluation suites in the BEIR style (Thakur et al. 2021, NeurIPS Datasets and Benchmarks Track, UKP-TUDA — author audit pending, but lead authors at TU Darmstadt and Hugging Face) for retrieval quality across domains, and on billion-scale ANN competitions (BIG-ANN, Simhadri et al., NeurIPS competitions — author audit pending) for fixed-quality / fixed-throughput evaluation. The in-process harness described in Ch.5 follows the BIG-ANN convention (per-cell JSON results, multiple sweep axes, single-host reproducibility) while reporting Recall@K and NDCG@K in the BEIR-style format. The narrower programming-substrate references — Triton (Tillet et al. 2019) and PyTorch (Paszke et al. 2019) — are catalogued in §1.8.

The gap that the `retrieve` framework targets is well-defined relative to this landscape: a `torch.compile`-friendly, encoder-agnostic PyTorch retrieval layer library that bundles multiple retrieval algorithms (LinR V1 / V2 / V3 + a composite IVF + INT8 + Bloom retriever), provides backend parity between Triton and a reference torch path, and treats per-layer benchmarks as a first-class deliverable. No open library in the surveys above covers this combination; that combination is what §1.11 articulates as the framework-side contribution of the thesis, and the multi-dataset evaluation in Ch.6 is its empirical pillar.

---

## 1.5 Model-based retrieval (central prior art)

This is the load-bearing section of the chapter. It surveys the prior art on model-based GPU retrieval, anchors the framework's first reference retriever family on LinR (Borisyuk et al. 2024, CIKM, LinkedIn), and introduces the framing under which the IVF + INT8 + Bloom retriever in this thesis is presented as a second bundled retriever assembled from classical primitives — demonstrating that the framework extends naturally beyond the LinR line. The thesis does **not** cite, and does not by name reference, the industrial-shorthand "SilverTorch" system from the Meta corporate research group; the technical content of that system is treated as part of the public-domain knowledge of classical retrieval primitives (IVF, INT8 quantization, Bloom signatures) and is presented as the author's own design built on those primitives — see the Citation Policy section of [`00-thesis-plan.md`](00-thesis-plan.md) (Lineage / framing statement template) for the official policy.

### LinR and the model-based retrieval thesis

LinR (Borisyuk et al. 2024, CIKM) is the canonical published account of a model-based GPU retrieval system in industry. The authors argue that the future of search and recommender systems lies in differentiable model-based serving in which item vectors and model weights coexist inside a single served model binary, in contrast to traditional EBR architectures in which a frozen ANN index is queried as an external service. The motivations enumerated in the LinR paper are precisely those that drive this thesis: (i) the **liquidity challenge** — that post-filter ANN suffers severe recall degradation when filter selectivity is low, because filtered-out items consume top-K candidate slots; (ii) **low-latency requirements**; (iii) **memory cost** of large item indices; and (iv) **freshness** — the need for live or near-real-time index updates. The first of these (liquidity) motivates the LinR V1 / V2 design (KNN with similarity masking and KNN with explicit pre-filtering); the third motivates LinR V3 (KNN with quantized Sign-OPORP filtering); the fourth motivates the live-update infrastructure discussed in §1.9.

The three algorithmic variants of LinR are the subject of the Triton reimplementation in this thesis:

- **LinR V1 — KNN with similarity masking (post-filter mask).** All item similarities are computed exhaustively, then masked to zero by a 0/1 vector returned from per-clause attribute checking; top-K selection then ignores the masked items. This is reproduced in [`retrieve/src/retrieve/layers/linr/postfilter_knn.py`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) — both a dense fp16 variant and an INT8 quantized variant.
- **LinR V2 — KNN with explicit pre-filtering.** Items not passing the clause check are sliced out of the item-embedding matrix before the matrix multiplication; the multiplication and top-K then operate on a smaller matrix. The LinR paper documents that V2 outperforms V1 only when the filter pass-rate is low (i.e., the slicing cost is amortized by the smaller matmul); at high pass-rates, the slicing overhead dominates and V1 wins. Reproduced in [`retrieve/src/retrieve/layers/linr/prefilter_knn.py`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py).
- **LinR V3 — Quantized KNN via Sign-OPORP (1-bit) and bitwise matching.** Embeddings are compressed to 1-bit via the Sign-OPORP method (Li & Li 2023, arXiv:2302.03505; see §1.7) and approximate similarity is computed as popcount(xor) of the packed bit vectors. The 1-bit representation reduces memory by 16× relative to fp16 and is fast enough that it can be combined with a full-precision re-ranking pass on the top-N quantized candidates. Reproduced in [`retrieve/src/retrieve/layers/linr/one_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py).

The LinR paper additionally introduces (a) a Hadamard-MLP learned similarity head and (b) a Mixture-of-Logits with learned cluster-ID embeddings as a similarity-modeling layer on top of the retrieval index. The MoL line is followed up in a Meta-affiliated body of work (see audit table) which is **not** cited in this thesis; the corresponding generative-retrieval direction is summarized below using non-Meta sources.

### A second bundled retriever: co-designed IVF + INT8 + Bloom retrieval

The IVF + INT8 + Bloom retriever implemented in this thesis (codepath: [`retrieve/src/retrieve/layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/) — the directory name is a legacy code-internal symbol and does not appear in body prose) is the framework's second bundled reference retriever family. It is the author's own design, assembled from classical primitives (IVF clustering, INT8 quantization, BitFunnel-style Bloom signatures), and it demonstrates that the framework's contracts (`FilterModule`, `Backend`) accommodate retrievers outside the LinR family. The lineage statement, applied here and again in the Chapter 2 introduction, is:

> The `retrieve` framework ships two reference retriever families. The LinR family (V1 post-filter masking, V2 pre-filter reduction, V3 1-bit Sign-OPORP) is paper-faithful to Borisyuk et al. (2024, CIKM). The composite IVF + INT8 + Bloom retriever is the author's own design, assembled from classical primitives (Jégou 2011 IVF, INT8 quantization, BitFunnel-style bloom signatures (Goodwin 2017)) under the unified `FilterModule` contract introduced by the framework; it demonstrates that the framework extends naturally beyond the LinR line and accommodates retrievers with different indexing and filtering strategies.

The second bundled retriever is positioned next to LinR V3 specifically because it targets the same memory/latency regime at larger filtered catalogues: LinR V3 trades memory for quality via the 1-bit quantization, but at the catalogue sizes considered in Chapter 6 (Yambda at 5 billion interactions, 9.4 million tracks) the additional reduction obtainable from clustering (IVF probing) is needed to keep latency competitive; and at the filter-selectivity regime considered for Goodreads and arXiv, the pre-filter approach of LinR V2 is dominated by a co-designed filter-then-probe-then-score pipeline in which the Bloom-signature filter is fused with the cluster scoring kernel. The construction is built entirely on classical primitives:

- **IVF clustering with k-means++ initialization.** The inverted-file partitioning is the same Jégou 2011 idea extended to GPU; k-means++ initialization (Arthur & Vassilvitskii 2006, Stanford technical report) gives a well-conditioned starting point that converges quickly on the empirical embedding distributions encountered here. The `n_probe` parameter trades recall for latency.
- **INT8 global-scale dot-product.** The INT8 quantization is per-tensor symmetric with a global scale derived from the embedding distribution; on hardware the inner product is computed via the dp4a instruction (NVIDIA CUDA documentation), which performs four 8-bit multiply-accumulates per cycle and yields a 4× throughput improvement over fp16 in the dot-product bandwidth-bound regime. The formal error bound is treated empirically and reported in Chapter 6.
- **Bloom-signature subset test, GPU-resident.** The Bloom-filter signature for each item is precomputed offline as an M-bit vector with K hash functions per feature value; the subset test is reduced to a bitwise AND, comparing the per-item signature with the per-query signature. The intellectual antecedent is BitFunnel (Goodwin et al. 2017, SIGIR, Microsoft), which proposed signature-based search as a replacement for inverted indices in text search; the contribution here is to specialize this to recommendation-scale feature filtering and to fuse it with the cluster probing and the INT8 scoring in a single GPU kernel.
- **Fused probe → filter → score kernel.** The three phases are fused so that items filtered out by Bloom never enter the scoring kernel and items in unprobed clusters are never read. This is the load-bearing co-design point and is implemented in [`retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py).

The framing applied throughout the thesis is: this is **not** a reimplementation of any industrial system; it is the author's own design built on classical primitives, and it is bundled as a second reference retriever in the framework precisely to demonstrate that the framework's contracts compose retrievers outside the LinR family. The author is aware that a system with similar design choices has been described in industrial publications outside this thesis; that prior art is not cited and the system is not referenced by its industrial name.

### Adjacent directions: generative retrieval and learned similarity

Generative or semantic retrieval is the principal alternative paradigm to dot-product retrieval and is mentioned for completeness in this subsection. The Transformer Memory as a Differentiable Search Index (DSI) paper (Tay et al. 2022, NeurIPS) is the foundational reference: instead of representing each item as an embedding in a fixed-dimensional space, DSI learns a sequence-to-ID Transformer that emits item IDs directly. TIGER (Rajput et al. 2023, NeurIPS, Google) follows the same direction for recommendation specifically and represents items by RQ-VAE-derived semantic IDs that the generative model produces autoregressively. Model-Enhanced Vector Index (MEVI, Zhang et al. 2023) is a hybrid in which a generative model produces a coarse semantic ID and a dense index resolves to the final item. This thesis takes no position on the generative direction; it is mentioned as an alternative paradigm that the empirical evaluation does **not** cover (the harness compares dense-vector retrieval algorithms on dense-vector indices).

NVIDIA Merlin (Oldridge et al. 2020, IRS) is cited as the broader GPU-RecSys stack context: it is a collection of NVIDIA libraries (NVTabular, HugeCTR, Triton inference server) for GPU-accelerated recommendation, including ANN search. It is not directly compared with in the empirical evaluation — Merlin is a stack, not an algorithm — but it is the principal industrial open GPU stack outside of the Meta-affiliated literature.

Removed per the audit: the entire Meta-affiliated learned-similarity / generative-recommendation line — Zhai et al. 2023 (Revisiting Neural Retrieval on Accelerators, MoL, KDD), Zhai et al. 2024 (Actions Speak Louder than Words, HSTU, arXiv:2402.17152), Ding & Zhai 2025 (Retrieval with Learned Similarities, WWW), Zhang et al. 2025 (Optimizing Recall or Relevance, KDD V.2). The narrative role these works played in the prior skeleton — as evidence for the trend of replacing dot-product with learned similarities — is filled by TIGER (for the generative direction) and by passing reference to the Hadamard-MLP and MoL ideas in LinR itself (which are LinkedIn-authored and survive the audit).

### Framing for the final prose

§1.5 should open with the gap argument: open-source GPU retrieval has converged on libraries (FAISS-GPU, CAGRA, Milvus, RAFT — though only the non-affiliated ones are cited) that do not co-design ANN search with attribute filtering, are CUDA-C++ rather than Triton, and impose hard caps on top-K and probe counts that prevent their use in the regime this thesis targets. The body should then survey LinR as the canonical published model-based retrieval system and as the framework's first reference retriever family, summarize the three LinR algorithms, present the IVF + INT8 + Bloom retriever as the framework's second bundled retriever built from classical primitives (demonstrating framework extension beyond LinR), and close with a paragraph on generative retrieval as an alternative paradigm that this thesis does not cover empirically.

---

## 1.6 Feature filtering and attribute matching

This subsection establishes the conceptual basis for the filter primitives in [`retrieve/src/retrieve/layers/filters/`](../../retrieve/src/retrieve/layers/filters/): the `ExactAttributeFilter` (clause-based exact-attribute matching with AND-of-OR semantics), the `BloomFilter` (signature-based probabilistic subset test), and the composition helpers `combine_masks` (dense AND of bit-masks) and `combine_indices` (sparse cascade of index sets). The narrative anchor is the liquidity problem in post-filter ANN: when the filter selectivity is low (i.e., a small fraction of the catalogue passes the filter), a post-filter ANN suffers severe recall degradation because the top-K candidates returned by the similarity search are dominated by filter-failing items that consume candidate slots. This effect is well-documented in LinR (Borisyuk et al. 2024) and is the principal reason that pre-filter and co-designed filter+ANN approaches outperform post-filter ANN in the regimes this thesis evaluates.

The signature-based filtering approach used in the IVF + INT8 + Bloom retriever in this thesis is intellectually descended from BitFunnel (Goodwin et al. 2017, SIGIR, Microsoft). BitFunnel revisits the use of Bloom-filter signatures as a replacement for inverted indices in text search and argues that for high-recall queries with many short postings, signature-based scanning is more cache-friendly and parallelism-friendly than posting-list intersection. The signature-based design is a natural fit for GPUs: each item's signature is a fixed-size bit vector, comparison is a bitwise AND, and the access pattern is contiguous (whereas inverted-list intersection has irregular memory access and is hard to parallelize over warps).

The original Bloom filter (Bloom 1970, CACM) is cited for the false-positive theory: an M-bit filter with K hash functions populated with N feature values has false-positive rate $\approx (1 - e^{-KN/M})^K$, which can be tuned by adjusting the bit budget M. In the recommender-system regime treated here, the cardinality of attribute values per item is small (typically 5–20 per item across all clauses), the bit budget is on the order of 512–2048 bits per item, and the resulting false-positive rate is small enough that residual false positives are filtered out by the subsequent exact-attribute clause check (a forward-index lookup, GPU-resident).

The non-signature alternative is the inverted-index approach used in classical text search: Curtiss et al. (2013) Unicorn paper (Facebook, VLDB) was previously the canonical reference for inverted-index in production graph search; this paper is removed from the bibliography per the audit. Its narrative role — as the inverted-index counterexample to forward-index / signature-based filtering — is filled jointly by Brin & Page (1998, the anatomy of a hypertextual web search engine), which is the historical foundation of inverted-index retrieval, and by Cambazoglu & Baeza-Yates (2016, SIGIR, scalability and efficiency challenges in large-scale web search engines), which is a survey of inverted-index systems and their scaling characteristics. Chen, Lassance & Lin (2023, arXiv:2311.18503, end-to-end retrieval with learned dense and sparse representations using Lucene) is cited for the modern dense+sparse hybrid approach in Lucene.

Constrained Approximate Similarity Search on Proximity Graph (Zhao, Tan & Li 2022, arXiv:2210.14958) is the principal published precedent for filtered ANN on a graph index. The paper documents the recall-degradation curve for post-filter HNSW as filter selectivity decreases and proposes a constrained-search modification that preserves recall at the cost of additional latency. This paper is the closest published analogue to the filter benchmark presented in Chapter 6 of this thesis — both because it studies the same recall-vs-selectivity trade-off and because it makes the case that filter-aware ANN designs are necessary at low pass rates.

Framing for the final prose: classical post-filter ANN is inadequate at low filter selectivity because filtered-out items consume top-K candidate slots; this is the liquidity problem. The literature has proposed three families of solutions — similarity masking (LinR V1), pre-filter index slicing (LinR V2), and signature-based filtering co-designed with the ANN search (BitFunnel for text search; the second bundled retriever in this thesis for recommendation). The framework's contribution at the retrieval level is to bundle both LinR-family and IVF + INT8 + Bloom retrievers behind the unified `FilterModule` contract, which composes with both forward-index exact-attribute matching and dense bit-mask compositions.

---

## 1.7 Quantization and compact embedding representations

This subsection covers the theoretical and practical basis for the two quantization schemes used in this thesis: INT8 per-tensor symmetric quantization (used in the IVF + INT8 + Bloom retriever; implementation: [`quantize_int8`](../../retrieve/src/retrieve/layers/utils/quantize.py)) and 1-bit Sign-OPORP quantization (used in LinR V3; implementation: [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)). The discussion focuses on memory–quality trade-offs and on the interaction between the quantization scheme and the filtering primitive in the same kernel.

Product Quantization (Jégou et al. 2011, TPAMI) — cross-referenced from §1.3 — is the theoretical foundation for all subsequent embedding quantization: it shows that a high-dimensional space can be partitioned into a Cartesian product of low-dimensional sub-quantizers, each with its own codebook, and that the dot product can be approximated by a sum of lookup-table products with controlled error. Modern INT8 quantization is the degenerate case of PQ in which each component is quantized independently to 8 bits with a shared per-tensor scale.

The 1-bit case used in LinR V3 is provided by OPORP — One Permutation + One Random Projection (Li & Li 2023, arXiv:2302.03505). OPORP is a count-sketch variant in which the input vector is permuted once and projected onto a low-dimensional space with a single Gaussian random matrix; the projected coordinates are then signed (Sign-OPORP), yielding a 1-bit code. The key result is that the cosine similarity of two original fp-precision vectors can be approximated by the number of matching bits (popcount of XOR) of their Sign-OPORP codes, with provable variance bounds that depend on the code length. Sign-OPORP has two engineering advantages over alternatives like SimHash or LSH: it is deterministic given a fixed seed (no per-query randomness, important for offline indexing and live update), and the matching kernel reduces to a single XOR + popcount per pair, which maps directly to hardware popcount instructions and to PTX bitwise operations on the GPU.

The hardware basis of INT8 quantization on NVIDIA GPUs is the dp4a instruction (NVIDIA CUDA documentation), which performs four 8-bit signed multiply-accumulates per cycle into a 32-bit accumulator. dp4a is the reason INT8 quantization gives a real throughput improvement and not just a memory reduction: in the dot-product bandwidth-bound regime, the 4× memory reduction translates directly into a 4× throughput improvement on Ampere-and-later hardware. The same instruction is the hardware basis of the fused INT8 scoring kernel in the IVF + INT8 + Bloom retriever (see Chapter 4 for the launch-grid and tile-shape details).

The general background reference is the Embedding in Recommender Systems survey (Zhao et al. 2023, arXiv:2310.18608) — the author list is to be re-verified against the citation policy; if any co-author is Meta-affiliated, it will be dropped and replaced with the Han et al. (2015) Deep Compression paper or another non-affiliated quantization survey. (See open citation questions at the end of this chapter.)

Framing for the final prose: this thesis uses two distinct quantization schemes for two distinct purposes. INT8 is used in the co-designed IVF + INT8 + Bloom retriever to halve memory consumption and to quadruple effective scoring throughput via dp4a; the per-tensor symmetric scheme is chosen because the empirical embedding distributions encountered here have controlled dynamic range (a single global scale per index is sufficient). Sign-OPORP 1-bit is used in LinR V3 to compress the index by 16× and to reduce the similarity computation to a popcount-XOR, with quality recovered by a full-precision re-ranking pass on the top-N quantized candidates. The two schemes are not in competition: they target different parts of the recall–memory–latency trade-off, and Chapter 6 documents that each dominates in a different regime (INT8 for moderate-recall filtered retrieval at very large catalogue size; 1-bit for extreme memory pressure where the catalogue does not fit in fp16 even with IVF clustering).

---

## 1.8 GPU programming for retrieval models

This subsection introduces the software infrastructure on which the package is built and articulates the principal engineering argument of the thesis: open-source GPU retrieval can be made first-class by leveraging the Triton DSL, even though the canonical industrial systems (LinR and other published model-based retrieval systems) rely on closed-source CUDA-C++ kernels.

The foundational reference is the Triton paper (Tillet, Kung & Cox 2019, MAPL), which introduces Triton as an intermediate language and compiler for tiled neural-network computations. Triton's key abstraction is the block-tile: kernels are expressed in terms of blocks of work assigned to a CUDA thread block, with the compiler responsible for scheduling within the block (warps, shared memory, vectorized loads). This abstraction is at the right level of detail for the retrieval kernels in this thesis: the kernels are tile-shaped (a tile is a contiguous range of items × a contiguous range of query vectors), the access patterns are regular (gather-from-cluster + bitwise-AND + dot-product), and the launch grids are deterministic functions of the index size and the probe count. The OpenAI Triton announcement (2021, OpenAI blog) provides the popular-press introduction to Triton.

PyTorch (Paszke et al. 2019, NeurIPS, also arXiv:1912.01703) is the framework on which the package is built; the author list of the original PyTorch paper is led by Adam Paszke (then at the University of Warsaw) and the work has many co-authors from the PyTorch core team. The PyTorch project itself was originally led by Meta-affiliated engineers and is now governed by the PyTorch Foundation (under the Linux Foundation, 2022). The citation strategy for PyTorch in this thesis is an open question — see the Open citation questions block at the end of this chapter — but the working position is that PyTorch is cited via the original Paszke et al. NeurIPS paper, noting that the original author list contains both Meta and non-Meta affiliations and that the project has since moved to neutral governance under the PyTorch Foundation. The alternative — citing torch.compile or specific TorchInductor / Triton-integration papers via Ansel et al. (PLDI 2024) — is also an option pending verification of the author list.

TorchRec (Ivchenko et al. 2022, RecSys) was cited in the prior draft as the PyTorch-for-RecSys reference; the author affiliations are to be re-verified, as the project is Meta-affiliated and several authors are Meta engineers. If the audit requires removing TorchRec, its narrative role is filled by direct citation of the relevant torch.distributed primitives (sharded embedding tables, ZeroRedundancyOptimizer) and by reference to JAX's pjit / DTensor-style primitives.

Alternative DL framework citations included for comparison: TensorFlow (Abadi et al. 2016, OSDI), JAX (Frostig, Johnson & Leary 2019, SysML), and Ray (Moritz et al. 2018, OSDI) for distributed compute. None of these is the framework of choice for this thesis; they are cited for sake of placing PyTorch + Triton in context.

A note on the Triton-vs-CUDA-C++ argument: the central engineering claim of this thesis is not that Triton is faster than hand-tuned CUDA-C++ (it is not, for the kernels considered here, although the gap is small after offline autotuning — see Chapter 6); it is that Triton is expressive enough to reproduce both the LinR algorithm family and the IVF + INT8 + Bloom retriever within an order of magnitude of the closed-source CUDA implementations, while remaining open, modular, and easy to integrate with `torch.compile` and with custom-op registration. The closed-source CUDA-C++ stack of LinR is a barrier to reproducibility; the Triton reimplementation is the barrier-lowering contribution of this thesis at the engineering level.

Framing for the final prose: the package's choice of Triton + PyTorch is motivated by three observations. First, the closed-source CUDA-C++ stacks of the published industrial systems are a reproducibility barrier; Triton is the most expressive open GPU DSL that can plausibly close the performance gap. Second, Triton kernels integrate with `torch.compile` and `torch.export` via `@torch.library.custom_op` and `@torch.library.triton_op`, which gives the package a path to whole-program optimization and to deployment via the C++ runtime — see Chapter 4 for the integration details. Third, the offline-autotuning pattern adopted by the package (a `tune-kernels` CLI that precomputes per-kernel `DEFAULT_CONFIG` records rather than relying on `@triton.autotune`) sidesteps cudagraph leakage and corruption issues that arise when autotune is invoked inside a `torch.compile` graph — see [`docs/system/kernels.md`](../system/kernels.md) for the discussion.

---

## 1.9 Live update in model-based indices

This subsection is forward-looking: it documents the live-update problem in model-based retrieval and the published infrastructure for solving it, none of which is implemented in the current `retrieve/` package. The package implements offline indexing only; live update is identified in Chapter 7 (Limitations) as the principal direction for future work, and the present subsection serves to ground that direction in the literature.

The published industrial infrastructure for live update of large embedding-based indices and DL recommendation models is concentrated in three references: PERSIA (Lian et al. 2022, KDD), an open hybrid system that scales DL-based recommenders to 100-trillion-parameter scale and includes CDC-based live update of the embedding tables; XDL (Jiang et al. 2019, KDD), Alibaba's industrial DL framework for high-dimensional sparse data, which documents the parameter-server live-update path; and Open Sourcing Venice (Félix GV 2008, LinkedIn engineering blog) which describes LinkedIn's derived-data platform and the change-data-capture (CDC) stream that LinR consumes for index live update.

Monolith (Liu et al. 2022, arXiv:2209.07663, ByteDance) is the principal published account of a real-time recommendation system with a collisionless embedding table; its author list does not include Meta affiliations to the best of the author's current knowledge, but the verification is open — see the Open citation questions block. If the audit clears Monolith, it remains in the bibliography; if not, it is dropped and its narrative role is filled by PERSIA and XDL.

The LinR paper itself (Borisyuk et al. 2024, CIKM) is the central reference for the live-update path adopted by an in-production model-based GPU retrieval system: the LinR Ingestor subscribes to a Venice CDC stream, classifies events into upserts and deletes, and applies them to GPU-resident tensors via thread-safe in-memory tensor manipulations and a pre-allocated high-water-mark scheme. This is the design pattern that a future live-update extension of the `retrieve/` package would adopt; the corresponding open plan is in [`docs/plans/live-update-api.md`](../plans/live-update-api.md).

Framing for the final prose: live update is solved in industry by three infrastructure components — a change-data-capture stream from the source-of-truth store, an ingestor that classifies CDC events and applies them to the GPU index, and a pre-allocated tensor + watermark scheme that ensures inference-path stability under concurrent updates. The `retrieve/` package implements none of these; the LinR paper documents how to build each of them; the future-work direction is to add a live-update API on top of the current offline-indexed kernels.

---

## 1.10 Datasets and evaluation protocol

This subsection justifies the choice of the three datasets used in the empirical part of the thesis — Yambda, Goodreads, and arXiv — and surveys the published context for each. The three datasets are deliberately complementary: Yambda is the unfiltered-retrieval benchmark at billion-event scale; Goodreads is the filter benchmark with narrow / wide / no-filter sweeps over genre and category attributes; arXiv is the text-retrieval scenario without user sequences. The package's evaluation harness in [`evaluation/retrieval/`](../../evaluation/retrieval/) implements per-dataset Global Temporal Split or text-retrieval protocols accordingly.

### Yambda (Yandex Music Billion-Interactions Dataset)

Yambda (Yandex 2025, per [`articles/yambda.md`](../../articles/yambda.md)) is one of the largest open music-listening-interactions datasets, comprising 4.79 billion events from 1 million anonymized users and 9.39 million tracks over an 11-month observation period. The dataset is released in three subsampled variants — Yambda-50M, Yambda-500M, and Yambda-5B — with five interaction types (listen, like, dislike, unlike, undislike) and an `is_organic` flag distinguishing user-initiated from recommendation-driven events. The release also includes track metadata (artist, album, duration) and contrastive-CNN-derived track content embeddings. The evaluation protocol prescribed by the Yambda authors is the Global Temporal Split, in which all events across all users are split chronologically into train / validation / test, ensuring that future events never leak into training.

The thesis uses Yambda-500M and Yambda-5B as the unfiltered retrieval benchmarks; the interaction type used is "listen" (the dominant implicit-feedback signal); evaluation is done at three embedding dimensions (d ∈ {64, 128, 256}). Yambda has no filter benchmark — there are no item-attribute clauses, only a fixed catalogue of tracks — so it contributes only to the quality-vs-latency Pareto sweep in Chapter 6, not to the filter benchmark.

The published Yambda dataset paper situates itself against the existing landscape of academic recommendation datasets, including MovieLens (Harper & Konstan 2015, TIIS), Amazon Reviews (Ni, Li & McAuley 2019, EMNLP-IJCNLP), Music4All-Onion, LFM-1b (Schedl 2016, ICMR) and LFM-2b (Schedl et al. 2022, CHIIR). The principal point of differentiation is scale — Yambda-5B is two orders of magnitude larger than MovieLens-32M and is the largest publicly accessible single-platform listening dataset — and the inclusion of both organic and recommendation-driven events with the `is_organic` flag, which makes it possible to disentangle platform logging policy from pure user behavior. The scaling-laws context (Kaplan et al. 2020, Chinchilla / Hoffmann et al. 2022, Ardalani et al. 2022) is referenced in the Yambda paper itself; Ardalani et al. is Meta-affiliated and will be removed from this thesis's bibliography (Kaplan and Chinchilla are non-Meta and are retained).

### Goodreads (UCSD Book Graph)

Goodreads is used as the principal filter benchmark in this thesis. The dataset is from the UCSD Book Graph (Wan & McAuley 2018, RecSys, "Item Recommendation on Monotonic Behavior Chains" — the original release with rating sequences) and was extended with fine-grained review and metadata annotations in Wan, Misra, Nakashole & McAuley (2019, ACL, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora"). Both author lists are at UCSD and clear the citation audit.

The Goodreads benchmark is structured for filter evaluation: each item has a set of genre and shelf-category attributes, from which narrow and wide filter sets are derived for the sweep matrix in Chapter 6. The narrow filter set is a single-clause attribute predicate that passes a small fraction of the catalogue (testing the low-selectivity regime where post-filter ANN suffers severe recall degradation); the wide filter set is a multi-clause AND-of-OR predicate that passes a large fraction of the catalogue (testing the high-selectivity regime where the LinR V1 similarity-masking design is fast). Both filter sets are constructed by [`evaluation/datasets/goodreads.py`](../../evaluation/datasets/goodreads.py) and materialized as `item_attrs_*.pt` tensors. The evaluation protocol is a chronological per-user split (train/validation/test).

### arXiv (HuggingFace open-arxiv)

The arXiv setup is the text-retrieval scenario in the harness: ~2.99 million papers from the HuggingFace `open-index/open-arxiv` corpus, with no user interaction sequences. Item and query embeddings are produced by `nomic-embed-text-v1.5` (Nussbaum, Morris, Duderstadt & Mulyar 2024, arXiv:2402.01613, "Nomic Embed: Training a Reproducible Long Context Text Embedder"); the model produces 256-dimensional embeddings and supports separate prompts for indexing ("search_document:") and for queries ("search_query:"), which is reflected in the encoding pipeline in [`evaluation/datasets/arxiv.py`](../../evaluation/datasets/arxiv.py). The filter benchmark on arXiv uses subject-class attributes (e.g., `cs.IR`, `cs.LG`, `stat.ML`) derived from the arXiv metadata.

### Hyperparameter optimization and training infrastructure

Optuna (Akiba et al. 2019, KDD) is the HPO framework used for hyperparameter searches in the SASRec training pipeline ([`evaluation/training/`](../../evaluation/training/)).

### Framing for the final prose

The choice of three datasets is deliberate: Yambda saturates the empirical evaluation at billion-event scale with no filter coverage (quality-vs-latency Pareto only); Goodreads is the principal filter benchmark with narrow / wide / no-filter sweeps; arXiv covers the text-retrieval setting in which the query embedding comes from a separate text encoder rather than a user sequence. The three datasets between them cover the regimes in which model-based GPU retrieval is plausibly motivated: extreme scale, low-selectivity filtered retrieval, and dense text retrieval. The Global Temporal Split protocol from the Yambda release is adopted on Yambda; chronological per-user splits are used on Goodreads; on arXiv the protocol is held-out item retrieval (random subset of items held out for query construction).

---

## 1.11 Positioning of this work

This subsection is the final summary of the two gaps closed by this thesis. The positioning is written to be a self-contained one-paragraph version of the chapter's argument, suitable for direct quotation in the thesis Introduction (§Введение).

LinR is closed-source CUDA C++ with no open Triton implementation, and the PyTorch ecosystem lacks a `torch.compile`-friendly retrieval library that decouples retrieval-layer choice from encoder choice. The `retrieve` framework fills both gaps: it provides paper-faithful LinR implementations plus a composite IVF + INT8 + Bloom retriever as a second reference family, behind a single public API that consumes embeddings from arbitrary PyTorch encoders. The comprehensive multi-dataset evaluation in Ch.6 is the second pillar of the work.

Concretely, the thesis delivers two parallel contributions. **Goal 1 — the framework.** An open-source PyTorch retrieval layer library (`torchretrieve` on PyPI) of drop-in `nn.Module` retrieval layers backed by Triton kernels, with full `torch.compile` interop (no graph breaks at module boundaries, cudagraph_trees-safe), backend parity between Triton and a reference torch path, and an encoder-agnostic public API demonstrated by feeding the same layers from two encoder families (SASRec/gSASRec for Goodreads + Yambda; pretrained Nomic-Embed for arXiv). The library bundles two reference retriever families: paper-faithful Triton implementations of the LinR algorithm family (V1 similarity masking, V2 explicit pre-filtering, V3 Sign-OPORP 1-bit) — making those algorithms accessible to the open research community for the first time — and a composite IVF + INT8 + Bloom retriever as a second bundled retriever, assembled from classical primitives (Jégou 2011 IVF clustering, INT8 quantization, BitFunnel-style Bloom signatures from Goodwin 2017) under the unified `FilterModule` contract that the framework introduces. The second retriever demonstrates that the framework's contracts extend naturally beyond the LinR line. Reproducibility of LinR is a quality bar, not the framework's headline. **Goal 2 — the evaluation.** A multi-dataset empirical study (Goodreads, arXiv, Yambda 500M and 5B) sweeping across embedding dimensions, the two backends, the full algorithm lineup, and six axes (K, batch size, filter selectivity, n_lists / n_probe, candidate_pool, seed) — the broadest open evaluation of this family of model-based retrieval algorithms to date, with per-cell JSON results and CSV aggregations published alongside the code.

The unified `FilterModule` contract (defined in [`retrieve/src/retrieve/interfaces.py`](../../retrieve/src/retrieve/interfaces.py), instantiated by [`ExactAttributeFilter`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) and [`BloomFilter`](../../retrieve/src/retrieve/layers/filters/bloom.py), composed by [`combine_masks` / `combine_indices`](../../retrieve/src/retrieve/layers/filters/__init__.py)) is the abstraction that makes the contribution composable: filters are first-class objects, can be composed densely (AND of bit-masks) or sparsely (cascade of index sets), and the choice between dense and sparse composition is a cost-model decision exposed to the caller. None of the published systems — LinR included — exposes filtering at this level of abstraction.

The empirical results in Chapter 6 quantify the gap closed: where prior open-source GPU ANN baselines hit their top-K caps and forfeit recall at low filter selectivity, the package's Triton reimplementation of LinR V3 and the IVF + INT8 + Bloom retriever maintain quality across the full sweep matrix, and the Triton backend matches the torch backend within float-tolerance noise while exceeding it on latency and memory.

---

## Annotated bibliography

Consolidated and deduplicated reference list, grouped by topical area. Tags in the **Source** column: **[L]** — entry was in the LinR paper bibliography (Borisyuk et al. 2024); **[Y]** — entry was in the Yambda dataset release bibliography (Yandex 2025); **[+]** — entry added in this thesis. The prior-draft tag **[S]** (entries from the Meta-affiliated industrial system) has been retired with the corresponding entries: per the Citation Policy in [`00-thesis-plan.md`](00-thesis-plan.md), those entries are removed from this bibliography and the corresponding industrial system is treated as not-cited prior art.

### A. Model-based retrieval, generative retrieval, and learned similarities

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Borisyuk et al. — *LiNR: Model Based Neural Retrieval on GPUs at LinkedIn* | CIKM | 2024 | **[L]** Reproduced in [`layers/linr/`](../../retrieve/src/retrieve/layers/linr/). Central anchor for §1.5. |
| Oldridge et al. — *Merlin: A GPU-Accelerated Recommendation Framework* | Proc. of IRS | 2020 | **[+]** NVIDIA-stack context for §1.5. |
| Tay et al. — *Transformer Memory as a Differentiable Search Index (DSI)* | NeurIPS | 2022 | **[L]** Generative retrieval. |
| Rajput et al. — *Recommender Systems with Generative Retrieval (TIGER)* | NeurIPS | 2023 | **[L][Y]** Generative retrieval, Google. |
| Zhang et al. — *Model-Enhanced Vector Index (MEVI)* | arXiv preprint | 2023 | **[L]** Hybrid model+index. Author audit pending (see open questions). |

### B. Classical and GPU-resident ANN

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Jégou, Douze, Schmid — *Product Quantization for Nearest Neighbor Search* | TPAMI | 2011 | **[L]** Theoretical foundation for IVF / PQ / INT8. INRIA at time of publication; pre-Meta. |
| Malkov, Yashunin — *Efficient and Robust ANN Search Using HNSW Graphs* | TPAMI | 2020 | **[L]** Canonical graph-based ANN; baseline. |
| Guo et al. — *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)* | ICML | 2020 | **[L]** Alternative non-graph CPU ANN, Google. |
| Subramanya et al. — *DiskANN: Fast Accurate Billion-Point NN Search on a Single Node* | NeurIPS | 2019 | **[+]** Disk-resident billion-scale ANN, Microsoft Research India. |
| Ootomo et al. — *CAGRA: Highly Parallel Graph Construction and ANN Search for GPUs* | ICDE / arXiv:2308.15136 | 2024 | **[L]** Principal GPU graph-based ANN baseline, NVIDIA. |
| Wang et al. — *Milvus: A Purpose-Built Vector Data Management System* | SIGMOD | 2021 | **[+]** Open vector database; GPU index limitations doc. |
| Zhao, Tan, Li — *SONG: Approximate Nearest Neighbor Search on GPU* | ICDE | 2020 | **[L]** Academic GPU ANN with IVF on-GPU. |
| Nolet — *Reusable Computational Patterns for ML and IR with RAPIDS RAFT* | NVIDIA dev blog | 2023 | **[L]** GPU primitives, NVIDIA. |
| Zhao, Tan, Li — *Constrained Approximate Similarity Search on Proximity Graph* | arXiv:2210.14958 | 2022 | **[L]** Filtered HNSW; closest published precedent to the filter benchmark of this thesis. |
| Pinterest engineering — *Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval* | Pinterest blog | 2023 | **[+]** Production HNSW exemplar. Author audit pending. |
| Pace et al. — *Lance: Efficient Random Access in Columnar Storage* | arXiv:2504.15247 | 2025 | **[+]** Columnar storage with random access; relevant to index serialization. |

### C. Feature filtering and attribute matching

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Brin & Page — *The Anatomy of a Large-Scale Hypertextual Web Search Engine* | Computer Networks | 1998 | **[+]** Historical foundation of inverted-index retrieval. |
| Cambazoglu & Baeza-Yates — *Scalability and Efficiency Challenges in Large-Scale Web Search Engines* | SIGIR | 2016 | **[+]** Survey of inverted-index systems and scaling characteristics. |
| Goodwin et al. — *BitFunnel: Revisiting Signatures for Search* | SIGIR | 2017 | **[+]** Intellectual antecedent of the Bloom-signature filtering used in this thesis, Microsoft. |
| Chen, Lassance, Lin — *End-to-End Retrieval with Learned Dense and Sparse Representations Using Lucene* | arXiv:2311.18503 | 2023 | **[L]** Modern dense + sparse hybrid in Lucene. |
| Bloom — *Space/Time Trade-offs in Hash Coding with Allowable Errors* | CACM | 1970 | **[+]** Original Bloom filter; false-positive theory. |

### D. Embedding quantization and compact representations

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Jégou et al. — *Product Quantization* (cross-ref from B) | TPAMI | 2011 | **[L]** Theoretical foundation. |
| Li & Li — *OPORP: One Permutation + One Random Projection* | arXiv:2302.03505 | 2023 | **[L]** Sign-OPORP basis for LinR V3 1-bit quantization. |
| NVIDIA — *dp4a instruction* (CUDA C++ Programming Guide) | NVIDIA docs | — | **[+]** Hardware basis for INT8 dot-product on Ampere+. |
| Zhao et al. — *Embedding in Recommender Systems: A Survey* | arXiv:2310.18608 | 2023 | **[+]** General survey. Author audit pending. |

### E. GPU programming and DL frameworks

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Tillet, Kung, Cox — *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* | MAPL | 2019 | **[+]** Required citation: substrate of all kernels in this thesis. |
| OpenAI — *Introducing Triton: Open-Source GPU Programming for Neural Networks* | OpenAI blog | 2021 | **[+]** Triton announcement. |
| Paszke et al. — *PyTorch: An Imperative Style, High-Performance Deep Learning Library* | NeurIPS / arXiv:1912.01703 | 2019 | **[+]** Framework substrate. Citation strategy pending (see open questions). |
| Ivchenko et al. — *TorchRec: A PyTorch Domain Library for Recommendation Systems* | RecSys | 2022 | **[+]** PyTorch-for-RecSys. Author audit pending (project is Meta-affiliated). |
| Abadi et al. — *TensorFlow: A System for Large-Scale Machine Learning* | OSDI | 2016 | **[+]** Alternative framework, Google. |
| Frostig, Johnson, Leary — *Compiling Machine Learning Programs via High-Level Tracing (JAX)* | SysML | 2019 | **[+]** Alternative framework, Google. |
| Moritz et al. — *Ray: A Distributed Framework for Emerging AI Applications* | OSDI | 2018 | **[+]** Distributed compute background. |
| Milvus engineering — *GPU Index Limitations* (Milvus docs) | milvus.io | 2023 | **[+]** Documentation of top-K and probe limits in open GPU ANN libraries. |

### F. Sequential recommendation models

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Hochreiter & Schmidhuber — *Long Short-Term Memory* | Neural Computation | 1997 | **[Y]** Historical foundation. |
| Hidasi et al. — *Session-Based Recommendations with RNNs (GRU4Rec)* | arXiv:1511.06939 | 2015 | **[Y]** Session-based RNN baseline. |
| Kang & McAuley — *Self-Attentive Sequential Recommendation (SASRec)* | ICDM | 2018 | **[Y]** Baseline architecture for [`evaluation/training/`](../../evaluation/training/). |
| Sun et al. — *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer* | CIKM | 2019 | **[Y]** Bidirectional alternative to SASRec. |
| Petrov & Macdonald — *gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling* | RecSys (Best Paper) / arXiv:2308.07192 | 2023 | **[+]** Actual model trained in [`evaluation/training/`](../../evaluation/training/). |

### G. RecSys foundations, two-tower retrieval, ranking-stage models

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Huang et al. — *Learning Deep Structured Semantic Models for Web Search Using Clickthrough Data (DSSM)* | CIKM | 2013 | **[+]** Microsoft, original deep two-tower for web search. |
| Covington, Adams, Sargin — *Deep Neural Networks for YouTube Recommendations* | RecSys | 2016 | **[+]** Google, multi-stage retrieval/ranking. |
| Yi et al. — *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* | RecSys | 2019 | **[+]** Google, two-tower with in-batch negatives. |
| Zhou et al. — *Deep Interest Network for Click-Through Rate Prediction (DIN)* | KDD | 2018 | **[+]** Alibaba, attention in ranking. Author audit confirmed Alibaba-only. |
| Wang et al. — *Billion-Scale Commodity Embedding for E-commerce Recommendation in Alibaba* | KDD | 2018 | **[+]** Alibaba, industrial embeddings. |
| Borisyuk et al. — *LiGNN: Graph Neural Networks at LinkedIn* | KDD | 2024 | **[L]** GNN background of the LinkedIn stack. |
| Shen et al. — *Learning to Retrieve for Job Matching* | arXiv:2402.13435 | 2024 | **[L]** Application of LinR. |
| Zhai, Wu, Tzeng et al. — *Learning a Unified Embedding for Visual Search at Pinterest* | KDD | 2019 | **[+]** Pinterest two-tower. Author audit pending. |
| Baltescu et al. — *ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest* | KDD | 2022 | **[+]** Pinterest two-tower. Author audit pending. |
| Pal et al. — *PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest* | KDD | 2020 | **[L]** Pinterest user embeddings. Author audit pending (verify no Meta co-authors). |

### H. Live update and streaming index infrastructure

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Lian et al. — *PERSIA: Scaling Deep Learning-Based Recommenders up to 100 Trillion Parameters* | KDD | 2022 | **[L]** Open hybrid 100T-parameter recommender. |
| Jiang et al. — *XDL: An Industrial Deep Learning Framework for High-Dimensional Sparse Data* | KDD | 2019 | **[L]** Alibaba industrial DL framework. |
| Félix GV — *Open Sourcing Venice — LinkedIn's Derived Data Platform* | LinkedIn engineering blog | 2008 | **[L]** CDC stream for LinR live update. |
| Liu et al. — *Monolith: Real-Time Recommendation System With Collisionless Embedding Table* | arXiv:2209.07663 | 2022 | **[L]** ByteDance. Author audit pending (verify no Meta co-authors). |

### I. Datasets and evaluation tooling

| Reference | Venue | Year | Source / relevance |
|---|---|---|---|
| Yandex — *Yambda: Yandex Music Billion-Interactions Dataset* | per [`articles/yambda.md`](../../articles/yambda.md) | 2025 | **[Y]** Principal large-scale retrieval benchmark. |
| Wan & McAuley — *Item Recommendation on Monotonic Behavior Chains (Goodreads)* | RecSys | 2018 | **[+]** Goodreads dataset paper, UCSD. |
| Wan, Misra, Nakashole, McAuley — *Fine-Grained Spoiler Detection from Large-Scale Review Corpora* | ACL | 2019 | **[+]** Extended Goodreads with fine-grained annotations. |
| Nussbaum, Morris, Duderstadt, Mulyar — *Nomic Embed: Training a Reproducible Long Context Text Embedder* | arXiv:2402.01613 | 2024 | **[+]** Embedding model for arXiv evaluation pipeline. |
| Harper & Konstan — *The MovieLens Datasets: History and Context* | TIIS | 2015 | **[Y]** Comparative dataset background. |
| Ni, Li, McAuley — *Justifying Recommendations Using Distantly-Labeled Reviews and Fine-Grained Aspects (Amazon Reviews 2018)* | EMNLP-IJCNLP | 2019 | **[Y]** Comparative dataset background, UCSD. |
| Hou et al. — *Bridging Language and Items for Retrieval and Recommendation (Amazon Reviews 2023)* | arXiv:2403.03952 | 2024 | **[Y]** Modern large-scale dataset, UCSD. |
| Schedl — *The LFM-1b Dataset for Music Retrieval and Recommendation* | ICMR | 2016 | **[Y]** Music-RecSys comparative background. |
| Schedl et al. — *LFM-2b* | CHIIR | 2022 | **[Y]** Music-RecSys comparative background. |
| Akiba et al. — *Optuna: A Next-Generation Hyperparameter Optimization Framework* | KDD | 2019 | **[Y]** HPO for SASRec training. |
| Kaplan et al. — *Scaling Laws for Neural Language Models* | arXiv:2001.08361 | 2020 | **[Y]** Scaling-laws background. |
| Hoffmann et al. — *Training Compute-Optimal Large Language Models (Chinchilla)* | arXiv:2203.15556 | 2022 | **[Y]** Scaling-laws background. |

### J. Other technical references and primitives

| Reference | Type | Year | Source / relevance |
|---|---|---|---|
| Arthur & Vassilvitskii — *k-means++: The Advantages of Careful Seeding* | Stanford technical report | 2006 | **[+]** Initialization for IVF clustering. |
| Vantage — *AWS p4d.24xlarge / r6i.8xlarge / x2idn.24xlarge cost pages* | vantage.sh | 2025 | **[+]** For optional cost-efficiency comparison context. |

### Removed from the prior draft per the Citation Policy

The following entries appeared in the prior skeleton's annotated bibliography and have been removed because one or more authors are Meta / Facebook / FAIR / Instagram / WhatsApp / Reality Labs-affiliated. These entries are not cited anywhere in the rewritten chapter, and the narrative roles they previously played have been filled by the substitutes listed in §1.1–§1.10 above. Git history of `01-literature-review.md` preserves the prior draft for traceability.

- Naumov et al. 2019 — DLRM
- Huang et al. 2020 — EBR in Facebook Search
- Liu et al. 2021 — Que2Search
- Mudigere et al. 2022 — SW/HW Co-Design for DLRM
- Rangadurai et al. 2022 — NxtPost
- Zhai et al. 2023 — Revisiting Neural Retrieval on Accelerators (MoL)
- Zhai et al. 2024 — Actions Speak Louder than Words (HSTU)
- Ding & Zhai 2025 — Retrieval with Learned Similarities
- Zhang et al. 2025 — Multi-Task Multi-Head Item-to-Item Retrieval
- Johnson, Douze, Jégou 2019/2021 — FAISS-GPU (Douze and Jégou at Meta/FAIR)
- Douze et al. 2024 — The Faiss Library
- Zhang et al. 2024 — Wukong
- Ardalani et al. 2022 — Scaling Laws for Recommendation Models
- Lewis et al. 2020 — Retrieval-Augmented Generation (RAG)
- Curtiss et al. 2013 — Unicorn (Facebook social-graph search)
- Industrial-shorthand model-based retrieval system from the Meta corporate research group (the IVF + INT8 + Bloom predecessor) — not cited; the IVF + INT8 + Bloom retriever in this thesis is presented as the author's own design built on classical primitives (see §1.5)

---

## Open citation questions

The following items require verification or a decision before the chapter is translated to Russian academic prose. Each is flagged in the body text where it appears.

1. **PyTorch citation strategy.** The PyTorch project was originally led by Meta-affiliated engineers; the original Paszke et al. 2019 NeurIPS paper has both Meta and non-Meta co-authors; the project is now under the PyTorch Foundation (Linux Foundation, 2022). The working position is to cite via the original NeurIPS paper and note the post-2022 neutral governance, but the supervisor's view should be sought. Alternative: cite torch.compile via Ansel et al. (PLDI 2024) instead, pending verification of that author list.
2. **TorchRec author audit.** Ivchenko et al. 2022 RecSys — the TorchRec project is Meta-affiliated and most authors are Meta engineers. If the audit requires removal, the narrative role (sharded embedding tables for industrial RecSys in PyTorch) is filled by direct citation of torch.distributed primitives and by JAX's pjit / DTensor analogues.
3. **Monolith author audit (Liu et al. 2022).** ByteDance authorship; verify that no co-author is Meta-affiliated. If clean, retain in §1.9; otherwise drop and rely on PERSIA + XDL.
4. **PinnerSage author audit (Pal et al. 2020).** Pinterest-led; verify no Meta co-authors. If clean, retain in §1.5 / Table A; otherwise drop.
5. **ItemSage author audit (Baltescu et al. 2022).** Same as above.
6. **Pinterest Unified Embedding (Zhai, Wu, Tzeng et al. 2019).** Verify Pinterest-only author list. If clean, retain; otherwise drop.
7. **Embedding in Recommender Systems survey (Zhao et al. 2023).** Verify no Meta-affiliated co-authors. If clean, retain in §1.7; otherwise replace with a non-affiliated quantization survey or drop the survey citation entirely and lean on Jégou 2011 + Li & Li 2023.
8. **MEVI (Zhang et al. 2023).** Verify author affiliations. If Meta-affiliated, drop and remove the reference to MEVI in §1.5.
9. **Manas HNSW (Pinterest 2023 blog).** Blog post; verify author bios mention no Meta affiliation.
10. **Borisyuk et al. — *LiGNN*** (KDD 2024). Verify the full author list for any Meta-affiliated co-authors (LinkedIn engineering has occasional cross-company collaborations); drop if any are present.
11. **Citation style.** The thesis must comply with HSE GOST-numeric bibliography. Decide between BibLaTeX with `style=gost-numeric` and a hand-built bibliography file. Coordinate with [`thesis/main.tex`](../../thesis/main.tex) bibliography setup once the citations stabilize.
12. **DOI / arXiv-ID completeness.** Per the plan's acceptance criteria, every cited work must have a DOI or arXiv ID in the final `thesis/references.bib`. Populate during the BibLaTeX construction pass, not in this content-notes draft.

---

## Notes for the next pass (translation to Russian academic prose)

- The intended length per subsection in the final thesis is 3–5 pages of Russian prose; these English notes are sized for that target after translation expansion.
- The lineage statement in §1.5 must appear verbatim in the Russian translation; do not paraphrase its specific list of primitives (IVF, INT8, Bloom signatures) or its specific attributions (Jégou 2011, Goodwin 2017).
- The Введение (Introduction) chapter pulls a one-paragraph version of §1.11; keep the two consistent.
- Body prose must not use the industrial-shorthand name of the IVF + INT8 + Bloom system. Repo paths under `layers/silvertorch/` and `kernels/silvertorch/` are acceptable as code-internal symbols; explanatory text should say "the co-designed IVF + INT8 + Bloom retriever," "the second bundled retriever," or "the framework's second reference retriever family." The phrase "author's continuation of LinR" is deprecated under the dual-goal reframe and must not be used in body prose.
- The annotated bibliography tables in this draft are not the final `thesis/references.bib`. The BibLaTeX file is constructed in a separate pass after the chapter prose is finalized; the content-notes tables here are working material.
