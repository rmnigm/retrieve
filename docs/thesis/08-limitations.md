# Ch.7 — Limitations and Future Directions (Ограничения и направления развития) — Reference Notes

**Status:** English reference notes for the downstream Russian-prose writing
agent. Not finished prose. Every claim is anchored to either a `path:line`
code reference, a `[DATA: ...]` pointer into the results, or a flagged
`[CITE: ... — KEEP/DROPPED]` candidate.

**Hard constraints carried from Citation Policy (00-thesis-plan.md §46–§94):**
- No paper with any Meta/Facebook/FAIR/Reality Labs/Instagram/WhatsApp
  author is cited (past or present on that paper). FAISS / FAISS-GPU
  family is excluded.
- The co-designed IVF+INT8+Bloom retriever is described neutrally
  ("co-designed IVF+INT8+Bloom retriever", "the co-designed retriever",
  "the IVF+INT8 module") and presented as the author's own continuation
  of the LinR research line. The internal code symbol `SilverTorch` may
  appear in code anchors / file paths only, never in body prose.

---

## 7.0 Scope and framing

- This chapter does two things: (i) catalogues the design boundaries that
  the open-source release inherits, and (ii) lays out the engineering
  roadmap derived from `docs/plans/` and from the failure modes recorded
  in Chapter 6 (results).
- The chapter does **not**: critique the execution of the experiments;
  question methodology choices already justified in Ch.5; or relitigate
  the algorithm-attribution decisions documented in Ch.1 §1.5 and §1.11.
- Cross-reference rule for the writer: every limitation that is also a
  failure mode in Ch.6 cites the relevant §6.x section number and the
  underlying `[DATA: ...]` pointer so readers can verify quickly.
- Organising principle: §7.1–§7.7 are scope boundaries (each pairs a
  limitation with a concrete future direction); §7.8 lists engineering
  roadmap items pulled from `docs/plans/`; §7.9 collects "interpretive
  limitations" — failure modes that are not yet roadmap items but that
  the honest reader should know about.

---

## 7.1 Single-GPU only — no multi-GPU index sharding

- **Body.** The entire library assumes a single CUDA device. The item
  embedding matrix and (where applicable) the IVF cluster table, the
  Bloom signature buffer, and the INT8-quantized table all live in one
  GPU's HBM. There is no tensor-parallel split across devices and no
  CPU-offload tier.
- **Failure regime.** At fp16 with $N = 10^7$ and $d = 256$ the dense
  index alone is $\approx 5{,}120$ MiB; the same configuration at d=128
  is $\approx 2{,}560$ MiB. The Yambda-5b cell at d=128 already records
  a `linr_v1` index of 1310 MiB (≈ 1.31 GiB) on `N ≈ 5.4M`
  [DATA: docs/thesis/07-results.md §6.3.4 yambda-5b memory rows]. Pushing
  N or d further requires either a smaller-memory algorithm
  (`silvertorch` at 800.3 MiB on the same cell) or multi-GPU sharding,
  which the harness does not exercise.
- **What is also missing.** The benchmark harness in
  `evaluation/retrieval/` never launches a multi-process / multi-GPU run;
  there is no NVLink / NCCL collective code path; there is no test that
  asserts a sharded layout returns equivalent top-K to the single-GPU
  reference.
- **Future direction.**
  - Tensor-parallel cluster-list sharding for the IVF probe: each shard
    holds a disjoint subset of cluster centroids and their assigned
    items; the per-query probe runs in parallel and is followed by a
    cross-shard top-K reduction. The existing IVF probe kernel
    [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)
    is reusable per shard with minimal change.
  - Embedding-parallel split for `linr_v1` / `linr_v4` (dense matmul):
    each shard holds rows $[k \cdot N/P, (k+1) \cdot N/P)$ of the item
    matrix; a per-shard partial top-K is followed by an all-gather and
    a global merge.
- **Citation candidate.** [CITE: NVIDIA cuVS / RAFT ANN multi-GPU
  documentation — NVIDIA — KEEP]. The RAFT / cuVS team publishes
  reference designs for multi-GPU ANN that are vendor-neutral and
  available as substitutes for FAISS-GPU references.

---

## 7.2 Offline indexing only — no live updates

- **Body.** Index construction is one-shot inside the layer constructor.
  After `__init__` returns, the item embedding table, the IVF cluster
  assignment, the per-row INT8 scales, and the packed Bloom signatures
  are all frozen. Updating a single item embedding requires
  re-instantiating the layer and re-running the build. Bloom signatures
  in particular cannot be edited in place once packed because the
  underlying word-aligned bitmap mixes the bit positions of different
  items inside the same 64-bit word.
- **Failure regime.** Any streaming RecSys workload where items churn
  on minute-or-shorter timescales is unsupported. Yambda's temporal
  split is honoured (train / validation / test partition by timestamp),
  but the index is rebuilt at the split boundary, not updated during
  serving.
- **Future direction (roadmap item).**
  `docs/plans/live-update-api.md` specifies an upsert / delete API for
  V1/V2/V3 LinR layers plus the BloomFilter and ExactAttributeFilter
  modules. The design uses preallocated buffers with tombstoned rows,
  plus mode-specific stable-sort permutations for live-set compaction
  in sparse paths. All new code is constrained to stay
  `torch.export`-clean: no `.item()`, no `.cpu()`, no
  `Optional[Tensor]` arguments. The plan explicitly covers
  PostfilterKNN, PrefilterKNN, OneBitKNN, FullScanKNN, BloomFilter,
  and ExactAttributeFilter; SilverTorch (the co-designed retriever)
  is intentionally out of scope for the first live-update iteration
  because the IVF clustering would itself need to be re-balanced on
  cluster-skew detection.
- **Citation candidates.** Vendor-neutral priors for incremental index
  update:
  - [CITE: Apache Lucene segment merge — Apache Software Foundation —
    KEEP] (incremental-build prior; widely cited)
  - [CITE: Vespa partial-update documentation — Vespa AI / Yahoo (now
    independent) — KEEP] (real-time update prior)

---

## 7.3 Inference only — no joint learning of similarity functions

- **Body.** Every retrieval module operates on frozen embeddings. SASRec
  trains the query encoder once on Goodreads / Yambda; nomic-embed-v1.5
  produces the arXiv corpus encoding without any thesis-side training.
  The similarity function is fixed (inner product or its 1-bit / INT8
  surrogate); no end-to-end gradient flows into quantization codebooks,
  into the OPORP rotation matrix
  [retrieve/src/retrieve/layers/linr/one_bit_knn.py](retrieve/src/retrieve/layers/linr/one_bit_knn.py),
  or into the IVF centroids
  [retrieve/src/retrieve/layers/utils/kmeans.py](retrieve/src/retrieve/layers/utils/kmeans.py).
- **Failure regime.** (i) The similarity surface cannot adapt to
  filter-aware workloads where the candidate distribution is
  non-stationary (e.g. a popularity-dependent prior). (ii) The
  OPORP rotation cannot be fine-tuned jointly with the encoder; the
  rotation is sampled once and held fixed. (iii) IVF centroids are
  fit by k-means rather than learned to minimise downstream Recall@K.
- **Future direction.**
  - Learned IVF assignment via differentiable clustering (Gumbel-Softmax
    over centroids, or neural quantization).
  - Anisotropic quantization in the style of [CITE: SCANN — Guo et al.
    2020 — Google — KEEP], which optimises codebooks for inner-product
    error rather than reconstruction error.
  - Generative retrieval as an alternative paradigm
    [CITE: TIGER — Rajput et al. 2023 — Google — KEEP].
- **Important caveat.** This is a scope choice, not a deficit — the
  LinR research line is itself inference-time only
  [CITE: Borisyuk et al. 2024, CIKM — LinkedIn — KEEP], and the
  thesis explicitly positions itself as a reproducibility study of
  that line plus the author's continuation of it; learned similarity
  is a different research line.

---

## 7.4 Benchmark coverage — three datasets, no web-scale text retrieval

- **Body.** Three datasets are evaluated:
  - **Goodreads:** ~797k books, 227.5M reads, 313k held-out test users,
    SASRec query encoder at $d \in \{64, 128, 256\}$ trained with
    gBCE loss.
  - **arXiv:** 2.99M papers, embeddings pre-computed with
    nomic-embed-v1.5 (no SASRec training); 10k test queries;
    Recall@100 ceiling = 1.0 by construction (the ground truth IS
    the exact top-K).
  - **Yambda (Yandex Music 2025):** two training scales — 500m
    (3.06M tracks, 466.5M events, $d \in \{64, 128, 256\}$) and
    5b (5.37M tracks, 4.65B events, $d \in \{64, 128\}$).
  [DATA: docs/thesis/04-datasets-notes.md;
   docs/thesis/results-data/results/sasrec_quality_ceiling.csv]
- **What is missing.** No web-scale corpus is evaluated: MS-MARCO
  passage (8.8M passages), the BEIR composite (18 retrieval datasets),
  LAION-style image-text retrieval at 10⁸ scale. The Goodreads /
  Yambda / arXiv triple covers two RecSys regimes (book and music
  recommendation) plus one text-retrieval regime (arXiv with constructed
  ground truth) but does not exercise the selectivity / filter-skew
  profiles characteristic of web search.
- **Reason for the scope cap.** Single-author thesis; encoder
  training time (SASRec, two scales of Yambda) already dominates
  the harness work; arXiv is chosen as the text-retrieval proxy
  because its exact ground truth makes Recall ceiling unambiguous.
- **Failure regime / known unknowns.** Web-scale corpora typically
  have vastly different filter selectivity distributions — a
  query like "papers about diffusion models from MIT after 2022"
  on MS-MARCO would touch a different (smaller, more skewed)
  candidate set than the goodreads `c0_genre` and arxiv `cat_q`
  clauses tested in §6.4. The filter-suite results in §6.4 should
  not be extrapolated to web search without re-evaluation.
- **Future direction.** Adding a BEIR / MS-MARCO benchmark requires
  (a) an MTEB-style sentence-encoder per dataset (off-the-shelf,
  no thesis-side training); (b) a new dataset config in
  `evaluation/config/`; (c) attribute-extraction step for filter
  evaluation. The harness is config-driven and reuses the same
  oracle / runner stack, so the marginal effort is integration, not
  new architecture.
- **Citation candidates:**
  - [CITE: BEIR — Thakur et al. 2021 — UKP-Lab TU Darmstadt — KEEP]
  - [CITE: MTEB — Muennighoff et al. 2023 — HuggingFace — KEEP]
  - [CITE: MS-MARCO — Bajaj et al. 2016 — Microsoft — KEEP]

---

## 7.5 No comparison with CUDA-native GPU ANN libraries on identical hardware

- **Body.** The co-designed retriever is benchmarked against the LinR
  V1/V2/V3/V4 algorithms on the same A100 hardware, but there is no
  head-to-head against hand-tuned CUDA libraries (NVIDIA CAGRA, Milvus
  GPU, SCANN, SONG). The Triton ceiling versus the CUDA ceiling on
  identical primitives (IVF probe, INT8 dot, Bloom membership test)
  is therefore an open empirical question.
- **Failure regime.** A 2× gap between Triton and a CUDA-native IVF-PQ
  implementation would not invalidate the thesis's qualitative findings
  (memory savings of INT8, value of kernel fusion, dominance of
  co-designed retrievers on Pareto frontier) but it would shift the
  absolute speedup numbers in §6.2 — and the writer should NOT
  overclaim absolute speedup until that comparison is done.
- **Explicit exclusion (citation policy).** FAISS / FAISS-GPU is the
  natural comparison baseline and is excluded because its core authors
  are Meta-affiliated. [CITE: FAISS — Johnson, Douze, Jégou — DROPPED,
  Meta (Douze and Jégou are at FAIR)].
- **Acceptable substitutes** (per Replacements table in 00-thesis-plan.md §85):
  - [CITE: CAGRA — Ootomo et al. 2024 — NVIDIA — KEEP]
  - [CITE: Milvus — Wang et al. 2021 — Zilliz — KEEP]
  - [CITE: SCANN — Guo et al. 2020 — Google — KEEP]
  - [CITE: SONG — Zhao et al. 2020 — Brown University — KEEP]
- **Future direction.** Integrate a CAGRA or Milvus baseline as a new
  `impl` value in `evaluation/retrieval/algos/`, exposed via the same
  CLI used for `linr_v1`/`linr_v3`/`linr_v4`/`silvertorch`. The
  harness's per-implementation latency / quality protocols are
  identical, so the comparison is config + adapter work, not new
  measurement methodology.

---

## 7.6 INT8 quantization has an empirical, not formal, error bound

- **Body.** Two INT8 variants are evaluated:
  - **Per-row INT8** in `linr_v4` via the helper that produces a
    per-row scale + an INT8 row, called from
    [retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py](retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py).
  - **Global-scale INT8** as the silvertorch default, using a single
    scalar scale for the entire INT8 table; see
    [retrieve/src/retrieve/layers/utils/quantize.py](retrieve/src/retrieve/layers/utils/quantize.py).
  Both are validated experimentally: `linr_v4` records zero measured
  Recall@100 loss across 11/11 quality cells for the SASRec / nomic
  embeddings actually used in this thesis
  [DATA: docs/thesis/07-results.md §6.6.3 rank stability table].
  No formal worst-case Recall bound is derived; the empirical claim
  depends on embedding distribution properties (ℓ2-normalised rows,
  low per-row dynamic range, well-behaved row norms).
- **Failure regime.** Embeddings with heavy-tailed row norms (some
  contrastive vision models, embeddings from un-normalised heads) may
  saturate the INT8 range and silently lose Recall. The global-scale
  variant is more sensitive than per-row: silvertorch documentation
  in
  [retrieve/src/retrieve/layers/silvertorch/main.py](retrieve/src/retrieve/layers/silvertorch/main.py)
  notes a ~1–2% Recall@K drop versus the per-item variant on
  long-tailed row-norm distributions (Quality knob section in the
  module docstring). User-supplied embeddings that are not
  ℓ2-normalised are untested.
- **Future direction.**
  - Per-block / per-cluster scales (a hybrid between per-row and
    global) for IVF-clustered tables; expected to preserve the
    memory savings of global-scale while limiting saturation in
    outlier clusters.
  - INT4 path; would halve memory again at additional Recall risk.
  - A formal worst-case Recall bound via concentration inequalities
    on bounded random vectors (the embeddings are unit-norm; the
    quantization error is bounded; a Hoeffding-style bound for the
    inner-product error and a derived bound on top-K rank stability
    are tractable).
- **Citation candidates** (INT8 quantization context, vendor-neutral):
  - [CITE: LLM.int8() — Dettmers et al. 2022 — University of Washington
    — KEEP]
  - [CITE: AWQ — Lin et al. 2023 — MIT — KEEP]
- **Anchors:** [DATA: docs/thesis/07-results.md §6.6.3 rank stability
  table];
  [retrieve/src/retrieve/layers/utils/quantize.py](retrieve/src/retrieve/layers/utils/quantize.py);
  [retrieve/src/retrieve/layers/silvertorch/main.py](retrieve/src/retrieve/layers/silvertorch/main.py)
  (Quality knob docstring).

---

## 7.7 Filter API limited — no learned filters, no range predicates

- **Body.** The current filter API supports two predicate classes:
  - **BloomFilter** (paper-strict Bloom subset test) with canonical
    configuration $m_{\text{bits}} = 1024$, $k_{\text{hash}} = 5$;
    implementation in
    [retrieve/src/retrieve/layers/filters/bloom.py](retrieve/src/retrieve/layers/filters/bloom.py).
  - **ExactAttributeFilter** as AND-of-OR clauses over discrete
    categorical attributes; implementation in
    [retrieve/src/retrieve/layers/filters/exact_attribute.py](retrieve/src/retrieve/layers/filters/exact_attribute.py).
  Three predicate classes are NOT supported:
  - **Range predicates** (e.g. `year > 2020`, `price ∈ [a, b]`).
  - **Learned filters** (neural filter heads, learned candidate
    selectors).
  - **Full-text Boolean queries** (token-level positional matching).
- **Failure regime.**
  - (i) Catalogue properties like timestamps and prices require
    pre-bucketing into discrete categorical attributes. The bucket
    boundaries are lossy and bucket-size-sensitive; the queries
    "papers from 2023" and "papers from 2023-Q4" both become bucket
    lookups whose precision depends on bucket granularity.
  - (ii) Bloom collisions cause an irreducible Recall ceiling on
    highly selective sweeps. Specifically: on arxiv `all4`
    (most-selective clause combination), Recall@100 plateaus at
    ≈ 0.94 even as $n_{\text{probe}} \to 256$, against an exact
    baseline of ≈ 0.99 — an irreducible ~5% loss attributable to
    Bloom approximation, not to insufficient candidate count.
    [DATA: docs/thesis/07-results.md §6.5.5 / §6.5.1 bloom Recall
    ceiling on arxiv all4]
  - (iii) The codesigned exact-clause kernel
    [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py)
    already supports a `clause_is_reverse` semantic (XOR on clause
    membership), but the wiring from `evaluation/sweep` →
    `build_algorithm` → the algorithm wrapper does not pass the flag
    through. As a result, reverse-clause sweeps on Goodreads
    (`c1_lang_reverse`, `c0c1`, `all4`) record Recall@100 ≈ 0.0005–0.04
    for the co-designed retriever versus baseline 0.5–0.6 — a
    measurement artefact, not an algorithm limitation.
    [DATA: docs/thesis/07-results.md §6.4.1 / §6.4.2 INCOMPATIBLE
    reverse-clause rows]
- **Future direction.**
  - **Roadmap item.** `docs/plans/silvertorch-reverse-clause-wrapper-fix.md`
    is a mechanical pass-through fix for the `clause_is_reverse:
    Tensor` argument; the library and kernel already support it; only
    the eval-side wiring is missing. Post-fix + oracle rebuild,
    Goodreads filter-sweep coverage is restored.
  - **Larger Bloom signatures** ($m_{\text{bits}} \in \{2048, 4096\}$)
    trade memory for Recall recovery on selective sweeps; the
    silvertorch index format is parameterised so the change is config,
    not architecture.
  - **Range-predicate kernel** using sorted-by-key IVF clusters: items
    within each cluster are sorted by the range key, enabling a
    binary-search prefix-skip during probe; this combines well with
    the existing `probe + INT8 dot` fused path.
- **Anchors:**
  - [retrieve/src/retrieve/layers/filters/bloom.py](retrieve/src/retrieve/layers/filters/bloom.py) — Bloom impl
  - [retrieve/src/retrieve/layers/filters/exact_attribute.py](retrieve/src/retrieve/layers/filters/exact_attribute.py) — exact-attribute impl
  - [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) — fused exact kernel with `clause_is_reverse` XOR
  - [DATA: docs/thesis/07-results.md §6.4.1 — reverse-clause INCOMPATIBLE]
  - [DATA: docs/thesis/07-results.md §6.5.5 — Bloom Recall ceiling]

---

## 7.8 Engineering roadmap items (from `docs/plans/`)

The `docs/plans/` directory contains four active plans, each described
below. Roadmap scope is exactly these four plans plus the
measurement-gap reruns listed under §7.8.5.

### 7.8.1 Kernel-level optimizations (Stage 3, deferred from `docs/plans/00-roadmap.md`)

- **Hardware popcount in `oporp_1bit_match_topk`.** Replace the
  software popcount in the OPORP 1-bit Hamming kernel with the PTX
  `popc.b64` instruction. Expected modest speedup on the
  `linr_v3` path; the dim-scaling analysis in §6.6.3 shows the
  co-designed retriever already at $\alpha \approx 0.07$
  (median_ms $\propto d^{\alpha}$), so this tightening is most
  visible for `linr_v3` which currently relies on software popcount.
  [DATA: docs/thesis/07-results.md §6.6.3 dim-scaling exponents]
- **Allocator hygiene.** Remove the `torch.full(-inf)` pre-fill in
  the top-K scratch path; replace `cat`-to-pad with a
  pre-allocate-and-slice idiom; both reduce CUDA allocator pressure
  under autograd / repeated calls.

### 7.8.2 Live-update API (`docs/plans/live-update-api.md`)

- Upsert / delete API for V1/V2/V3 LinR layers plus the BloomFilter
  and ExactAttributeFilter modules. Preallocated buffers with
  tombstoned rows; mode-specific stable-sort permutations for
  sparse-path compaction. All new code is `torch.export`-clean
  (no `.item()`, no `.cpu()`, no `Optional[Tensor]`).
  See §7.2 for the user-facing motivation. SilverTorch is out of
  scope for the first iteration (cluster-rebalance complexity).

### 7.8.3 `torch.export` refactor (`docs/plans/torch-export-refactor.md`)

- Replace `Optional` constructor / forward arguments with layer-side
  mode flags — concretely `Literal["full", "masked", "candidates"]`
  — across PostfilterKNN (V1), PostfilterKNNInt8 (V4), PrefilterKNN
  (V2), OneBitKNN (V3), FullScanKNN (the torch_knn baseline), and
  the co-designed retriever. The plan promises **export-readiness**
  (each layer exports cleanly via `torch.export`) but explicitly
  does NOT promise shipping `.pt2` artefacts; consumers compose
  layers in their own models. The algorithm-wrapper layer threads
  `mode=` into constructors so eager-mode benchmarking exercises
  the export-clean code paths.

### 7.8.4 Silvertorch reverse-clause wiring fix (`docs/plans/silvertorch-reverse-clause-wrapper-fix.md`)

- Mechanical pass-through of `clause_is_reverse: Tensor` from
  `evaluation/sweep` → `build_algorithm` → the algorithm wrapper.
  Library and kernel already support the flag; only the eval-side
  plumbing is missing. Post-fix + oracle rebuild, Goodreads
  filter-sweep coverage is restored for `c1_lang_reverse`, `c0c1`,
  and `all4` clauses. See §7.7 for the user-facing impact.

### 7.8.5 Measurement-gap reruns (Stage 4 in `docs/plans/00-roadmap.md`)

- **Goodreads d128/d256 filter rerun against a fresh oracle.** The
  filter-suite cells for Goodreads d128 and d256 are tainted by a
  stale oracle cache: `linr_v1` (which is exact by construction)
  records Recall@100 ≈ 0.60 (d128) and 0.49 (d256) instead of the
  expected ≈ 1.0. Root cause: the cached
  `gt_topk_*.pt` ground-truth file was produced from an earlier
  item-embedding checkpoint and the cache validates only by shape,
  not by content hash. Hardening: add an `item_embs_hash` field to
  the cache metadata so subsequent runs invalidate on mismatch.
  Goodreads d64 filter results, all arXiv filter results, and all
  quality-suite results are unaffected.
  [DATA: docs/thesis/07-results.md §6.0 ACTION REQUIRED block;
  §6.4.2 stale-oracle diagnosis]
- **Yambda-5b-d256 cell extension.** Yambda-5b is currently
  evaluated only at $d \in \{64, 128\}$; the d=256 cell completes
  the dim-scaling grid.
- **Throughput (`throughput_qps`) and p99 latency schema
  extensions.** Current schema records median, p20, p80; p99 and
  qps require a small schema bump and re-run.
- **Extended batch-size sweeps.** Quality suite runs at $B = 1$
  only; filter suite covers $B \in \{1, 8, 16\}$; per-query
  amortisation behaviour beyond B=16 is unmeasured.
- **Per-kernel latency breakdown.** End-to-end forward-pass latency
  is the only timing measurement; the IVF probe vs INT8 dot vs
  Bloom membership vs top-K reduction breakdown would identify the
  binding-constraint kernel per cell.
- **Build / index-construction time.** Only registration-time peak
  memory is measured; wall-clock build time (KMeans iterations,
  quantization encoding, IVF assignment) is not.

---

## 7.9 Interpretive limitations (failure modes not yet roadmap items)

These are honest disclosures that the reader should know about. Each is
a measurement-time finding from Ch.6 that is currently flagged with
"hypothesis" or "suspected" language because the root cause has not
been definitively isolated.

- **`linr_v2` Triton ↔ torch non-parity on arXiv.** 270 of 543 arXiv
  pivot rows (49.7%) record $|\Delta\,\text{Recall}@100| > 10^{-3}$
  between the Triton-backend and the torch-reference implementations,
  with a systematic Triton < torch bias. Max $|\Delta R| =
  5.28 \cdot 10^{-3}$ at d=64, attenuating with dimension
  ($5.28 \cdot 10^{-3} \to 3.75 \cdot 10^{-3} \to 2.99 \cdot 10^{-3}$
  across $d \in \{64, 128, 256\}$). On Goodreads the same algorithm
  is bit-identical. Suspected cause: sparse-gather tile layout
  differs between Triton and torch on arXiv's sparser filter sets
  (0.5–5% pass-rate versus Goodreads' 28–53%), causing top-K
  tie-break shifts that drop 1–2 items per query at the boundary.
  Benign at the 1e-3 tolerance used for parity tests, but
  systematic and unexplained.
  [DATA: docs/thesis/07-results.md §6.7.7]
- **INT8 dense throughput regression at $B = 1$.** `linr_v4` is
  slower than `linr_v1` at single-query batch size despite using
  dp4a which has 4× theoretical INT8 throughput versus fp16:
  0.52 ms vs 0.36 ms on Goodreads-d128, 1.67 ms vs 1.07 ms on
  arXiv-d128. Root cause: the per-row dequantization epilogue and
  the per-row INT32 accumulator materialisation are not amortised
  at $B = 1$. The throughput advantage of dp4a only manifests at
  larger batch sizes; INT8's value at $B = 1$ is the 50% memory
  reduction (§6.3.4), not latency.
  [DATA: docs/thesis/07-results.md §6.6.3 speedup matrix;
  §6.2.1 observation]
- **Co-designed retriever index memory inflated on Goodreads.** The
  measured index memory on Goodreads is 2–5× above the algebraic
  floor (§6.3.4). Hypothesised cause: KMeans Lloyd-iteration
  intermediates are not freed before the memory measurement is
  taken; explicit
  `del kmeans_state; torch.cuda.empty_cache()` inside the fit step
  is the suggested fix. The anomaly does not appear on arXiv or
  Yambda, both of which have larger $N$ that swamps the KMeans
  scratch.
  [DATA: docs/thesis/07-results.md §6.3.4 silvertorch anomaly]
- **IVF cluster-layout latency anomaly at $n_{\text{lists}} = 8192$
  on arXiv.** Median latency jumps from 0.14 ms ($n_{\text{probe}} =
  4$) to 0.72–0.87 ms ($n_{\text{probe}} \geq 8$) despite a
  *smaller* probed pool than at $n_{\text{lists}} = 1664$ (42k vs
  232k items). Root cause: the kernel ships with a single hardcoded
  `DEFAULT_CONFIG` tuned for the $\sqrt{N}$ regime
  ($n_{\text{lists}} \approx 1664$ on arXiv); under-filled
  `BLOCK_P=256` tiles at small cluster sizes pay per-cluster
  launch overhead. Not a correctness bug; an autotune-coverage
  gap. The autotune CLI in `retrieve/src/retrieve/tune.py` exists
  but has not been applied to the 8192-list regime.
  [DATA: docs/thesis/07-results.md §6.5.1;
  [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py) (DEFAULT_CONFIG comments)]
- **Wide-attribute filter benchmark unused.** Dataset preparation
  writes `item_attrs_wide.pt` and `query_attrs_wide_*` columns,
  but zero YAML configs reference them. Wide-attribute (shelf /
  keyword bag) filtering exists at the API level but is unevaluated;
  only narrow attribute clauses ($C = 5$, $A_{\text{max}} = 4$)
  are reported.
  [DATA: docs/thesis/07-results.md §6.4.1 wide_1shelf finding]

---

## Citation candidates — kept vs dropped (Ch.7 only)

| Citation | Affiliation | Decision | Used in |
|----------|-------------|----------|---------|
| Borisyuk et al. 2024, CIKM (LinR) | LinkedIn | KEEP | §7.3 |
| CAGRA — Ootomo et al. 2024 | NVIDIA | KEEP | §7.5 |
| Milvus — Wang et al. 2021 | Zilliz | KEEP | §7.5 |
| SCANN — Guo et al. 2020 | Google | KEEP | §7.3, §7.5 |
| SONG — Zhao et al. 2020 | Brown University | KEEP | §7.5 |
| TIGER — Rajput et al. 2023 | Google | KEEP | §7.3 |
| LLM.int8() — Dettmers et al. 2022 | University of Washington | KEEP | §7.6 |
| AWQ — Lin et al. 2023 | MIT | KEEP | §7.6 |
| BEIR — Thakur et al. 2021 | UKP-Lab TU Darmstadt | KEEP | §7.4 |
| MTEB — Muennighoff et al. 2023 | HuggingFace | KEEP | §7.4 |
| MS-MARCO — Bajaj et al. 2016 | Microsoft | KEEP | §7.4 |
| NVIDIA cuVS / RAFT ANN docs | NVIDIA | KEEP (docs ref) | §7.1 |
| Apache Lucene segment merge | Apache Software Foundation | KEEP | §7.2 |
| Vespa partial-update docs | Vespa AI | KEEP | §7.2 |
| BitFunnel — Goodwin 2017 | Microsoft | KEEP (already in Ch.1) | §7.7 |
| Jégou et al. 2011 (IVF) | INRIA | KEEP (already in Ch.1) | implicit in §7.7 |
| KMeans / Lloyd algorithm | classical | KEEP (textbook, no cite needed) | §7.1 |
| FAISS / FAISS-GPU — Johnson, Douze, Jégou | Meta (Douze, Jégou at FAIR) | **DROPPED** | — |

**Kept:** 16 citations (including 2 already in Ch.1 reused here).
**Dropped:** 1 (FAISS family, with explicit justification in §7.5).

---

## Writer's notes (Ch.7)

- **Tone.** Factual catalogue, not apologetic. Each subsection is a
  pairing of "scope boundary now" + "concrete future direction" — the
  reader should come away with a clear sense of what the open-source
  release does NOT do and what the obvious next steps are. Avoid
  hand-wringing language ("unfortunately", "regrettably").
- **Cross-references.** For every failure mode that is also a §6.x
  finding, cite the section number and a `[DATA: ...]` pointer so the
  Russian writer can produce in-line hyperlinks.
- **Don't compete with §1.5 / §1.11.** The attribution argument (LinR
  reproduction + author's continuation) was made in Ch.1; Ch.7 should
  reference but not re-make it.
- **Lineage statement is NOT required here** — only in Введение and
  §1.5 of the literature review (per Citation Policy §82–§85).
- **Open questions / [TODO: clarify with author — ...] markers:**
  - Should §7.5 list specific CAGRA / Milvus / SCANN comparison plans,
    or only acknowledge the gap?
  - Should the `linr_v2` non-parity be in §7.9 (interpretive) or split
    out to a new §7.10 "Hardening tasks" section that also covers the
    oracle cache content-hash validation?
  - Should the codesigned retriever's reverse-clause issue (§7.7 iii)
    be moved to §7.9 since it's a wiring bug rather than an API
    limitation? The current placement under §7.7 is justified by the
    fact that the user-facing symptom is "the filter API doesn't
    handle reverse clauses on the co-designed retriever".

---

## Sources consulted (Ch.7)

- `docs/thesis/00-thesis-plan.md` (Ch.7 contract at §367–§391; Citation
  Policy at §46–§94; §1.5 / §1.11 framing)
- `docs/thesis/07-results.md` (§6.0 schema recap, §6.2 quality summary,
  §6.3.4 memory analysis, §6.4 filter suite, §6.5 deep sweeps,
  §6.6 dim-scaling, §6.7 parity findings)
- `docs/plans/00-roadmap.md` (Stage 3 deferred items + Stage 4
  measurement gaps)
- `docs/plans/live-update-api.md` (upsert/delete API design)
- `docs/plans/torch-export-refactor.md` (mode-flag refactor)
- `docs/plans/silvertorch-reverse-clause-wrapper-fix.md` (reverse-clause
  wiring fix)
- `retrieve/src/retrieve/layers/silvertorch/main.py` (Quality knob
  docstring; default `n_lists`, `n_probe`, $m_{\text{bits}}$,
  $k_{\text{hash}}$)
- `retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py` (per-row
  INT8 dequant epilogue)
- `retrieve/src/retrieve/layers/linr/one_bit_knn.py` (OPORP rotation,
  fixed at construction)
- `retrieve/src/retrieve/layers/utils/quantize.py` (global vs per-row
  INT8 variants)
- `retrieve/src/retrieve/layers/utils/kmeans.py` (KMeans for IVF
  centroids — frozen post-build)
- `retrieve/src/retrieve/layers/filters/bloom.py` (BloomFilter impl)
- `retrieve/src/retrieve/layers/filters/exact_attribute.py`
  (ExactAttributeFilter impl)
- `retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py`
  (IVF+INT8 fused kernel; `DEFAULT_CONFIG` notes)
- `retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py`
  (fused exact-clause kernel with `clause_is_reverse` XOR)
- `retrieve/src/retrieve/tune.py` (autotune CLI, mentioned in §7.9)
