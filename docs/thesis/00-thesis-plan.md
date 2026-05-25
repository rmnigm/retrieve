# Thesis Compilation Plan

**Goal:** A design doc that downstream agents use to compile a Russian-language MSc thesis at HSE structured around **two deliverables**:

1. **`retrieve` framework** — a minimally working open-source PyTorch package (`torchretrieve` on PyPI) of efficient GPU retrieval layers (LinR family + a composite IVF+INT8+Bloom retriever), compatible with `torch.compile` and arbitrary PyTorch encoders (decoupled from the query-encoder choice).
2. **Comprehensive empirical evaluation** of the framework's reference implementations on three open-source RecSys / IR datasets (Goodreads, arXiv, Yambda), sweeping quality, latency, memory, batch size, filter selectivity, and parameter sensitivity.

**Positioning anchor — analogous to Liger Kernel (Hsu et al. 2024, LinkedIn; safe under citation policy).** Liger Kernel packages efficient Triton-optimized kernels for LLM *training* as drop-in PyTorch `nn` layers, shipped with benchmarks; it claims no algorithmic novelty. `retrieve` adopts the same product shape for GPU *retrieval (inference)*: a small collection of drop-in PyTorch layers with documented per-layer benchmarks. The analogy frames both the scope (small, focused, library-shaped — not a full retrieval system with live updates, sharding, etc.) and the contribution (engineering + measurement, not novel algorithms).

Paper-faithful reproduction of LinR (Borisyuk et al. 2024) is a quality bar for the reference implementations — not the thesis's primary contribution. The framework's value is being usable as a library by downstream PyTorch projects; the evaluation's value is breadth across datasets, encoders, and operating regimes.

**Output artifact:** `thesis/main.tex` (HSE/GOST template) + per-chapter Markdown drafts in `docs/thesis/`, then weaved into LaTeX `\input{}` files.

---

## Context

### What exists
- **Code:** `retrieve/` (PyPI `torchretrieve`) — Triton + torch layers and kernels for LinR (postfilter/prefilter/1-bit OPORP) and SilverTorch (IVF + INT8 + Bloom). `evaluation/` — sweep harness over Goodreads, arXiv, Yambda; SASRec/gSASRec training; results in `.perkernel/` JSON.
- **Thesis skeleton:** `thesis/main.tex` — Russian, XeLaTeX/LuaLaTeX, GOST-numeric bib, HSE title page. **Title and author on the title page are placeholders from a different topic and must be replaced.**
- **Literature-review draft:** `docs/thesis/01-literature-review.md` — 11-subsection skeleton + 10-table annotated bibliography. **Currently violates the Meta-citation constraint (see Citation Policy below) and requires a full rewrite.**
- **System docs:** `docs/system/{architecture,kernels,filtering,testing,evaluation,checkpoints}.md` — internal documentation, reusable as primary source material for Ch.4 (Implementation) and Ch.5 (Evaluation Protocol).
- **Source articles:** `articles/{linr,silvertorch,yambda}.md` — author notes from the source papers.
- **Paper-conventions field guide:** [`docs/thesis/research_of_paper_conventions.md`](research_of_paper_conventions.md) — survey of reproducibility / ANN-benchmark / kernel-paper conventions (MLRC + ReScience C; Aumüller ANN-Benchmarks plot grammar; Big-ANN '23 filtered-track protocol; FlashAttention / Liger-Kernel kernel-paper shape; canonical metrics, datasets, plots). Use as a QC checklist for the empirical-evaluation pillar (Ch.5 protocol + Ch.6 results) and for kernel reporting in Ch.4. **Read with two caveats:** (a) the guide is written for D&B/MLSys A* reproducibility-paper framing — the thesis has explicitly moved to a dual-goal library + empirical-study framing, so wholesale adoption is not the goal; (b) the guide anchors heavily on SilverTorch as a co-system, which the Citation Policy forbids by name and replaces with the "author's own continuation of LinR" framing. Treat structural conventions (Pareto plot grammar, workload-characterization-first, per-algo parameter sweeps, selectivity binning, hardware disclosure, numerical-equivalence audit) as in-scope; treat venue-specific items (MLRC scope-of-reproducibility tables, ACM AE badges, $/1M-query economics, QPS@recall≥τ leaderboard metric) as optional, judged per chapter.

### Constraint: Citation Policy
The HSE supervisor restricts citations of any work where at least one author was affiliated with Meta (Facebook, FAIR, Reality Labs, Instagram, WhatsApp). This is a hard rule and the most load-bearing constraint in the rewrite.

**The thesis is positioned as a continuation of the LinR article (Borisyuk et al., LinkedIn — OK).** The "SilverTorch" algorithm is not referred to by name and is not cited; instead, it is presented as the author's own design built on classical primitives (IVF clustering, INT8 quantization, Bloom signatures) — see Citation Policy section below.

### Reader expectation
The thesis is software-engineering-heavy. The contribution is twofold:
1. An open-source PyTorch-native retrieval framework with a stable public API, full `torch.compile` interop (no graph breaks, cudagraph_trees-safe), backend parity (Triton ↔ torch), and an encoder-agnostic contract — any PyTorch module that emits query embeddings can be plugged in (demonstrated on SASRec/gSASRec for Goodreads & Yambda and on Nomic-Embed for arXiv, i.e. two encoder families across RecSys and text-IR).
2. A multi-dataset empirical study of the framework's reference implementations across quality / latency / memory / selectivity / batch-size / parameter axes.

The bar is API usability and benchmark breadth, not novel algorithmic theory.

### Status note (2026-05-25): downstream chapter drafts
Chapters 1, 3, 4 (=`03-methods.md`), 5 (=`05-implementation.md`), 6 (=`06-eval-protocol.md`) and 7 (=`07-results.md`) already exist as Markdown drafts under `docs/thesis/`. Those drafts were written under the **old "reproducibility-only" framing** and are **deliberately not updated as part of this plan revision** — the plan is the design contract, downstream agents reconcile their chapters against it on the next pass. Each affected chapter contract below carries a `[reframe pending]` marker on every framing change introduced by the dual-goal positioning, so a future cleanup pass can scope its edits precisely without re-reading the whole plan.

---

## Target chapter map

| # | Chapter (RU title)                                                    | Source-of-truth in repo                                     | Deliverable file                  |
|---|-----------------------------------------------------------------------|-------------------------------------------------------------|-----------------------------------|
| — | Front matter (title page, аннотация, abstract, ToC)                   | `thesis/main.tex` lines 168–240                             | edit `thesis/main.tex`            |
| — | Введение                                                              | this doc + lit review                                       | `docs/thesis/02-introduction.md`  |
| 1 | Обзор литературы                                                      | `docs/thesis/01-literature-review.md` (rewrite)             | `docs/thesis/01-literature-review.md` |
| 2 | Методы модельного GPU-retrieval                                       | `articles/linr.md`, `articles/silvertorch.md`               | `docs/thesis/03-methods.md`       |
| 3 | Датасеты и обучение query-моделей                                     | `evaluation/datasets/`, `evaluation/training/`              | `docs/thesis/04-datasets.md`      |
| 4 | Фреймворк `retrieve`: дизайн и реализация (goal 1)                    | `retrieve/src/retrieve/`, `docs/system/*`                   | `docs/thesis/05-implementation.md`|
| 5 | Протокол оценки                                                       | `evaluation/retrieval/`, `docs/system/evaluation.md`        | `docs/thesis/06-eval-protocol.md` |
| 6 | Эмпирическое сравнение реализаций (goal 2)                            | `evaluation/results/`, `SUMMARY.quality-deep.txt`           | `docs/thesis/07-results.md`       |
| 7 | Ограничения и направления развития                                    | `docs/plans/*`                                              | `docs/thesis/08-limitations.md`   |
| — | Заключение                                                            | synthesis                                                   | `docs/thesis/09-conclusion.md`    |
| — | Приложения (термины, сокращения, фрагменты кода, длинные таблицы)     | `thesis/main.tex` lines 372–390                             | edit `thesis/main.tex`            |

---

## Citation Policy (load-bearing)

### Authors / affiliations to AUDIT and remove from `docs/thesis/01-literature-review.md`
The current draft cites at least the following Meta-affiliated works that must be **removed**:

| Currently cited                                          | Why excluded                                  |
|----------------------------------------------------------|-----------------------------------------------|
| **SilverTorch** (`articles/silvertorch.md`)              | Meta paper                                    |
| Naumov et al. 2019 — DLRM                                | Meta authors                                  |
| Huang et al. 2020 — EBR in Facebook Search               | Meta authors                                  |
| Liu et al. 2021 — Que2Search                             | Meta authors                                  |
| Mudigere et al. 2022 — SW/HW Co-Design for DLRMs         | Meta authors                                  |
| Zhai et al. 2023 — Revisiting Neural Retrieval (MoL)     | Meta authors                                  |
| Zhai et al. 2024 — Actions Speak Louder than Words (HSTU)| Meta authors                                  |
| Ding & Zhai 2025 — Retrieval with Learned Similarities   | Meta authors                                  |
| Zhang et al. 2025 — Multi-Task Multi-Head I2I            | Meta authors                                  |
| Johnson, Douze, Jégou 2019/2021 — FAISS-GPU              | Douze, Jégou are at Meta/FAIR                 |
| Douze et al. 2024 — The Faiss Library                    | Meta authors                                  |
| Zhang et al. 2024 — Wukong                               | Meta authors                                  |
| Ardalani et al. 2022 — Scaling Laws for RecSys (arXiv:2208.08489) | Meta authors                         |
| Lewis et al. 2020 — RAG                                  | Meta authors                                  |
| Pal et al. 2020 — PinnerSage                             | check Pinterest authors only; if any Meta co-author, drop |
| Baltescu et al. 2022 — ItemSage                          | check Pinterest authors only; if any Meta co-author, drop |

The audit subagent **must verify each author list against author bios on arXiv/DBLP** before deciding to keep; assume Meta-affiliation if any author lists Facebook/FAIR/Meta in any affiliation on the paper (current or past on that paper).

### Replacements / acceptable substitutes

| Topic                          | Drop                          | Use instead                                                   |
|--------------------------------|-------------------------------|---------------------------------------------------------------|
| GPU ANN baseline               | FAISS / FAISS-GPU             | NVIDIA CAGRA (Ootomo 2024), Milvus (Wang 2021), SCANN (Guo 2020), SONG (Zhao 2020), HNSW-GPU (Manas Pinterest if no Meta authors) |
| Industrial RecSys context      | DLRM, EBR-Facebook            | YouTube DNN (Covington 2016, Google), TorchRec (Ivchenko 2022 — verify), TIGER (Rajput 2023, Google) |
| Learned similarities / MoL/HSTU| Zhai et al. series            | DSSM (Huang 2013, Microsoft), Two-tower (Yi 2019, Google), generative-retrieval line (TIGER, DSI) |
| Co-designed IVF+INT8+Bloom     | SilverTorch paper             | Cite component primitives: PQ/IVF (Jégou 2011), BitFunnel (Goodwin 2017, Microsoft), INT8 quantization general references, Bloom 1970. **Present as the author's own design.** |

### Reframing language for Ch.2 (Methods)
The "SilverTorch" implementation in this thesis must be described as **"co-designed inverted-file index with INT8-quantized scoring and fused Bloom-filter prefilter, developed as a continuation of the LinR research line."** Do not mention SilverTorch by name in the body text. The README, internal docs, and code-internal naming may keep the `SilverTorch` symbol (well-known shorthand in the repo) but explanatory prose should use neutral terms ("co-designed IVF retriever," "the IVF+INT8 module").

### Lineage / framing statement template (for §1.5 of lit review and §2 intro) [reframe pending in downstream drafts]
> The `retrieve` framework ships two reference retriever families. The LinR family (V1 post-filter masking, V2 pre-filter reduction, V3 1-bit Sign-OPORP) is paper-faithful to Borisyuk et al. (2024, CIKM). The composite IVF+INT8+Bloom retriever is the author's own design, assembled from classical primitives (Jégou 2011 IVF, INT8 quantization, BitFunnel-style bloom signatures (Goodwin 2017)) under the unified `FilterModule` contract introduced by the framework; it demonstrates that the framework extends naturally beyond the LinR line and accommodates retrievers with different indexing and filtering strategies.

---

## Per-chapter authoring contracts

### Введение [reframe pending in downstream drafts]
**Inputs:** this doc, completed Ch.6 summary (for "main results" paragraph).
**Outputs (~3–5 pages):**
- Актуальность: model-based GPU retrieval is becoming the dominant retrieval paradigm in industrial RecSys / IR; the PyTorch ecosystem lacks a `torch.compile`-friendly, encoder-agnostic library that bundles modern retrieval algorithms — practitioners assemble bespoke kernels per-project or reach for monolithic C++/CUDA libraries with weak Python interop and brittle integration with arbitrary PyTorch encoders. The closest precedent on the training side is Liger Kernel (Hsu et al. 2024, LinkedIn), which packages efficient Triton kernels for LLM training as drop-in PyTorch layers with benchmarks; no comparable artifact exists for the retrieval inference side. The `retrieve` framework targets that gap.
- Цель: построить открытый PyTorch-фреймворк `retrieve`, реализующий эффективные GPU-слои retrieval (семейство LinR + композитный IVF+INT8+Bloom), совместимый с `torch.compile` и произвольными PyTorch-энкодерами, и провести комплексную эмпирическую оценку его реализаций на трёх открытых датасетах.
- Задачи (must explicitly cover both goals; 6–8 bullets):
  - **Goal 1 — framework:**
    - Реализовать набор retrieval-слоёв в Triton и эталонных torch-вариантах с парностью результатов.
    - Обеспечить совместимость с `torch.compile` (no graph breaks at module boundaries, cudagraph_trees-safe, `mode="reduce-overhead"`).
    - Определить единый контракт `FilterModule` для фильтров (clause + bloom + композиции) как точку расширения.
    - Декуплить API от выбора query-энкодера (любой `nn.Module`, выдающий тензор эмбеддингов, подходит).
  - **Goal 2 — evaluation:**
    - Построить sweep-харнесс с метриками качества и латентности.
    - Обучить SASRec/gSASRec на двух RecSys-датасетах (Goodreads, Yambda) и переиспользовать готовые Nomic-Embed эмбеддинги на одном text-IR датасете (arXiv) — демонстрация encoder interop.
    - Замерить качество (Recall/NDCG/MRR@K), латентность, память, влияние селективности фильтра, чувствительность к параметрам, парность Triton ↔ torch.
- Объект / предмет (HSE convention): объект — GPU retrieval в задачах информационного поиска и рекомендательных систем; предмет — PyTorch-нативная реализация retrieval-слоёв и их эмпирическое сравнение.
- Methods: experimental ML / GPU programming.
- Empirical base: Goodreads, arXiv, Yambda 500m/5b — открытые датасеты, покрывающие RecSys (Goodreads, Yambda) и text-IR (arXiv) сценарии, две семьи энкодеров (SASRec/gSASRec и Nomic-Embed) и два режима фильтрации (clause + bloom).
- Brief result statement (filled after Ch.6 done).
- Structure of the work (1 paragraph).
**Constraint:** the lineage / framing statement (see Citation Policy) must appear here too.

### Chapter 1 — Обзор литературы (rewrite) [reframe pending in downstream drafts]
**Source:** `docs/thesis/01-literature-review.md` (current draft).
**Authoring agent task:**
1. Audit every citation against the Citation Policy table; remove all Meta-affiliated works.
2. Preserve subsection structure 1.1–1.11 (it's well-designed).
3. Substitute citations from the Replacements table.
4. **Rewrite §1.5 (Model-based retrieval)** so it anchors on LinR (Borisyuk et al. 2024 CIKM, LinkedIn) as the framework's primary reference family and presents the composite IVF+INT8+Bloom retriever as a second bundled implementation built from classical primitives (see Reframing language).
5. **Rewrite §1.11 (Positioning)** to drop direct SilverTorch reference and to reflect the dual-goal framing. Replacement: "LinR is closed-source CUDA C++ with no open Triton implementation, and the PyTorch ecosystem lacks a `torch.compile`-friendly retrieval library that decouples retrieval-layer choice from encoder choice. The `retrieve` framework fills both gaps: it provides paper-faithful LinR implementations plus a composite IVF+INT8+Bloom retriever as a second reference family, behind a single public API that consumes embeddings from arbitrary PyTorch encoders. The comprehensive multi-dataset evaluation in Ch.6 is the second pillar of the work."
6. **NEW (framework framing):** add or repurpose one subsection (suggest §1.10 — currently lightly used) on **PyTorch layer libraries, retrieval libraries, and benchmark conventions**. Anchor it on the Liger Kernel precedent (Hsu et al. 2024, LinkedIn — safe under citation policy): an open Triton-kernel layer library for LLM training, shipped with benchmarks, no algorithmic novelty claim. State explicitly that the `retrieve` framework adopts the same product shape on the retrieval / inference side. Also cover `torchrec` (verify no Meta authors), pytorch-metric-learning, CUDA-native ANN libraries used as conceptual comparators (CAGRA, Milvus, SCANN), and IR/RecSys benchmark conventions (BEIR — verify, BIG-ANN). Conclude with the gap the `retrieve` framework targets (compile-friendly, encoder-agnostic, multi-algorithm, Triton + torch parity, benchmarks first-class).
7. Expand each subsection from skeleton to ~3–5 pages of Russian prose.
8. Build `thesis/references.bib` from the final, audited bibliography (BibLaTeX/GOST-numeric).

**Deliverable acceptance criteria:**
- Zero Meta-affiliated authors in the final bib (verify via grep on author names).
- Every cited work has DOI or arXiv ID in the `.bib`.
- §1.5 prose explicitly contains the lineage statement.

### Chapter 2 — Методы
**Sources:**
- `articles/linr.md` — LinR paper notes.
- `articles/silvertorch.md` — keep as internal reference only (do not cite); use it to derive the formal description that will be presented as author's design.
- `retrieve/src/retrieve/layers/*` for algorithm specifics (especially file headers/docstrings).

**Sections:**
- 2.1 Постановка задачи retrieval: top-K по dot-product/cosine, фильтрация по атрибутам, формализация.
- 2.2 Семейство LinR:
  - 2.2.1 V1 — FullScan / Postfilter-mask (dense fp16 и INT8 варианты)
  - 2.2.2 V2 — Prefilter (sparse + reduced matmul)
  - 2.2.3 V3 — Sign-OPORP 1-bit + Hamming popcount
  - For each: pseudocode (LaTeX `algorithm` env), complexity (time & memory), preconditions on filter selectivity.
- 2.3 Co-designed IVF + INT8 + Bloom (author's continuation):
  - 2.3.1 IVF clustering и probe (k-means++ init, n_probe parameter)
  - 2.3.2 INT8 global-scale dot-product (motivation, error bound, dp4a hardware fit)
  - 2.3.3 Bloom-signature subset test (forward index, GPU-friendly bitwise)
  - 2.3.4 Fused probe→filter→score kernel (three-phase pipeline)
- 2.4 Filter primitives:
  - 2.4.1 ExactAttributeFilter (clause matching, AND-of-OR, reverse clauses)
  - 2.4.2 BloomFilter
  - 2.4.3 Composition: `combine_masks` (dense AND) vs `combine_indices` (sparse cascade)
- 2.5 Квантизационные схемы:
  - 2.5.1 INT8 per-tensor symmetric (formal definition, scale derivation)
  - 2.5.2 Sign-OPORP 1-bit (one permutation + one random projection, seed-based determinism)

**Visual deliverables:**
- 1 figure per algorithm family — block diagrams of the data flow (e.g., LinR V3: items → 1-bit codes → popcount-xor → top-K).
- 1 algorithm box per algorithm (Russian-language pseudocode).
- Memory/complexity table comparing all variants.

### Chapter 3 — Датасеты и обучение query-моделей
**Sources:**
- `evaluation/datasets/{goodreads,arxiv,yambda,synth_arxiv}.py`
- `evaluation/training/{train_sasrec,evaluate,losses,model,config}.py`
- `articles/yambda.md` for Yambda paper context
- `docs/system/checkpoints.md` for checkpoint conventions

**Sections:**
- 3.1 Goodreads (UCSD Book Graph):
  - Source (Wan & McAuley 2018, RecSys; Wan et al. 2019 ACL — both OK to cite)
  - Scale: ~313k users, item counts (run actual counts from data dir)
  - Split: train/val/test by chronology (verify against `goodreads.py`)
  - Attributes: narrow vs wide filter sets, structure of `item_attrs_*.pt`
  - Pipeline: download → convert → prep (CLI subcommands)
- 3.2 arXiv:
  - Source: HuggingFace `open-index/open-arxiv`, ~2.99M papers
  - No user sequences → text retrieval setup
  - Pre-encoding: nomic-embed-text-v1.5 @ 256-d (cite Nussbaum 2024 — OK)
  - Query encoding: separate "search_query:" prefix pass
  - Attribute construction: how narrow/wide filter sets are derived
- 3.3 Yambda (Listen+):
  - Source: Yandex Music dataset (cite per `articles/yambda.md`)
  - Two scales: 500m (full) and 5b (subset) — verify which is which
  - Interaction type = "listens"
  - Global Temporal Split protocol
  - No filter benchmark on Yambda (quality only)
- 3.4 Обучение query-моделей (SASRec/gSASRec):
  - Архитектура: SASRec (Kang & McAuley 2018 ICDM — OK) с gBCE-лоссом (Petrov & Macdonald 2023, RecSys Best Paper — OK).
  - Конфигурация: `evaluation/training/config.py` (dataclass dump).
  - Гиперпараметры: d ∈ {64, 128, 256}, dropout=0.5, max_seq_length=200.
  - Loss: GBCE (`evaluation/training/losses.py`) — формальное определение, мотивация против overconfidence.
  - Training loop: epochs, optimizer, scheduler, eval cadence.
  - Per-epoch metrics: NDCG@{10,100}, Recall@{10,100} (`evaluation/training/evaluate.py:57-100`).
  - **Table:** final SASRec quality per dataset × dimension (Recall@100, NDCG@100, training time, peak GPU mem).
  - **Plot:** training curves (val Recall@100 vs epoch) for at least one dataset × dim.
  - Checkpoint locations: `data/<dataset>/checkpoints/gsasrec-d<N>-drop0.5-id/best_model.pt`.

**Visual deliverables:**
- 1 table per dataset: rows, items, attributes, sparsity, embedding dim (final, after preprocessing).
- 1 consolidated SASRec quality table.
- 1 training-curve plot (or grid of plots).
- 1 schematic figure: dataset pipeline (raw → preprocessed → SASRec encoder → query embeddings).

### Chapter 4 — Фреймворк `retrieve`: дизайн и реализация (goal 1) [reframe pending in downstream drafts]
**Framing.** This chapter delivers goal 1. It must read as the description of a reusable library — Liger-Kernel-shaped (Hsu et al. 2024, LinkedIn): drop-in PyTorch layers, Triton-optimized kernels under the hood, benchmarks alongside — not a reproduction artifact. Three properties carry the framework claim and must be explicit, with evidence: (a) `torch.compile` interop (no graph breaks at module boundaries, cudagraph_trees-safe), (b) encoder-agnostic public API (consumes embeddings, agnostic to how they are produced — demonstrated by SASRec and Nomic-Embed encoders feeding the same layers in evaluation), (c) extension points (`FilterModule`, `Backend`, swappable kernels). Reproducibility of LinR is acknowledged as a quality bar; it is not the section's headline. The chapter's deliverable list ends with per-layer micro-benchmarks (Triton-vs-torch speedup, compile-vs-eager speedup) shipped in the same spirit as Liger's benchmarks directory.

**Sources:**
- `retrieve/README.md`, `retrieve/src/retrieve/__init__.py` (public API).
- `retrieve/src/retrieve/layers/` (modules).
- `retrieve/src/retrieve/kernels/` (Triton kernels).
- `retrieve/src/retrieve/interfaces.py` (`Backend`, `FilterModule`).
- `retrieve/src/retrieve/tune.py` (offline autotuning).
- `docs/system/architecture.md`, `docs/system/kernels.md`, `docs/system/filtering.md`, `docs/system/testing.md`.

**Sections:**
- 4.0 **Фреймворк как библиотека** (new, ~1 page): contract сourface — what does a user import, what assumptions does the framework make about inputs (`queries: Tensor`, `items: Tensor | indexed`), what does it return (top-K indices + scores), what extension points are exposed (`Backend` enum, `FilterModule` ABC, kernel autotuning hooks). Cite `retrieve/__init__.py` exports as the canonical surface area.
- 4.1 Архитектурный обзор: workspace layout (uv workspace; `retrieve/` published as `torchretrieve`; `evaluation/` not published); `layers/` vs `kernels/` vs `interfaces.py` separation.
- 4.2 Публичный API:
  - **Table:** 6 KNN modules (FullScan, OneBitKNN, Postfilter, PostfilterInt8, Prefilter, IVF+INT8+Bloom — i.e., SilverTorch class) + 2 filters + helpers, with one-line descriptions.
  - Code listing: minimal end-to-end example (from `retrieve/README.md`).
- 4.3 Triton-ядра:
  - 4.3.1 LinR kernels: `fused_masked_knn_topk`, `oporp_1bit_match_topk_full`/`indirect`.
  - 4.3.2 Co-designed kernels: `codesigned_probe_score`, `codesigned_probe_score_bloom`, `codesigned_probe_score_exact`, `bloom_match`.
  - 4.3.3 Filter kernels: `clause_mask`, `clause_compact`, `bloom_compact`.
  - For each: launch grid, tile shapes (block_n, num_warps, num_stages from `DEFAULT_CONFIG`), memory access pattern, what it produces.
- 4.4 Бэкенд-абстракция: `Backend = Literal["torch", "triton"]`, инвариант paritет результатов (см. testing).
- 4.5 `torch.compile` поддержка:
  - `@torch.library.custom_op` (compact kernels) vs `@torch.library.triton_op` (scoring kernels) — motivation.
  - No graph breaks at module boundaries.
  - cudagraph_trees compatibility, `mode="reduce-overhead"`.
  - **Table:** speedup of compiled vs eager for each module (run from `retrieve/tests/compile/` benchmarks if present, otherwise from results JSONs).
- 4.6 Оффлайн autotuning:
  - Why no `@triton.autotune` (cudagraph leakage, corruption of compact kernels).
  - CLI: `uv run tune-kernels <kernel-subcommand>`.
  - `DEFAULT_CONFIG` dataclass per kernel.
- 4.7 Filter composition: `FilterModule` интерфейс; sparse vs dense композиция; cost model (when AND vs cascade).
- 4.8 Quantization primitives: `quantize_int8`, `quantize_int8_global`, `quantize_oporp_1bit` (code refs, formal definitions are in Ch.2).
- 4.9 Тестирование:
  - Parity tests: Triton vs torch backends.
  - Compile parity: eager vs `torch.compile(dynamic=True)`.
  - Test layout: `retrieve/tests/{layers,kernels,compile}/`.
- 4.10 Объёмные характеристики (HSE-style table):
  - LOC per module (run `cloc` or `wc -l`).
  - # public classes/functions.
  - # tests.
  - # Triton kernels.
- 4.11 **Encoder-agnostic interop** (new, ~1 page): the framework consumes `(queries, items)` tensors regardless of how they were produced. Two concrete demonstrations from the evaluation: (a) SASRec/gSASRec sequence model encoder for Goodreads and Yambda, (b) Nomic-Embed text encoder (out-of-the-box pretrained, not retrained) for arXiv. Show that the same `KNN` / `IVF+INT8+Bloom` layers are used unchanged across both. Cross-link to Ch.3 for encoder details.

**Visual deliverables:**
- 1 architecture diagram: `retrieve/` module tree with arrows from `layers/` → `kernels/`.
- 1 sequence diagram for the SilverTorch (co-designed IVF) forward pass: input batch → IVF probe → filter phase → INT8 score → top-K.
- Algorithm boxes for kernel pseudocode (selected: the fused codesigned kernel is mandatory).
- Volumetric table (4.10).
- Compile-speedup table (4.5).

### Chapter 5 — Протокол оценки
**Sources:**
- `evaluation/retrieval/{metrics,sweep,queries_cache,oracle,results_io,config}.py`
- `evaluation/retrieval/cli/run_evaluation.py`
- `evaluation/config/*.yaml`
- `docs/system/evaluation.md`

**Sections:**
- 5.1 Архитектура харнесса:
  - CLI: `run_evaluation.py` (suites: filter, quality, deep_sweeps, yambda).
  - Cell = (dataset, dimension, filter_kind, sweep).
  - Per-algo runner orchestration.
- 5.2 Метрики (formal):
  - Recall@K, Precision@K, MRR@K, NDCG@K — формулы.
  - Per-query accumulation; finalization as mean.
  - Implementation references: `metrics.py:39–122`, batch: `:125–156`.
- 5.3 Query generation:
  - SASRec encoding pass → `encoded_queries_<split>.pt` cache.
  - Cache key: `(ckpt_mtime, max_seq_length, users_limit)` (see `queries_cache.py`).
  - Disk-budget guard (recent commit `2ea3dae`).
- 5.4 Candidate sets and ground truth:
  - Yambda: full-scan ground truth (no filter benchmark there).
  - Goodreads/arXiv: filtered (clause/bloom) — narrow vs wide vs no-filter.
  - Oracle module (`oracle.py`) for cache-efficient sweeping.
- 5.5 Sweep dimensions:
  - K ∈ {100, 200, 400} (full sweep) or {100, 500} (defaults).
  - Batch sizes ∈ {1, 8, 16}.
  - Backends: torch, triton.
  - Seeds (replication count — pull from configs).
  - Algorithm lineup: linr_v1, linr_v1_filter_mask, linr_v3, linr_v4, silvertorch (and SilverTorch+exact / SilverTorch+bloom variants).
- 5.6 Окружение и hardware:
  - GPU model, driver/CUDA version, PyTorch + Triton version (pull from a known machine spec or have user provide).
  - Determinism settings.
- 5.7 Result schema (table):
  - Fields: `suite, cell, filter_kind, sweep, impl, backend, device, seed, batch_size, k, n_users_kept, median_ms, p20_ms, p80_ms, peak_mem_mib, index_mem_mib, fwd_scratch_mib, recall@K, ndcg@K, extra.params`.
  - Distinguish quality fields from latency fields.
- 5.8 Layout результатов:
  - `.perkernel/` directory convention.
  - `evaluation/results/{arxiv,goodreads,yambda}/**/*.json`.
  - `upload_results.py` classifier into filter/quality/yambda/deep_sweeps.
  - Runlog: `SUMMARY.quality-deep.txt`.

**Visual deliverables:**
- 1 flow diagram: cell → algorithms → sweep → metrics → JSON.
- 1 table of sweep dimensions and value ranges.
- 1 schema table (5.7).

### Chapter 6 — Эмпирическое сравнение реализаций (goal 2) [reframe pending in downstream drafts]
**Framing.** This chapter delivers goal 2. It must read as a comprehensive comparative empirical study of the framework's reference implementations, not as a reproduction check against the LinR paper. The breadth claim is concrete: three datasets × multiple embedding dims × two backends × the full algorithm lineup × six sweep axes (K, batch size, filter selectivity, n_lists/n_probe, candidate_pool, seed). The §6.8.3 (G10) mirror of LinR Tables 3/4 stays as one piece of evidence among many, not as the chapter's thesis.

**Sources:**
- `evaluation/results/{arxiv,goodreads,yambda}/**/*.json` (parse via `results_io.load_rows`).
- `evaluation/results/_runlogs/SUMMARY.quality-deep.txt`.
- Per-kernel breakdowns in `.perkernel/` directories.

**Sections (9-part structure mapping to the figure catalog A1–G10; full per-figure specs in [docs/thesis/results-data/recipes/plot_catalog.md](results-data/recipes/plot_catalog.md), prose-ready stubs in [07-results.md](07-results.md)):**

- **6.1 Workload characterization** (~1 page)
  - 6.1.1 = D3 — Per-clause selectivity distribution (bars per dataset).
  - 6.1.2 = E3 — Per-query selectivity CDF per dataset.
  - 6.1.3 = G8 — SASRec quality ceiling (`sasrec_quality_ceiling.csv`).
  - *Rationale:* characterize the workload before showing system numbers (ACORN/Filtered-DiskANN convention).
- **6.2 Quality–latency tradeoffs (per dataset × dim)** (~3–4 pages)
  - 6.2.1 Goodreads / 6.2.2 arXiv / 6.2.3 Yambda full numeric tables.
  - 6.2.4 = Recall at K=100/200/400 (all cells).
  - 6.2.5 Cross-dataset summary.
  - 6.2.6 = A1 — Recall–latency Pareto (bs=1), faceted 3×3.
  - 6.2.7 = A2 — Recall–latency Pareto (bs=16), faceted 3×3.
  - 6.2.8 = G1 — Headline summary table.
  - 6.2.9 = G2 — Latency-at-fixed-recall table.
  - 6.2.10 = G3 — Recall-at-fixed-latency table.
- **6.3 Quality–memory tradeoffs** (~2 pages)
  - 6.3.1 = A3 — Recall–memory Pareto.
  - 6.3.2 = A4 — Recall × latency × memory bubble (executive summary visual).
  - 6.3.3 = G6 — Memory breakdown table (extend `d128_quality_memory.csv` to all dims).
  - *Anchor:* LinR's 16× memory-reduction claim — confirm or refine.
- **6.4 Filter behavior (Goodreads + arXiv)** (~3 pages)
  - 6.4.1 Filter taxonomy (narrow / wide / bloom; reverse-clause caveat).
  - 6.4.2 / 6.4.3 / 6.4.4 Per-cell tables for goodreads-d128 / arxiv-d128 + d64/d256 sub-tables.
  - 6.4.5 = D2 — Recall vs selectivity curves (twin-panel; direct LiNR Fig 7 template).
  - 6.4.6 = D1 — Recall–latency Pareto stratified by selectivity bucket.
  - 6.4.7 = G7 — Cross-over points table.
- **6.5 Parameter sensitivity (deep sweeps)** (~2–3 pages)
  - 6.5.1 = B1 — Silvertorch (n_lists × n_probe) heatmap with iso-contours.
  - 6.5.2 = B2 — LinR V3 candidate_pool curves + bit-budget axis (planned).
  - 6.5.3 = B3 — K sensitivity curve (planned, blocked on K=10 re-run).
  - 6.5.4 = B4 — Batch-size scaling (planned, blocked on extended bs sweep).
  - 6.5.5 Pareto interpretation across both deep sweeps.
- **6.6 Scaling** (~1–2 pages)
  - 6.6.1 = C1 — Dataset N scaling (Yambda 500m vs 5b).
  - 6.6.2 = C2 — Dimensionality scaling (d64 / d128 / d256).
- **6.7 Engineering validation** (~2 pages)
  - 6.7.1 = F1 — Backend parity scatter (with Jaccard).
  - 6.7.2 = F2 — Backend speedup distribution (faceted).
  - 6.7.3 = G4 — Backend parity table.
  - 6.7.4 = G5 — Speedup matrix (Triton/torch).
  - 6.7.5 = F3 — Kernel-level breakdown (planned, Nsight Compute).
  - 6.7.6 = F4 — Build / index-construction time (planned, schema ext).
- **6.8 Reproducibility** (~1–2 pages)
  - 6.8.1 = E2 — Seed stability bar chart (planned, multi-seed re-run).
  - 6.8.2 = G9 — Reproducibility / hardware disclosure (GPU/driver/CUDA/torch/triton/seed).
  - 6.8.3 = G10 — Mirror LiNR paper Tables 3/4 (strongest reproducibility evidence).
- **6.9 Executive summary** (~1 page)
  - 6.9.1 = A4 (reprise) — same bubble chart, different framing.
  - One headline paragraph stating the three Pareto dominance regions.

**Throughput / QPS metric convention.** Chapter 6 uses `median_ms` (per-batch latency) on X for all Pareto plots — the BIG-ANN 2023 ranking rule ("highest QPS at ≥90% recall@10") is referenced descriptively but **not adopted as the primary metric** until a direct `throughput_qps` measurement is added (see "Planned schema extensions" in §5.7 of [06-eval-protocol.md](06-eval-protocol.md)). We use latency-at-fixed-recall (G2) as our analogue and acknowledge the substitution in chapter prose.

**Visual deliverables (mandatory per section).** Full catalog at [docs/thesis/results-data/recipes/plot_catalog.md](results-data/recipes/plot_catalog.md); cross-listed in [06-eval-protocol.md](06-eval-protocol.md) (Chapter-6 visual deliverables section). The mandatory minimum per section:
- §6.1 — D3, E3, G8.
- §6.2 — A1, A2, G1.
- §6.3 — A3, A4, G6.
- §6.4 — D2, D1, G7.
- §6.5 — B1, B2.
- §6.6 — C1, C2.
- §6.7 — F1, F2, G4, G5.
- §6.8 — G9, G10.
- §6.9 — A4 (reprise).

Plots/tables marked "planned" in the catalog (B3, B4, E1, E2, F3, F4, G2) depend on the "Planned schema extensions" in §5.7 of `06-eval-protocol.md` and the "Stage 4 — Results expansion" milestone in `docs/plans/00-roadmap.md`. They are not blockers for §6.2 / §6.4 / §6.5 prose drafting.

**Data-extraction agent task before writing prose:**
1. Walk `evaluation/results/**/*.json`, use `results_io.load_rows`.
2. Aggregate into a Pandas DataFrame keyed by (suite, cell, impl, backend, k, batch_size, seed).
3. Compute medians/p20/p80 across seeds; ensure each (cell, impl, backend) has all expected k×batch_size combinations.
4. **Add derived columns**: `bytes_per_vec = index_mem_mib * 2^20 / n_items`, `ratio_vs_fp32 = bytes_per_vec / (4*d)` (unblocks A3, A4, G1, G6).
5. Emit a `docs/thesis/results-data/` directory of CSVs + matplotlib-rendered PNGs. (One script per figure family — see catalog for naming convention.)
6. **Do not regenerate results data** — the harness output JSONs are authoritative.

### Chapter 7 — Ограничения и направления развития
**Sources:**
- `docs/plans/00-roadmap.md`, `live-update-api.md`, `torch-export-refactor.md`, `yambda-hf-migration.md`, `03-kernel-optimizations.md`, `evaluation-retrieval-cleanup.md`.

**Sections:**
- 7.1 Single-GPU only — no multi-GPU index sharding.
- 7.2 Offline indexing only — no live updates (cite `live-update-api.md` as planned).
- 7.3 Inference only — no joint learning of similarity functions.
- 7.4 Three datasets are RecSys-leaning + text-retrieval (arXiv); no web-scale text retrieval comparison.
- 7.5 No comparison with CUDA-native GPU ANN libraries on identical hardware — Triton ceiling vs hand-tuned CUDA is open.
- 7.6 INT8 quantization assumes embedding distribution properties — formal error bound is empirical.
- 7.7 Filter API limited to bloom + exact-attribute clauses — no learned filters, no range predicates.
- 7.8 Roadmap from `docs/plans/`: kernel optimizations, torch.export refactor, Yambda HF migration.

### Заключение [reframe pending in downstream drafts]
**Inputs:** completed Ch.6 + Ch.7.
**Outputs (1–2 pages):**
- Restate the two goals 1:1 with Введение and tick off both:
  - **Goal 1 — framework:** `retrieve` shipped as `torchretrieve` on PyPI; LinR family + composite IVF+INT8+Bloom; full `torch.compile` interop with backend parity; encoder-agnostic API demonstrated on SASRec and Nomic-Embed.
  - **Goal 2 — evaluation:** N (insert number) sweep cells across three datasets × dims × backends × algorithms × six sweep axes; Pareto fronts characterized; per-regime recommendations stated.
- Quantitative highlight: best Recall@100 on each dataset, biggest speedup result.
- Open-source release statement (PyPI `torchretrieve`).
- Outlook tied to Ch.7.

### Front matter (edit `thesis/main.tex` directly) [reframe pending in downstream drafts]
- Lines 182: replace title — proposed (dual-goal framing): **"`retrieve`: PyTorch-фреймворк эффективных GPU-слоёв retrieval и комплексное эмпирическое сравнение его алгоритмов на открытых датасетах"**. Alternative (more academic): "Открытый PyTorch-фреймворк retrieval-слоёв с поддержкой `torch.compile` и эмпирическое сравнение его реализаций на трёх RecSys/IR датасетах". Old reproducibility-only proposal ("Воспроизводимая реализация и эмпирическое сравнение алгоритмов модельного GPU-retrieval на открытых датасетах с применением Triton") is retained here only as a fallback.
- Lines 187: replace student name — **confirm with user before editing**.
- Lines 189–191: replace supervisor info — **confirm with user**.
- Lines 205–214: write Russian аннотация (1500–2000 chars). Template: "Объектом исследования являются алгоритмы модельного GPU-retrieval... Предметом — их воспроизводимость на открытых датасетах с использованием Triton..."
- Lines 220–231: English abstract, mirroring Russian (1500–2000 chars).
- Add `\input{chapters/01-...}` lines after the ToC, one per chapter Markdown converted to LaTeX (pandoc or hand-translated).

### Приложения
- A — Терминологический словарь: define IVF, INT8, OPORP, Bloom, clause, narrow/wide filter, codesigned kernel, top-K, Recall@K, etc.
- B — Список сокращений: keep existing (ВКР, НИУ ВШЭ) + add ANN, IVF, INT8, OPORP, KNN, BCE, GBCE, NDCG, MRR.
- C — Ключевые фрагменты исходного кода: pick 3 short snippets — the fused codesigned kernel, the OPORP popcount kernel, and one composition example (`combine_indices`).
- D — Длинные таблицы результатов: full per-cell quality/latency tables in `longtable` env.

---

## Workflow for downstream agents

Order with parallelization opportunities marked:

**Phase A — Foundation (parallel, 2 agents):**
- Agent A1: Rewrite Ch.1 (literature review) — citation audit + prose expansion.
- Agent A2: Write Ch.2 (Methods) — formal description of algorithms.

**Phase B — Body (parallel, 2 agents, depends on Phase A):**
- Agent B1: Write Ch.3 (Datasets + SASRec training).
- Agent B2: Write Ch.4 (Implementation).

**Phase C — Evaluation (sequential, depends on B):**
- Agent C1: Write Ch.5 (Evaluation Protocol) — depends on B2 (impl references).
- Agent C2: Data-extraction pass for Ch.6 — parse all `evaluation/results/*.json`, produce CSVs and plots in `docs/thesis/results-data/`.
- Agent C3: Write Ch.6 (Results) prose around C2's artifacts — depends on C1 (protocol terminology) and C2 (data).

**Phase D — Closing (parallel, 2 agents, depends on C):**
- Agent D1: Write Ch.7 (Limitations).
- Agent D2: Write Введение + Заключение.

**Phase E — Integration (single agent):**
- Fix `thesis/main.tex` front matter (title, author — needs user confirmation before edit).
- Convert each `docs/thesis/0X-*.md` chapter to LaTeX (pandoc-based) into `thesis/chapters/`.
- Wire up `\input{}` in `main.tex`.
- Build `references.bib` from the audited Ch.1 citations.
- Compile with the documented xelatex/biber/xelatex/xelatex sequence.
- Verify: no Meta-affiliated names in `references.bib` (grep audit).

---

## Verification

End-to-end checks before declaring the thesis ready:

1. **Citation audit:** `grep -i -E "(meta|facebook|fair|instagram|whatsapp)" thesis/references.bib` should match zero author-affiliation occurrences. Manually verify all author lists against arXiv/DBLP for the final bibliography.
2. **Compile:** run `cd thesis && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex` — must produce `main.pdf` without errors and with all `\cite{}` resolved.
3. **Data integrity in Ch.6:** every plot/table must trace to a CSV in `docs/thesis/results-data/`; CSVs must be reproducible from `evaluation/results/*.json` via the agreed extraction script. Commit both the extraction script and the CSVs.
4. **Cross-references:** every `\ref{}` in main.tex resolves; ToC entries are populated; figure and table numbering is by chapter (per the existing template).
5. **Word/char counts:** Russian аннотация and English abstract both in 1500–2000 chars (the template enforces this).
6. **Code parity claim:** state in Ch.5 that Triton and torch backends produce identical top-K (within float tolerance). Cite the parity test file.

---

## Critical files

| File / dir                                                | Role                                  |
|-----------------------------------------------------------|---------------------------------------|
| `thesis/main.tex`                                         | LaTeX root — front matter + `\input{}`s |
| `thesis/references.bib`                                   | TO CREATE — audited GOST bib          |
| `thesis/chapters/`                                        | TO CREATE — per-chapter `.tex`        |
| `docs/thesis/00-thesis-plan.md`                           | TO CREATE — published copy of this doc|
| `docs/thesis/01-literature-review.md`                     | REWRITE — citation audit + prose      |
| `docs/thesis/02..09-*.md`                                 | TO CREATE — per-chapter Markdown drafts |
| `docs/thesis/results-data/`                               | TO CREATE — CSVs + PNGs for Ch.6      |
| `docs/system/{architecture,kernels,filtering,testing,evaluation,checkpoints}.md` | READ — source for Ch.4, Ch.5 |
| `articles/{linr,silvertorch,yambda}.md`                   | READ — source for Ch.2 (silvertorch.md NOT cited) |
| `retrieve/src/retrieve/`                                  | READ — source for Ch.4                |
| `evaluation/{datasets,training,retrieval}/`               | READ — source for Ch.3, Ch.5, Ch.6    |
| `evaluation/results/`                                     | READ (authoritative) — Ch.6 data      |
| `evaluation/results/_runlogs/SUMMARY.quality-deep.txt`    | READ — Ch.6 run inventory             |
| `docs/plans/`                                             | READ — Ch.7 roadmap                   |
