# Thesis — Consolidated remaining-work plan

Compiled 2026-05-25 from `00-thesis-plan.md` + per-chapter TODO sweep of `01-` through `09-*.md` + `results-data/recipes/plot_catalog.md` + `docs/plans/00-roadmap.md`.

Status legend used by the plot catalog (preserved here for the §3 plot section):
- ✅ derivable — data exists, only needs the plotting script
- 🟡 upgrade — initial PNG exists, needs spec upgrade
- 🆕 new — no prior render, but source CSV is ready
- ⏳ partial — data partially present, one column or dataset missing
- 🚧 blocked — depends on a Stage 4 schema extension or new sweep
- ⚠ publication blocker — depends on the stale-oracle rerun (see §0)

---

## Decisions log (resolved with author)

Decisions are listed in resolution order. Each entry references the clarification number in §1; items in §1 are crossed out as they close.

- **D1 — PyTorch citation** (#1): cite **GitHub repository URL only** (footnote / URL in bibliography); no Paszke et al. paper, no Ansel et al. paper, nothing Meta-adjacent.
- **D2 — LinR V4 in Введение** (#2): frame V4 as **"one of the methods additionally introduced in the framework, an extension of a method proposed in LinR"** — i.e., V4 is in the lineage statement as an author-side extension, not deferred to Ch.2.
- **D3 — Lineage statement format** (#3): **Russian paraphrase only** — no verbatim English blockquote. The thesis is compact; no oversized citation displays.
- **D4 — Введение headlines** (#4): keep **Pareto-across-datasets** + **silvertorch 10.94× max speedup** in Введение. V4 memory reduction + V3 candidate_pool recovery curve move to Ch.6 only. V4 is "just quantization" — minimal emphasis in Введение.
- **D5 — Заключение scope** (#5, #6, #7, #8, Q10 follow-up): Заключение is **delivered-work-only**. Sections: (a) restate Цель, (b) tick off Задачи with shipped artifacts, (c) the same 1–2 quantitative highlights as Введение. **No future-work outlook**, no PyPI/release statement, no commit hash, no 7th Задача mention. Multi-GPU sharding + live-update API stay in Ch.7 Limitations only — they are framework-extension directions, not Заключение material.
- **D6 — Citation Policy strict drop** (#9–#19): if any author has Meta/Facebook/FAIR affiliation (current or on the paper), **drop the citation**. Where a clean substitute is natural, propose it (e.g., HNSW → Malkov & Yashunin 2018 only); where no clean substitute exists, drop entirely with no replacement. **No citation > weak citation that risks the audit.**
- **D7 — Silvertorch reverse-clause** (§0 blocker 2, also #41-adjacent): **fix it**. Silvertorch supports reverse clauses only via the **exact-index path** (clause-table lookup), not the Bloom path. Required work: (a) wire `clause_is_reverse` through `SilvertorchAlgo.__init__` → `SilverTorch.register_index` → codesigned exact-clause kernel; (b) GPU-rerun Goodreads d64/d128/d256 filter sweeps; (c) document the **Bloom-path reverse-clause non-support** as a current limitation in §7.7 (not the wiring gap — the algorithmic constraint).
- **D8 — yambda-5b-d256** (#27): **won't be done** — insufficient compute to train and converge a SASRec checkpoint at that scale. Footnote intentional omission in §6.2.3 / §6.6.1 / G8 / C1. Drop from GPU rerun list.
- **D9 — Заключение scope confirmed** (Q10 = yes): D5 stands. Three components only: Цель restated · Задачи tick-off · 1–2 quantitative highlights identical to Введение.
- **D10 — Ch.4 implementation decisions** (Q11):
  - (a) `tune.py` listed in §4.2 (public API).
  - (b) README example **dropped from §4.2** — README itself is English, no need to mirror it in the Russian thesis.
  - (c) `torch.export` mention in §4.5 **skipped** (defer to §7 Limitations / roadmap).
  - (d) `tune-kernels` actual-run transcript in §4.6 **skipped**.
  - (e) Test counts and test-case discussion **dropped from §4.9 entirely** — no test mentions in the thesis at all. (Parity claims still cite test files in §4.4 / §4.9; just no "162 tests" headline.)
- **D11 — Ch.6 protocol callouts** (Q12):
  - (a) Pareto plot torch markers: **deferred** ("no idea now"). Plot-time decision; revisit when rendering A1/A2.
  - (b) bs > 1 scratch profile: **extract** from filter suite for §6.3 completeness.
  - (c) Yambda corpus sizes: **use measured 3.06M / 5.37M**, and add a short paragraph in §6.0 / §6.6.1 explaining the provenance — which dataset slices were kept, which action type was filtered (listens only), how the measured "retrieval corpus" was derived from the raw 9.39M catalogue.
  - (d) LinR 16× memory: **add explicit clarifying sentence** in §6.3 (1-bit signature footprint, not V3 cascade footprint).
  - (e) Silvertorch peak-memory anomaly: **re-profile** with explicit `del kmeans_state; torch.cuda.empty_cache()` (~15 min GPU). Use re-profiled number as deployment-relevant.
  - (f) linr_v2 arxiv parity Δ: **root-cause investigation queued** (tentative — "maybe rerun"). For now, footnote it as a known systematic shift; revisit if time permits.
  - (g) Seeds: **drop the topic entirely** — no §7 "no multi-seed sensitivity study" bullet; also drop the multi-seed Stage 4b rerun and the E2 seed-stability plot.
- **D12 — Eval-protocol cleanup** (Q13):
  - (a) Result-JSON `suite` field: **no code change**. In the thesis text, use the cleaner naming ("filtered" / "unfiltered" or equivalent) and footnote the historical field name once. Leave `evaluation/retrieval/results_io.py` and JSON schema untouched.
  - (b) `docs/system/evaluation.md` drift: **fix inline in the system doc**. Thesis does not footnote these mechanical drifts.
  - (c) NDCG notation: **include the formula somewhere** as a sanity reference (base-2 log, code-faithful). No HSE/GOST-specific style adjustments.
  - (d) CUDA driver version: A100 is SM 80, CUDA 12.8 per `uv.lock`. Author to confirm exact driver string at submission; G9 row populated with what is known now.
  - (e) Yambda dropout = 0.5 across every (scale, d) cell: **confirmed**.
- **D13 — Remaining small items** (Q14):
  - (a) Yambda-5b query-cache disk-budget footnote: **drop** — does not belong in the paper.
  - (b) **No usage of the synthetic 15M arxiv dataset anywhere in the thesis** — drop all references from §5.4 oracle discussion and elsewhere. Oracle transpose memory caveat is irrelevant at the catalogue sizes actually used.
  - (c) arXiv encoder = **frozen Nomic-Embed only**; no fine-tune "what if" baseline.
  - (d) §6.7 cross-backend memory parity for F1/G4: **include as a writer note**; render-time decision whether to add `index_mem` / `peak_mem` / `fwd_scratch` columns.
  - (e) linr_v2 arxiv parity Δ root-cause: **drop** — footnote-and-move-on; no rerun.
- **D14 — Title / structure / scope** (Q15–Q18):
  - **Title**: prefer the package-naming variant ("`retrieve`: PyTorch-фреймворк …"), but simpler. **Author has a pre-defined title** to be filled in at `thesis/main.tex:182` before submission. Plan-doc fallback retained.
  - **Chapter order**: confirmed as in `00-thesis-plan.md` (Введение · Ch.1 Lit review · Ch.2 Methods · Ch.3 Datasets · Ch.4 Implementation · Ch.5 Eval protocol · Ch.6 Results · Ch.7 Limitations · Заключение). No reordering.
  - **Page budget**: **20–40 pages of Russian text, target 25–30**. Compact thesis, not a full monograph. Implication: every chapter compresses aggressively. Suggested per-chapter rough budget — Введение ~2 pp · Ch.1 ~5 pp · Ch.2 ~4 pp · Ch.3 ~3 pp · Ch.4 ~4 pp · Ch.5 ~3 pp · Ch.6 ~4–5 pp · Ch.7 ~1–2 pp · Заключение ~1–2 pp = ~28–30 pp. Cut nice-to-haves rather than try to fit everything.
  - **Workflow phasing**: not needed. All research work is done; this doc stays a flat inventory. Agent execution ordering is left to whoever runs the work.
- **D15 — Figure / schema-extension scope** (Q19):
  - **Dropped figures (out of scope, not "blocked"):** A4 (3D bubble), B2b (bit-budget heatmap), B3 (K sensitivity), B4 (batch-size scaling), E1 (latency violins), F3 (Nsight kernel breakdown), F4 (build-time stacked bar), G2 (latency-at-fixed-recall), G6 (memory breakdown — fold into G1 if needed).
  - **Stage 4a schema extensions dropped (knock-on):** `mean_ms`, `p99_ms`, `build_time_s` + phase split, per-Triton-kernel timing via Nsight, per-query latency vector.
  - **Stage 4a schema extensions kept:**
    - `throughput_qps` — user wants it; unblocks throughput-axis variants of A2 (bs=16 Pareto) if writer chooses.
    - `topk_ids_jaccard_vs_torch` — unblocks F1 / G4 stronger-parity claim.
  - **Stage 4b reruns kept (revisitable):** the user may revisit extensions later; for now SimHashKNN k_bits deep sweep (B2b) is out, but the K=10 rerun, extended bs grid, etc. are gated by which figures the writer keeps.
  - **Mandatory figure set:** **writer agent decides** at render time. Suggested defaults (writer can trim further): A1 · A3 · B1 · B2a · C2 · D2 · D3 · F1 · F2 · G1 · G3 · G5 · G9 · G10 (~14 figures + tables). Long-tail figures listed in §5 are now "writer's call."

---

## §0 — Publication blockers (must clear before submission)

1. **Goodreads d128 + d256 filter-suite stale oracle rerun.** [07-results.md §6.0 — ACTION REQUIRED block; plot catalog ⚠]
   - Evidence: `linr_v1_filter_mask` reports Recall@100 = 0.5994 (d128) / 0.4887 (d256); expected ≈1.0. `linr_v3` deep_sweep on the same sweep reaches ≈0.99, contradicting the oracle.
   - Root cause: `evaluation/retrieval/oracle.py:113` validates oracle cache by shape only, not content.
   - Required:
     - (a) Delete stale oracle files `data/goodreads-work-id/<gt_subdir>/gt_topk_*.pt` for d128 and d256.
     - (b) GPU-rerun `evaluation/config/goodreads/d{128,256}-filter.yaml` against current SASRec checkpoints.
     - (c) Verify `linr_v1_filter_mask` Recall@100 ≥ 0.99 across all sweeps before extracting tables.
     - (d) Re-upload results; re-extract §6.4.2 / §6.4.3 / §6.4.4 / §6.2.4 cross-dim tables.
     - (e) Harden: add `item_embs_hash` field to oracle cache metadata.
   - Blocks: §6.4 Goodreads filter prose; plot D2 (Goodreads d128/d256 panels); Введение filter highlights; Заключение memory-vs-recall claims.

2. **Silvertorch reverse-clause wiring fix (D7).** [07-results.md §6.4.1 / lines 696–751; `docs/plans/silvertorch-reverse-clause-wrapper-fix.md`]
   - On Goodreads `c1_lang_reverse / c0c1 / all4`, silvertorch returns Recall@100 ≈ 0.0005–0.04 (vs baselines 0.5–0.6). The eval-side `SilvertorchAlgo.__init__` never propagates `clause_is_reverse` into `SilverTorch.register_index`; the codesigned exact-clause kernel already supports it.
   - Resolved path (D7): **fix it**, but only on the exact-index path (Bloom path does not algorithmically support reverse clauses).
   - Required:
     - (a) Wire `clause_is_reverse` through `SilvertorchAlgo.__init__` → `SilverTorch.register_index` → codesigned exact-clause kernel call.
     - (b) Keep the Bloom path raising (or skipping with a clear sentinel) on reverse-clause registration — Bloom subset tests are asymmetric, this is an algorithmic constraint, not a wiring gap.
     - (c) GPU-rerun Goodreads d64 / d128 / d256 filter sweeps (silvertorch rows on `c1_lang_reverse / c0c1 / all4` cells).
     - (d) Update §6.4 prose: drop the "INCOMPATIBLE" framing and report real numbers.
     - (e) Document in §7.7 the **remaining** limitation: silvertorch with Bloom filter does not support reverse clauses (algorithmic, not engineering).

---

## §1 — Open questions (still need a decision)

Resolved items live in the Decisions log above; only open / deferred questions remain here.

1. **Pareto-plot torch markers** — second color shade, or rely on §6.7 parity panel? Deferred to A1/A2 render time. — `07-results.md:428`.

---

## §2 — Agent text work (prose writing / reframing — no GPU, no data)

### Введение (`02-introduction.md`)
- Write full Russian prose from current English reference notes; full dual-goal framing (~3–5 pages). Sections: Актуальность · Цель · Задачи (6–8 bullets, both goals) · Объект/Предмет · Методы · Эмпирическая база · Brief results · Structure · Lineage statement.
- Per D2: include V4 in the lineage statement as "additional method introduced in the framework, an extension of a method proposed in LinR." Minimal emphasis ("just quantization").
- Per D3: lineage statement is a **Russian paraphrase**, no verbatim English blockquote.
- Per D4: brief-results paragraph cites **Pareto across datasets** and **silvertorch 10.94× max speedup** only. Drop V4 memory and V3 candidate_pool curves from Введение (those move to §6).
- Fill the `[DATA: …]` placeholders for the two retained headlines from `results-data/all_results_long.csv` once §4 extraction lands.
- Honour Goodreads-d128/d256 stale-oracle caveat: do NOT cite filter-suite numbers for those cells until §0 blocker 1 clears.

### Chapter 1 — Обзор литературы (`01-literature-review.md`) [reframe pending]
- Shift opening (lines 3, 5–13) from "reproducibility-only" to dual-goal framing; reposition citation audit from primary constraint to background hygiene.
- §1.1 "Connection to retrieve package": expand by 1–2 paragraphs linking the two-stage retrieval paradigm to the empirical chapter structure (Ch.3 datasets → Ch.5 protocol → Ch.6 results).
- §1.4 GPU-resident ANN: add 1–2 sentences connecting documented CAGRA/Milvus limitations (top-K cap, probe cap, no inline filtering) to the custom-Triton-kernels contribution.
- §1.5 Model-based retrieval: ensure intro (lines 69–72) frames LinR as the primary anchor and IVF+INT8+Bloom as the author's extension built from classical primitives.
- **NEW §1.10 — PyTorch retrieval-library landscape (~1 page)**: anchor on Liger Kernel (Hsu et al. 2024, LinkedIn — safe under citation policy) as the precedent for Goal-1 framing; cover torchrec (pending audit, almost certainly drop per D6), pytorch-metric-learning, conceptual ANN comparators (CAGRA / Milvus / SCANN), benchmark conventions (BEIR / BIG-ANN — pending audits per D6); end with the gap `retrieve` targets.
- §1.11 Positioning: rewrite final paragraph to match the dual-goal language template in `00-thesis-plan.md` lines 134–145.
- Per D1: PyTorch reference is a GitHub URL footnote, not a paper cite. Apply consistently in §1.6 (Triton/PyTorch substrate).
- Per D6: execute audits #9–#19 — for each, verify author list on arXiv/DBLP; if any Meta affiliation, drop. Propose substitute where natural (e.g., HNSW retains Malkov & Yashunin 2018); otherwise drop entirely. Rebuild `thesis/references.bib` from the surviving set. Audits are listed in the table at lines 67–84 of `00-thesis-plan.md` (TorchRec, Monolith, Embedding survey, PinnerSage, ItemSage, Pinterest Unified, MEVI, Manas HNSW blog, LiGNN, BEIR, BIG-ANN).
- Per D14 page budget: **Ch.1 target ~5 pages total** (not 3–5 per subsection). Compress every subsection — keep one paragraph per subsection minimum, cut secondary references. Subsection inventory stays from `00-thesis-plan.md`, but prose density drops.

### Chapter 2 — Методы (`03-methods.md`)
- Translate §2.3 intro lineage statement verbatim into Russian; it is structurally load-bearing for the citation audit.
- §2.3.4 add either an "Algorithm 4b" pseudocode box for the exact-mode kernel variant or an inline annotation in Algorithm 4.
- §2.3 motivation: add forward-reference connecting the co-designed retriever's design to §6.5.1 Pareto-dominance observation.
- "Open notes" §550–556: pull concrete `L`, `L_p`, `M`, `H`, `n_lists`, `n_probe` defaults from `evaluation/config/*.yaml` and fill them as examples in algorithm descriptions.

### Chapter 3 — Датасеты (`04-datasets-notes.md`)
- §2.7 wide-schema (lines 310–313) — write design-rationale prose for the `[N+1, 1, 32]` wide tensor.
- §3.3 arXiv (lines 381–386) — state explicitly that arXiv uses a frozen Nomic-Embed text encoder, not SASRec, before the §7 SASRec quality table. Per D13c: no fine-tune baseline; frozen only.
- Per D13b: do not include `synth_arxiv` (15M synthetic catalogue) as a dataset row anywhere in Ch.3.
- §4.4 Yambda (lines 506–510) — quote the Yambda paper's rationale for the 30-min train/test gap ("mimics the latency between model training and deployment").
- Per D8: §6.8 / §7.5 footnote yambda-5b-d256 as intentionally omitted (compute budget — SASRec convergence at scale not reached).

### Chapter 4 — Фреймворк (`05-implementation.md`) [reframe pending]
- **NEW §4.0 — "Фреймворк как библиотека" (~1 page)**: contract surface (`queries: Tensor` / `items: Tensor | indexed` → top-K ids+scores), extension points (`Backend`, `FilterModule`, autotune hooks), Liger-Kernel-shaped positioning. Cite `retrieve/__init__.py` exports.
- **NEW §4.11 — "Encoder-agnostic interop" (~1 page)**: the framework consumes `(queries, items)` regardless of producer; demonstrate via SASRec/gSASRec (Goodreads, Yambda) and Nomic-Embed (arXiv); cross-link Ch.3.
- Per D10a: list `tune.py` as part of §4.2 public API (alongside layers).
- Per D10b: **no README example reproduction** in §4.2.
- Per D10c: **skip torch.export mention** in §4.5.
- Per D10d: **no `tune-kernels` transcript** in §4.6.
- Per D10e: **no test counts or test-case discussion** anywhere in Ch.4. Parity claims can cite specific test files when relevant, but no headline "162 tests" or `pytest --collect-only` numbers. Skip §4.9 LOC/test-count re-pinning at submission.
- §4.2 — note in prose that `quantize_int8_global` is package-internal vs `quantize_int8` public (FLAG-CONSISTENCY callout).
- §4.3.1 — cite the A100 tuning rationale (`block_n=32`, `num_warps=8`) from kernel file headers.
- §4.3.3 — one-sentence note on Triton's 3D launch-grid workaround (per-axis 65,535 limit).
- §4.4 — honest note: PostfilterKNN / PostfilterKNNInt8 run identical torch code under both `backend="torch"` and `backend="triton"` (parity is trivial here).
- §4.7 — document the `ExactAttributeFilter.evaluate_subset` O(N) fallback as a current limitation; reference `retrieve/src/retrieve/interfaces.py:22-25` invitation to ship a fused O(P) kernel.
- §4.8 — reconcile OPORP citation: in-code docstring says "Li et al. 2019" but the correct reference is Li & Li 2023, arXiv:2302.03505 (confirmed in `articles/linr.md:368-369`).

### Chapter 5 — Протокол (`06-eval-protocol.md`)
- §5.3 — describe dispatch as "two orthogonal axes — encode mode × filter mode" (current text drifts from `docs/system/evaluation.md:37-44`).
- Per D12a: thesis prose uses "filtered" / "unfiltered" naming; footnote once that the JSON schema retains the historical `suite` field (`yambda` / `filter`). No code migration.
- Per D12b: fix `docs/system/evaluation.md` inline (`conf/` → `config/`, batch size and reps drifts, CLI command names, `algo_modules`, `evaluate.py` → `cli/evaluate.py`) — separate commit, do not mention in the thesis.
- Per D12c: §5.2 Metrics — keep one NDCG formula box (base-2 log, `r_i` notation) as a sanity reference; no HSE/GOST-specific tooling.
- Per D12d: §5.6 / G9 — record GPU = A100-SXM4-80GB (SM 80), CUDA 12.8 per `uv.lock`; author confirms driver version at submission.
- Per D12e: §5.6 / §7.5 — Yambda dropout = 0.5 across every (scale, d) cell.
- Per D13a: drop the §5.3 footnote about yambda-5b query-cache disk-budget guard.
- Per D13b: **strip all references to the synthetic 15M arxiv dataset from the thesis**. §5.4 oracle-transpose memory caveat is removed; do not include `synth_arxiv` anywhere in the chapter contracts, dataset table, or sweep dimensions list. (`evaluation/datasets/synth_arxiv.py` stays in the repo as an unbenchmarked utility.)
- Per D13c: arXiv encoder is **frozen Nomic-Embed only**; §5.5 confirms no fine-tune path.
- Drop the optional JSON-Schema appendix idea (per D12a, no schema formalization).

### Chapter 6 — Результаты (`07-results.md`) [reframe pending]
- Per D7: report real silvertorch reverse-clause numbers (after the wiring fix + rerun) in §6.4.1 / §6.4.2 / §6.4.3 / §6.4.4. **Do not** use the "INCOMPATIBLE / greyed" framing. Add a short paragraph explaining the Bloom-path algorithmic limitation and §7.7 cross-reference.
- Per D8: footnote yambda-5b-d256 absence in §6.2.3 / §6.6.1 / G8 / C1 as intentional ("compute budget — SASRec convergence at scale was not reached").
- Per D11d: §6.3 — add LinR-paper-16× clarifying paragraph (1-bit signature alone, not V3 cascade footprint).
- Per D11c: §6.0 / §6.6.1 — use measured Yambda corpus sizes (3.06M / 5.37M); add a short paragraph explaining provenance (which slices kept, listens-only action filtering, how the retrieval corpus was derived from the raw 9.39M catalogue).
- Per D13e: §6.7.7 — footnote linr_v2 arxiv parity Δ ≤ 5.3e-3 as a known systematic shift; no root-cause investigation.
- Per D13d: §6.7 / F1 / G4 — include a writer-note for the optional cross-backend memory parity columns (`index_mem` / `peak_mem` / `fwd_scratch`); render-time decision.
- Per D11g: drop seed-stability section (§6.8.4) entirely; no E2 plot, no multi-seed discussion.
- Per D13b: scrub any references to the synthetic 15M arxiv dataset from §6 prose / tables / sweep listings.
- Reframe chapter intro and §6.0 around comparative-empirical-study framing (not reproducibility check); keep §6.8.3 / G10 as one piece of evidence among many.
- Add §6 prose for V4 50% memory reduction at zero Recall loss and V3 candidate_pool recovery (relocated from Введение per D4).
- Fill all `[DATA: …]` and Stats-blocked numbers after §3/§4/§5 of this plan complete.
- Per D14 page budget: **Ch.6 target ~4–5 pages of prose**. Figures and tables carry most of the load; prose is short connectives between them. Cut sections that depend on dropped material (E2 seed-stability, synth-15M anything). Pareto sections (§6.2, §6.4) get the longest prose; F-series engineering validation gets short paragraphs around the tables.

### Chapter 7 — Ограничения (`08-limitations.md`)
- Per D7: §7.7 limitation now reads "silvertorch's codesigned Bloom-path probe kernel does not support reverse-clause predicates (algorithmic — Bloom subset tests are asymmetric). Exact-clause path supports reverse via XOR." Remove any "wiring gap" framing.
- Per D5: multi-GPU sharding (§7.1) and live-update API (§7.2) remain Limitations content; do NOT migrate to Заключение outlook.
- Add "no multi-seed sensitivity study" bullet if clarification #41 says yes.

### Заключение (`09-conclusion.md`)
- Per D5: delivered-work-only. Structure = (a) restate Цель, (b) tick off Задачи with shipped artifacts, (c) the same 1–2 quantitative highlights as Введение (Pareto across datasets, silvertorch 10.94×).
- **Do not include**: outlook / future-work paragraph, open-source release statement, PyPI package name, license type, commit hash, 7th Задача.
- Multi-GPU + live-update mentions belong only in Ch.7 Limitations; do not duplicate in Заключение.
- ~1–2 pages Russian prose. Avoid Goodreads-d128/d256 filter numbers until §0 blocker 1 clears.

### Front matter (`thesis/main.tex`)
- Per D14: replace placeholder title (line 182) with the **author's pre-defined title** (simpler variant of the package-named option (a)).
- Replace student name (line 187) and supervisor info (lines 189–191) — needs author confirmation.
- Write Russian аннотация (lines 205–214, 1500–2000 chars) and English abstract (lines 220–231, 1500–2000 chars).
- Wire `\input{chapters/0X-...}` lines after the ToC.

### Глоссарий / приложения
- App. A — терминологический словарь: IVF, INT8, OPORP, Bloom, clause, narrow/wide filter, codesigned kernel, top-K, Recall@K, NDCG, MRR.
- App. B — список сокращений: existing + ANN, IVF, INT8, OPORP, KNN, BCE, GBCE, NDCG, MRR.
- App. C — 3 short snippets: fused codesigned kernel, OPORP popcount kernel, one `combine_indices` composition.
- App. D — длинные таблицы per-cell quality/latency in `longtable`.

---

## §3 — Coding work (no GPU required, but harness or scripts change)

### Schema extensions in `evaluation/retrieval/bench_tools.py` (Stage 4a — D15)
Two items only; the rest were dropped with the long-tail figures.

1. **`throughput_qps`** = `n_queries_total / total_wall_clock_s` over the timing window. Unblocks the throughput-axis variant of A2 (bs=16 Pareto) if the writer keeps it.
2. **`topk_ids_jaccard_vs_torch`** between Triton and torch backends (target >0.999). Unblocks F1 / G4 stronger-than-recall parity claim.

### Eval-harness fixes / hardening
3. **Oracle content-hash validation** in `evaluation/retrieval/oracle.py:113`: add `item_embs_hash` to cache metadata; bust on mismatch. Required by §0 blocker 1.
4. **Silvertorch `clause_is_reverse` plumbing** (D7 — confirmed): wire through `SilvertorchAlgo.__init__` → `SilverTorch.register_index` → codesigned exact-clause kernel call. Bloom path must reject reverse-clause registration with a clear error (algorithmic limitation, not wiring). Plan doc: `docs/plans/silvertorch-reverse-clause-wrapper-fix.md`.
5. **`suite` field naming**: no code change (D12a). Skip.

### Plot-rendering scripts (under `docs/thesis/results-data/recipes/`)
Need a shared `palette.py` for cross-figure color consistency (algo→color, backend→marker shape, dim→marker). Then one script per figure family (writer agent picks which to actually render from this list — D15):
6. `A1_pareto_recall_latency_bs1.py` — upgrade from `pareto_quality_3x3.png` to the catalog spec.
7. `A2_pareto_recall_latency_bs16.py` — new; throughput-axis variant possible once `throughput_qps` lands.
8. `A3_pareto_recall_memory.py` — new; requires `bytes_per_vec` derivation in `extract.py`.
9. `B1_silvertorch_heatmap.py` — upgrade with iso-contours.
10. `B2a_linr_v3_pool_curves.py` — upgrade with twin-y latency overlay.
11. `D1_recall_latency_by_selectivity.py` — new; needs per-query selectivity buckets.
12. `D2_filter_recall_vs_selectivity_<dataset>.py` — upgrade (blocked on §0 for goodreads d128/d256).
13. `D3_selectivity_bars_<dataset>.py` — new for arXiv (blocked on Stats item 11).
14. `E3_selectivity_cdf_<dataset>.py` — new for arXiv (same blocker).
15. `F1_parity_scatter.py` — partial; gains Jaccard col once schema item 2 lands.
16. `F2_speedup_dist_by_algo.py` — upgrade to faceted (algo × bs).
17. `C1_yambda_n_scaling.py` — partial; yambda-5b-d256 omitted per D8, so the C1 chart has only 5b-d64 + 5b-d128 + 500m-{d64,d128,d256}.
18. `C2_dim_scaling.py` — straightforward render.

### Extract / data plumbing
19. **`docs/thesis/results-data/extract.py` — add derived columns**: `bytes_per_vec = index_mem_mib * 2^20 / n_items` and `ratio_vs_fp32 = bytes_per_vec / (4*d)`. Required by A3 / G1.
20. **`docs/thesis/results-data/recipes/walk_results.py`** — extend with the per-cell aggregation needed for G1 / G10.
21. **`results-data/recipes/stat_tests.py`** — new helper for paired permutation tests + bootstrap 95% CIs on Recall difference (used selectively for §6.2 / §6.4 prose claims).

---

## §4 — Stats / data extraction (no GPU; reading existing files)

### From `evaluation/results/**/*.json` (already on disk)
1. Build `all_results_long.csv` keyed by (suite, cell, impl, backend, k, batch_size, seed); medians/p20/p80 across seeds. (Most of `extract.py` does this — verify + extend.)
2. G1 headline summary table: `(dataset, dim, algo) × {Recall@100, median_ms bs=1, bytes/vec, peak_mem}`; bold Pareto-cell rows.
3. G3 recall-at-fixed-latency table (✅ derivable today).
4. G5 Triton/torch speedup matrix (✅ derivable today).
5. G6 memory-breakdown table: extend `d128_quality_memory.csv` to d64 and d256.
6. G10 mirror of LinR Tables 3/4 (partial — `p95_ms` and K=2000 missing; depends on Stage 4a + 4b).
7. Backend-parity dataset for F1 / G4 (`backend_parity_recall.csv` exists; needs Jaccard col post schema item 5).

### From SASRec training-time JSONs (need GPU host access to copy off)
8. Read `train_metrics.json` per checkpoint at `data/<dataset>/checkpoints/gsasrec-d<N>-drop0.5-id/` → extract `total_time_sec`, `peak_gpu_mem_bytes`, per-epoch losses, val Recall@100 per epoch. Fills Ch.3 §7.5 training table + §7.8 training-curve plots (items 8/9 in §5 plotting).
9. G8 SASRec quality ceiling CSV — already partial (`sasrec_quality_ceiling.csv`); finish by adding yambda-5b-d256 once GPU rerun item 38 completes.

### From dataset parquets (CPU-bound)
10. Run `goodreads.py prep` locally → re-run Recipe A → re-emit `goodreads_seq_len_cdf.csv`, item-popularity rank-frequency CSV.
11. Run `arxiv.py convert` + `attrs` → emit `arxiv_clause_c0_*.csv`, `arxiv_clause_c1_*.csv`, etc. Mirrors existing Goodreads files. Required by D3 / E3 arxiv panels.
12. Run `yambda.py prep` (500m and 5b) → emit listens-per-user CDF, item-popularity CDF; extract item counts from `prep_log.json` for §6.8 cross-dataset table.

### From repo / lockfile metadata (G9)
13. Reproducibility-disclosure G9 row values: GPU SKU (already known: A100-SXM4-80GB), CUDA driver (clarification #24), PyTorch + Triton versions (`uv.lock`), seeds (configs), container image tag.

### Profiling / debugging stats
14. Bs > 1 fwd_scratch profile from filter-suite (per D11b).
15. Silvertorch peak-memory re-profile with explicit KMeans cleanup (per D11e — paired with §6 GPU item 3).

---

## §5 — Plotting / figures (rendering only — data is ready, modulo §3/§4)

Cross-ref: `docs/thesis/results-data/recipes/plot_catalog.md` is the source of truth.

### Ch.6 figures (Pareto / sensitivity / scaling / filter / parity)
Writer agent picks the actually-rendered subset (D15). Long-tail figures (A4, B2b, B3, B4, E1, F3, F4, G2, G6) are dropped from this list entirely.

1. A1 — 🟡 Recall–latency Pareto bs=1, 3×3 faceted.
2. A2 — 🆕 Recall–latency Pareto bs=16, 3×3 faceted; throughput-axis variant possible once `throughput_qps` lands.
3. A3 — 🆕 Recall–memory Pareto (needs `bytes_per_vec`).
4. B1 — 🟡 Silvertorch (n_lists × n_probe) heatmap with iso-contours.
5. B2a — 🟡 LinR V3 candidate_pool curves with twin-y latency.
6. C1 — ⏳ Yambda 500m vs 5b N scaling (5b-d256 omitted per D8).
7. C2 — ✅ Dimensionality scaling (d64 / d128 / d256).
8. D1 — 🆕 Recall–latency Pareto by selectivity bucket.
9. D2 — 🟡 + ⚠ Recall vs selectivity (twin-panel); goodreads d128/d256 panels blocked on §0.
10. D3 — ⏳ Per-clause selectivity bars; arXiv panel blocked on Stats item 11.
11. E3 — 🆕 Per-query selectivity CDF; arXiv panel blocked on Stats item 11.
12. F1 — ⏳ Backend parity scatter (Jaccard col blocked on schema item 2).
13. F2 — 🟡 Backend speedup distribution (faceted).
14. G1 — 🆕 Headline summary table.
15. G3 — ✅ Recall-at-fixed-latency.
16. G4 — ⏳ Backend parity table (Jaccard col).
17. G5 — ✅ Speedup matrix.
18. G7 — 🆕 Cross-over points (writer's call).
19. G8 — ⏳ SASRec quality ceiling (5b-d256 omitted per D8).
20. G9 — 🆕 Reproducibility / hardware disclosure (driver TBD).
21. G10 — ⏳ Mirror LinR Tables 3/4 (writer's call on which axes to keep).

### Ch.2 figures (methods block diagrams)
32. Per-algorithm block diagrams (LinR V1, V2, V3, V4; co-designed IVF+INT8+Bloom). Hand-drawn / TikZ.
33. One algorithm box per algorithm (Russian pseudocode in `algorithm2e`).
34. Memory/complexity comparison table.

### Ch.3 figures (datasets)
35. Goodreads pipeline schematic.
36. Goodreads sequence-length histogram (`goodreads_seq_len_cdf.png` exists).
37. Goodreads item-popularity rank-frequency plot (`goodreads_item_popularity.png` exists).
38. arXiv pipeline schematic (text encoder).
39. arXiv papers-per-category and papers-per-year bars (blocked on Stats item 24).
40. Yambda GTS timeline with per-scale `n_users_kept` annotations.
41. Yambda item-popularity CDF (blocked on Stats item 25).
42. SASRec training-loss curve grid (blocked on Stats item 21).
43. SASRec val-Recall@100 vs epoch with best-epoch marker (blocked on Stats item 21).

### Ch.4 figures (implementation)
44. Architecture diagram — `retrieve/` module tree with `layers/` → `kernels/` arrows. TikZ.
45. Sequence diagram for co-designed (silvertorch) forward pass: batch → IVF probe → filter phase → INT8 score → top-K. tikz-uml.
46. Algorithm boxes — fused `codesigned_probe_score` (mandatory); `oporp_1bit_match_topk` (optional).
47. Volumetric table (LOC / classes / functions / tests) — already drafted in §4.10.
48. Compile-vs-eager speedup table — already drafted in §4.5.

---

## §6 — GPU reruns (need A100-SXM4-80GB host)

Approximate hours per item are pessimistic single-GPU; many can be parallelized.

### Publication-blocker reruns
1. **Goodreads d128 + d256 filter suite vs fresh oracle** — §0 blocker 1. Delete stale `gt_topk_*.pt`, re-run `evaluation/config/goodreads/d{128,256}-filter.yaml`. ~2 h. Verify `linr_v1_filter_mask` Recall@100 ≥ 0.99 on all sweeps. Re-upload, re-extract.

### Silvertorch reverse-clause rerun (D7)
2. **Silvertorch Goodreads d64 / d128 / d256 filter rerun** with the `clause_is_reverse` wiring fix (coding §3 item 9). Targets `c1_lang_reverse / c0c1 / all4` cells. ~1–1.5 h. Unblocks §6.4 prose + §7.7 limitation rewording.

### Profiling
3. **Silvertorch peak-memory re-profile** with explicit `del kmeans_state; torch.cuda.empty_cache()` inside `KMeansTorch.fit` (per D11e). ~15 min.

(Stage 4b extended-bs / K=10 / SimHashKNN-k_bits reruns are dropped per D15 — the corresponding figures B3 / B4 / B2b / G2 are out of scope. User may revisit.)

Pessimistic critical-path total (serial, items 1–3): ~3 h GPU time.

---

## §7 — Verification / submission gates

Before declaring the thesis ready:
1. Citation audit: `grep -i -E "(meta|facebook|fair|instagram|whatsapp)" thesis/references.bib` returns zero author-affiliation matches; manually verify final bib against arXiv/DBLP.
2. Compile: `cd thesis && xelatex main.tex && biber main && xelatex main.tex && xelatex main.tex` produces `main.pdf` with all `\cite{}` resolved.
3. Data integrity: every Ch.6 plot/table traces to a CSV in `docs/thesis/results-data/`; CSVs reproducible from `evaluation/results/*.json` via committed extraction script.
4. Cross-refs: every `\ref{}` resolves; ToC populated; figure/table numbering by chapter.
5. Char counts: Russian аннотация and English abstract both within 1500–2000 chars.
6. Backend parity claim cited from `retrieve/tests/`.
7. Re-pin LOC + test-count + commit SHA in §4.9 / §4.10 / Заключение.
8. Goodreads d128/d256 §0 blocker cleared; silvertorch reverse-clause decision committed and reflected in §6.4 + §7.7.
9. Final grep: no "SilverTorch" (cap S) in body prose (allowed only in code paths / file names).
10. Lineage statement appears (translated) in both Введение and §1.5 of Ch.1.

---

## Critical-path summary (shortest route to submittable)

Per D14 / D15 — flat inventory; agent execution ordering is the runner's choice. The hard dependencies that matter:

1. **§0 blocker 1** (Goodreads filter rerun + oracle hardening) gates Ch.6 §6.4 finalization and any filter-suite numbers cited elsewhere (Введение, Заключение avoid them until cleared).
2. **§0 blocker 2 / D7** (silvertorch reverse-clause wiring fix + rerun) gates §6.4 final numbers and §7.7 final wording.
3. **§3 schema items 1, 2** (`throughput_qps`, `topk_ids_jaccard_vs_torch`) are the only remaining harness extensions; everything else (mean_ms, p99_ms, build_time_s, per-kernel, per-query) is dropped with the long-tail figures (D15).
4. **§4 stats extraction** (CPU-only) is parallel with everything.
5. **§5 plotting** waits on §3 schema (for F1 Jaccard) and §0 / §6 GPU reruns (for D2, C1, G8 cells).
6. **§2 prose** is the long pole; Ch.1 audits and Ch.6 reframe gate Введение and Заключение. Page budget 25–30pp Russian (D14).
7. **§7 verification** is the final gate.

This file is the canonical inventory; remove items as they close.
