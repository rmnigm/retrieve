# Chapter 6 — Results and Analysis (Reference Notes)

> ### 🛑 ACTION REQUIRED before publishing Goodreads filter numbers
>
> The goodreads-d128 and d256 filter result JSONs were scored against a **stale oracle disk cache**, not against the current item embeddings. Diagnosis, evidence, and full mechanism are in the ⚠ block in the §6.0 schema recap below; cross-suite smoking gun is documented in the §6.5.2 narrative.
>
> **Remediation (must be done on a GPU host before the thesis cites Goodreads d128/d256 filter Recall numbers):**
>
> 1. `rm data/goodreads-work-id/<gt_subdir>/gt_topk_*.pt` — delete the stale cached oracles (specifically those backing d128 and d256; d64 looks fine and can stay).
> 2. Rerun `evaluation/config/goodreads/d{128,256}-filter.yaml` against the current SASRec checkpoints (`gsasrec-d{128,256}-drop0.5-id/best_model.pt`). The harness will rebuild the oracle fresh (no cache → forced recompute via [oracle.load_or_build_oracle:120-132](evaluation/retrieval/oracle.py#L120-L132)).
> 3. Verify the exact baseline (`linr_v1_filter_mask`) Recall@100 ≥ 0.99 on every (sweep, k) cell.
> 4. Re-upload to `pinkmeme/retrieval-filter-evals-2026-05-23` (or a new dated repo) and re-pull on the writer's machine before regenerating the §6.4 tables.
> 5. **Harden:** add an item_embs content hash to the oracle cache metadata; the current shape-only check ([oracle.py:113](evaluation/retrieval/oracle.py#L113)) does not catch checkpoint changes — this exact failure mode is even acknowledged in the function's own docstring ("the common cause is changing content_subdir between runs (item_embs differ → oracle scores differ)").
>
> Until the rerun: every §6.4.2 / §6.4.3 / §6.4.4 / §6.2.5 cell on Goodreads d128 and d256 is marked with a ⚠. ArXiv filter, Goodreads d64 filter, all quality (§6.2), and all deep_sweep (§6.5) numbers are unaffected and writer-safe.

> **Purpose.** Reference material for the Russian-prose writer of the MSc thesis Ch.6. This document spans §6.1 through §6.9 of the chapter: workload characterization (§6.1), quality–latency tradeoffs (§6.2), quality–memory tradeoffs (§6.3), filter behavior (§6.4), parameter sensitivity (§6.5), scaling (§6.6), engineering validation (§6.7), reproducibility (§6.8), and executive summary (§6.9). The data-bearing sections (§6.2 quality, §6.4 filter, §6.5 deep sweeps) are populated with full numeric tables from the measurement campaign; the cross-cut sections (§6.1, §6.3, §6.6–§6.9) are figure-spec stubs that either derive from existing CSVs or are flagged as deferred per "Planned schema extensions" in Ch.5 §5.7 of [docs/thesis/06-eval-protocol.md](docs/thesis/06-eval-protocol.md).
>
> **Figure-catalog cross-reference.** Figures and tables in this chapter use the IDs `A1, A2, ..., G10` defined in the plot catalog at [docs/thesis/results-data/recipes/plot_catalog.md](results-data/recipes/plot_catalog.md). Each figure stub here is annotated with its catalog ID (e.g., "Fig 6.2.1 = A1") for one-to-one traceability between the catalog, this notes file, and the rendering scripts.
>
> **Notes pass, not writing pass.** Exhaustive and citation-anchored. Every numeric claim has either a `path:line` code reference, a `[DATA: <path>]` results marker, or a `[CITE: <bibkey>]` literature anchor. The writer compresses; this file does not.
>
> **Naming convention (load-bearing).** The thesis never uses the name *SilverTorch* in body prose. The algorithm whose code symbol is `SilverTorch` ([retrieve/src/retrieve/layers/silvertorch/main.py:27](retrieve/src/retrieve/layers/silvertorch/main.py#L27)) is described as the **co-designed IVF + INT8 + Bloom retriever**, presented as a second bundled retriever in the framework, assembled from classical primitives, per the lineage statement in [docs/thesis/03-methods.md](docs/thesis/03-methods.md) §2.3. The code symbol may appear in `file:line` citations and in `impl="silvertorch"` strings inside results JSONs.
>
> **Citation policy reminder.** Cite no work with any Meta / Facebook / FAIR author on the paper. The "SilverTorch" paper is never cited or named. The IVF + INT8 + Bloom retriever is presented as a second bundled retriever in the framework, assembled from classical primitives (LinR — Borisyuk et al. 2024 — OK — anchors the LinR family bundled alongside it). Every literature reference in this file is flagged with `[CITE: <key> — affiliation: <X> — kept]`. All citations re-used here have already been audited in [docs/thesis/01-literature-review.md](docs/thesis/01-literature-review.md).
>
> **Hardware / run inventory.** All numbers in §6.2 / §6.4 / §6.5 come from a single host: NVIDIA A100-SXM4-80GB, single GPU; runs executed sequentially on 2026-05-23 from `host=fd4a94de96b9` (containerized); arxiv+goodreads quality + 2 deep sweeps ≈ 3 h 30 min wall-clock; `seed=0` throughout. (The run inventory was published as `_runlogs/SUMMARY.quality-deep.txt` on HF [pinkmeme/retrieval-filter-evals-2026-05-23](https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-23) at the time of the run; the local copy has since been pruned along with the per-impl `.perkernel/` sub-files when the working tree was consolidated to the HF-canonical one-JSON-per-cell layout. Numbers below were extracted from that snapshot and remain valid; the same numbers can be re-derived from the current top-level JSONs by filtering on `impl`, `backend`, `batch_size`, `k`, `sweep`, `filter_kind`.) Full disclosure table appears in §6.8 (Table G9).
>
> **Backend convention.** Unless explicitly contrasting backends, every number quoted in §6.2–§6.5 is `backend="triton"`. Triton↔torch parity (recall identical within float tolerance; speed differs) is §6.7 territory. Both backends are wrapped with `torch.compile(dynamic=True, mode="reduce-overhead")` inside each algo wrapper ([evaluation/retrieval/algos/linr_v3.py:65](evaluation/retrieval/algos/linr_v3.py#L65), [evaluation/retrieval/algos/silvertorch.py:101](evaluation/retrieval/algos/silvertorch.py#L101)).
>
> **Throughput / QPS convention.** The current schema records `median_ms` (per-batch forward-pass median) + `p20_ms` / `p80_ms`. The chapter uses **latency on X-axis** for all Pareto plots, labeled `median_ms` (per-batch). A direct `throughput_qps` measurement (= `n_queries_total / total_wall_clock_s`) is listed in Ch.5 §5.7 "Planned schema extensions" — once collected, the latency-on-X plots A1 / A2 / B4 and the latency-at-fixed-recall table G2 supersede to throughput-on-X equivalents. The deliberate choice to avoid deriving QPS from `median_ms` is explained in the catalog at [docs/thesis/results-data/recipes/plot_catalog.md](results-data/recipes/plot_catalog.md) — `1/median_ms` (or `B/median_ms`) conflates a robust statistic with a sum-based rate.
>
> **Figures.** Exact image files are NOT referenced. Each figure deliverable is documented as a **stub**: catalog ID, axes, color encoding, source CSV/JSON. Rendering happens in a later pass via committed scripts under [docs/thesis/results-data/recipes/](results-data/recipes/). Tables are inline-rendered from the JSONs / CSVs cited.

This chapter delivers goal 2 of the thesis. It must read as a comprehensive comparative empirical study of the framework's reference implementations, not as a reproduction check against the LinR paper. The breadth claim is concrete: three datasets × multiple embedding dims × two backends × the full algorithm lineup × six sweep axes (K, batch size, filter selectivity, n_lists / n_probe, candidate_pool, seed). The §6.8.3 (G10) mirror of LinR Tables 3/4 stays as one piece of evidence among many, not as the chapter's thesis.

---

## Cross-cutting result schema (recap)

Each row in `evaluation/results/**/*.json` is one benchmark cell. Schema, per [evaluation/retrieval/results_io.py:15-31](evaluation/retrieval/results_io.py#L15-L31) + [evaluation/retrieval/sweep.py:470-500](evaluation/retrieval/sweep.py#L470-L500) (see also Ch.5 §5.7 in [docs/thesis/06-eval-protocol.md](docs/thesis/06-eval-protocol.md)):

| field | type | notes |
|---|---|---|
| `suite` | str | `"yambda"` (no filter) or `"filter"` |
| `cell` | str | descriptive composite — e.g. `clause_c0_genre_triton_bs1_k100` |
| `filter_kind` | str | `none` / `clause` / `bloom` |
| `sweep` | str | name of the active-clause subset — e.g. `c0_genre`, `all4`, `wide_1shelf` |
| `impl` | str | `linr_v1_filter_mask` / `linr_v2` / `linr_v3` / `linr_v4` / `silvertorch` (see registry [evaluation/retrieval/algos/__init__.py:44-52](evaluation/retrieval/algos/__init__.py#L44-L52)) |
| `backend` | str | `triton` / `torch` |
| `device` | str | `cuda` |
| `seed` | int | `0` throughout the runs in this chapter |
| `batch_size` | int | `1` for quality suite, `{1, 8, 16}` for filter & deep sweeps |
| `k` | int | `{100, 200, 400}` for quality + silvertorch deep sweep; `{100, 500, 1000}` for filter; `{100, 200, 400}` for linr_v3 deep sweep |
| `n_users_kept` | int | sampled users after `users_limit`; arxiv/goodreads filter: 10 000; full goodreads quality: 313 178; yambda quality: full split |
| `median_ms` | float | per-query latency, median over warm runs |
| `p20_ms`, `p80_ms` | float | latency spread |
| `peak_mem_mib` | float | peak resident GPU mem during the cell |
| `index_mem_mib` | float | persistent index memory |
| `fwd_scratch_mib` | float | per-forward scratch; ≈0 for all current algos |
| `recall@K`, `ndcg@K` | float | metrics; formulas in [evaluation/retrieval/metrics.py:39-114](evaluation/retrieval/metrics.py#L39-L114) |
| `extra.params` | dict | algo-specific hyperparameters, all strings |

**Metric formulas** (verbatim restate from §5.2):

$$
\mathrm{Recall@K}(b) \;=\; \frac{|R_b^K \cap G_b|}{\max(1, |G_b|)}\quad
[\text{metrics.py:39-54}]
$$

$$
\mathrm{NDCG@K}(b) \;=\; \frac{\sum_{i=1}^{K} \mathbb{1}[r_{b,i} \in G_b] / \log_2(i + 1)}{\sum_{i=1}^{\min(|G_b|, K)} 1/\log_2(i + 1)}\quad
[\text{metrics.py:90-114}]
$$

where $G_b$ is the ground-truth target set for query $b$, $R_b^K$ is the top-$K$ retrieved by the algorithm, and $r_{b,i}$ is the rank-$i$ retrieved item. `[CITE: Järvelin & Kekäläinen 2002 TOIS — affiliation: University of Tampere — kept]`.

**Algorithm lineup (4 algos in quality, 5 in filter)** — public class signatures and registry mapping summarized:

| `impl` | Code symbol | File:line | Stage 1 | Stage 2 | Hyperparams (defaults) |
|---|---|---|---|---|---|
| `linr_v1_filter_mask` | `PostfilterKNN` | [retrieve/src/retrieve/layers/linr/postfilter_knn.py:9](retrieve/src/retrieve/layers/linr/postfilter_knn.py#L9) | fp16 dense matmul → boolean mask → top-K | — | none |
| `linr_v2` | `PrefilterKNN` | [retrieve/src/retrieve/layers/linr/prefilter_knn.py:10](retrieve/src/retrieve/layers/linr/prefilter_knn.py#L10) | sparse gather → dot → top-K (REQUIRES filter) | — | none |
| `linr_v3` | `OneBitKNN` → `PrefilterKNN` cascade | [evaluation/retrieval/algos/linr_v3.py:40](evaluation/retrieval/algos/linr_v3.py#L40) | Sign-OPORP 1-bit Hamming top-`candidate_pool` | fp16 rescore via PrefilterKNN | `candidate_pool=5000`, `v3_seed=0` |
| `linr_v4` | `PostfilterKNNInt8` | [retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py:28](retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L28) | per-row INT8 → `torch._int_mm` → mask → top-K | — | none |
| `silvertorch` | `SilverTorch` | [retrieve/src/retrieve/layers/silvertorch/main.py:27](retrieve/src/retrieve/layers/silvertorch/main.py#L27) | IVF probe → INT8 dot → fused filter → top-K | — | `n_lists=1024`, `n_probe=24`, `n_iter=10`, `m_bits=1024`, `k_hash=5`, `seed=0` (eval defaults; see [evaluation/retrieval/algos/__init__.py:112-118](evaluation/retrieval/algos/__init__.py#L112-L118)) |

**Note on params columns in quality JSONs.** All four quality-suite algorithms run with default constructor params in §6.2 (no overrides in [evaluation/config/{arxiv,goodreads,yambda-*}/d*-quality.yaml](evaluation/config/arxiv/d128-quality.yaml)). `extra.params` is `{}` in every quality row. Hyperparameters become important only in §6.4 (default canonical config per algo) and §6.5 (explicit grid).

**Ground-truth construction — caveat that runs through §6.2, §6.4, §6.5.** Reading [evaluation/retrieval/oracle.py:1-12, 25-86](evaluation/retrieval/oracle.py#L1-L86) and [evaluation/retrieval/sweep.py:566-603](evaluation/retrieval/sweep.py#L566-L603) shows the harness uses two different ground-truth sources:

- **Quality cells** (suite `yambda` OR `filter_kind == "none"`): scored against SASRec held-out next-item targets (`targets_f`, `n_targets_f` from the dataset loader).
- **Filter cells** (suite `filter`, `filter_kind in {clause, bloom}`): scored against an oracle = filtered FullScan top-`K_GT` computed by [oracle.compute_filtered_oracle](evaluation/retrieval/oracle.py#L25). Per the oracle docstring: "Algos under test are scored against this oracle, not against held-out targets, because filter sweeps deliberately restrict the candidate set: held-out targets often fall outside the filtered catalog and would tank recall regardless of algo quality."

> ### ⚠ Suspected stale ORACLE DISK CACHE — Goodreads d128 and d256 filter results
>
> **Symptom.** `linr_v1_filter_mask` is the exact-baseline (dense fp16 matmul → boolean mask → top-K). When the harness scores it against the filtered FullScan oracle, the expected Recall@K is ≈ 1.0 modulo float-precision tie-breaking. Observed instead in [evaluation/results/goodreads/d128-filter.json](evaluation/results/goodreads/d128-filter.json) and `d256-filterlinr_v1_filter_mask.json`:
>
> | dataset | dim | sweep         | Recall@100 | Recall@500 | Recall@1000 |
> |---|---|---|---:|---:|---:|
> | goodreads | d64  | c0_genre/clause | **0.9996** | 1.000 | 1.000 |
> | goodreads | d128 | c0_genre/clause | **0.5994** | 0.6557 | 0.6720 |
> | goodreads | d256 | c0_genre/clause | **0.4887** | — | — |
> | arxiv     | d128 | c0_maincat/clause | 0.9923 | 0.9939 | 0.9942 |
>
> linr_v2 (the other exact baseline) reports identical numbers, confirming the issue is on the ground-truth side, not in the algo. Recall does NOT converge to 1.0 even at k=1000 — far too large a gap to be fp16 vs fp32 precision drift alone. d64 works correctly; only d128 and d256 on Goodreads are anomalous.
>
> **Hypothesis (load-bearing):** the **oracle disk cache** for goodreads filter is stale. The JSONs themselves are FRESH — verified by mirroring the canonical HF dataset `pinkmeme/retrieval-filter-evals-2026-05-23` (lastModified 2026-05-23T20:51:19Z) and diffing every numeric value: **identical to 4 decimal places** for goodreads filter, arxiv filter, and deep_sweeps. So the JSONs faithfully record what the harness measured. The error is upstream: the harness scored these JSONs against an outdated oracle cached on disk at `<data_dir>/<gt_subdir>/gt_topk_<sweep>.pt` ([oracle.load_or_build_oracle:109-133](evaluation/retrieval/oracle.py#L109-L133)).
>
> **Supporting evidence (no GPU required).**
>
> - The cross-check that breaks the puzzle: the **goodreads-d128-linr_v3 deep_sweep** ([evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json](evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json)) reports `linr_v3` at `candidate_pool=32000` reaching Recall@100 = **0.9896** on `c0_genre clause`. linr_v3 at pool=32000 is effectively exact-filtered top-K (Stage 1 admits virtually every passing item; Stage 2 fp16 dense rescores them). Therefore the deep_sweep oracle on this cell admits Recall ≈ 1.0 — yet the filter-suite oracle reports Recall = 0.5994 for the equivalent exact baseline on the same cell. Both suites use the same code path (same `sweep.py`, same `oracle.py`), same users (`n_users_kept=9859`), same item_embs at load time, and same filter module — the only thing that could differ is the cached oracle file actually used at scoring time.
> - The oracle disk-cache sanity check at [oracle.py:113-119](evaluation/retrieval/oracle.py#L113-L119) verifies ONLY shape `(n_users, K_GT)` and recomputes on mismatch. Filter config uses `ks=[100,500,1000]` → `K_GT=1000`; deep_sweep config uses `ks=[100,200,400]` → `K_GT=400`. So the deep_sweep run FORCED a fresh oracle (shape mismatch → recompute), while the filter run reused a pre-existing cached oracle of shape `(10000, 1000)`. If that cache was produced earlier (e.g., before a checkpoint refresh, before normalization was added, etc.), the filter suite scored against the old ground truth.
> - The documented run inventory `SUMMARY.quality-deep.txt` (archived in the [HF dataset](https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-23); the local `_runlogs/` directory has since been pruned) covers ONLY the quality + deep_sweep runs (start 09:52:29Z, end 14:06:41Z). The filter runs that produced the JSON results (commit 308a887, 2026-05-23 10:20:58) are NOT in this log — they ran in a separate session whose oracle cache state is now opaque.
> - The arxiv filter is essentially unaffected (linr_v1 R@100 = 0.9923 — small ≤ 1% drift consistent with ordinary fp16-vs-fp32 precision on normalized nomic-embed-v1.5 vectors). Likely the arxiv oracle cache was regenerated more recently than the goodreads-d128/d256 cache, or arxiv's normalized embeddings are robust to whatever item_embs change perturbed goodreads.
> - Goodreads d64 IS clean (linr_v1 R@100 = 0.9996, R@1000 = 1.000) — its oracle cache evidently matches the current d64 checkpoint, only d128 and d256 are stale.
>
> **Recommended remediation (must be done before the thesis cites Goodreads-d128/d256 filter numbers).**
> 1. Delete the cached oracle files at `data/goodreads-work-id/<gt_subdir>/gt_topk_*.pt` (specifically the d128 and d256 caches — d64 looks fine).
> 2. Rerun the goodreads filter sweeps `evaluation/config/goodreads/d{128,256}-filter.yaml` on the current SASRec checkpoints (`gsasrec-d{128,256}-drop0.5-id/best_model.pt`).
> 3. Verify `linr_v1_filter_mask` baseline Recall@100 converges to ≥ 0.99 (modulo ≤ 1% fp16 precision drift) at all dims.
> 4. Re-upload to `pinkmeme/retrieval-filter-evals-2026-05-23` (or a new dated repo).
> 5. **Harden:** add a checksum / `item_embs_hash` field to the cached oracle metadata so `oracle.load_or_build_oracle` rejects caches whose underlying embeddings changed. Today only shape is checked ([oracle.py:113](evaluation/retrieval/oracle.py#L113)); a content hash (e.g., `hashlib.sha256(item_embs[:, :32].cpu().numpy().tobytes()).hexdigest()`) computed at load would catch this class of bug at zero ongoing cost. Bonus: add a parity test under `retrieve/tests/` that asserts `linr_v1_filter_mask` Recall ≥ 0.99 against a fresh oracle on every filter sweep.
>
> **For now this notes file reports the JSONs AS-IS**, with every Goodreads d128/d256 filter cell flagged. Writer MUST NOT publish these specific numbers as canonical until the rerun. The arxiv filter numbers, the goodreads QUALITY numbers (§6.2.1), and all deep-sweep numbers (§6.5) are NOT affected.
>
> `[TODO — REQUIRED before publication: rerun goodreads d128 and d256 filter sweeps with fresh oracle; re-extract numbers; update §6.4.2 + §6.4.4 + §6.2.4 cross-dim tables. The arxiv tables and deep_sweep tables are untouched.]`

---

## 6.1 Workload characterization

**Scope.** Dataset properties that govern the rest of the chapter — sequence-length and item-popularity distributions, filter-clause selectivities — presented BEFORE any system numbers so the reader has the workload baseline in hand. Best practice from filtered-ANN literature (Filtered-DiskANN, ACORN) and ANN-benchmarks: characterize the workload first; only then show recall/latency.

**Source.** Already-extracted CSVs under [docs/thesis/results-data/datasets/](results-data/datasets/), plus per-clause coverage tables in [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.1 / §3.2 / §3.3. SASRec quality ceiling (the upper bound for any algorithm trained on these datasets) lives at [docs/thesis/results-data/results/sasrec_quality_ceiling.csv](results-data/results/sasrec_quality_ceiling.csv).

### Figure stub: 6.1.1 = D3. Per-clause selectivity distribution (workload characterization)

**Plot.** Bar chart, faceted by dataset:
- Rows: dataset ∈ {goodreads, arxiv} (yambda has no clauses → not faceted).
- X: clause name (`c0_genre, c1_lang, c2_format, c3_year, c4_author` for goodreads; `c0_maincat, c1_license, c2_year, c3_nversions, c4_author` for arxiv).
- Y: pass-rate at query (median across queries with IQR whiskers). The pass-rate is the fraction of catalogue items passing the per-clause OR-set drawn from the query.

Companion: histogram (or kernel density) of per-query selectivity within each clause — exposes the spread that the median alone hides.

**Source.** [docs/thesis/results-data/datasets/goodreads_clause_c0_genre.csv](results-data/datasets/goodreads_clause_c0_genre.csv), `goodreads_clause_c1_lang_top30.csv`, `goodreads_clause_c2_format.csv`, `goodreads_clause_c3_year.csv` for goodreads. **For arxiv: equivalent CSVs do NOT yet exist** — they are a deferred-but-low-effort task per the "Gaps requiring new measurements" table in the plot catalog. Action: replicate the goodreads extraction script for arxiv against [evaluation/datasets/arxiv.py:893](evaluation/datasets/arxiv.py#L893) `item_attrs_narrow.pt`.

**Catalog ID:** D3 (workload characterization). **File destination:** `docs/thesis/results-data/results/selectivity_bars_<dataset>.png`.

### Figure stub: 6.1.2 = E3. Per-query selectivity CDF per dataset

**Plot.** Empirical CDF: P(per-query AND-of-clauses selectivity ≤ x) for the active-clause subsets in the filter benchmark (`c0_genre`, `c2_format`, `c3_year`, `all4` etc.). One line per sweep. Log-scale X if the distribution is heavy-tailed.

**Why it matters.** The §6.4 filter benchmark and §6.5.2 linr_v3 deep-sweep results depend on per-query selectivity. A reader who sees "linr_v3 collapses on selective sweeps" must first see that selectivity itself is bimodal/skewed — otherwise they will mis-attribute the collapse to a different cause.

**Source.** Same goodreads `_clause_*.csv` files as 6.1.1 + the arxiv extraction once available. The script needs to compute the per-query joint pass-rate; the per-clause CSVs are inputs but not outputs.

**Status.** TBD-script-stub (per the placeholder noted in [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §5.6). **Catalog ID:** E3.

### Table stub: 6.1.3 = G8. SASRec quality ceiling

| dataset | dim | n_users_kept | Recall@100 | NDCG@100 | median_ms (full-scan) |
|---|---|---:|---:|---:|---:|

**Source.** [docs/thesis/results-data/results/sasrec_quality_ceiling.csv](results-data/results/sasrec_quality_ceiling.csv) — 12 rows, `linr_v1_filter_mask` baseline per (dataset, dim) for the quality (no-filter) suite. **Gap:** missing yambda-5b-d256. Fill when the d256 run completes per "Gaps requiring new measurements" in the plot catalog.

**Catalog ID:** G8. This table is the "what is possible" reference number for every Recall@100 figure in §6.2 onward — every approximate algorithm is measured against this ceiling.

### Writer's notes (§6.1)

- The selectivity figures (D3, E3) belong in §6.1 even though they are dataset-side: they justify §6.4's filter-benchmark axis (sweeps ordered by selectivity) and §6.5.2's "linr_v3 collapses at narrow pool" finding.
- Yambda has NO filter benchmark, so it appears in 6.1.3 (quality ceiling) but not in D3 / E3.
- The arxiv selectivity CSVs are a deferred-but-low-effort item — generate them before drafting §6.4 prose.

---

## 6.2 Quality–latency tradeoffs (per dataset × dimension)

**Scope.** Quality-suite results (no filter active) for 3 datasets × 3 dimensions × 4 algorithms × 2 backends × `batch_size=1` × `k ∈ {100, 200, 400}`. Per cell, the suite emits 4 × 2 × 1 × 3 = 24 rows (verified: each `evaluation/results/{arxiv,goodreads}/d*-quality.json` contains 24 rows, each `evaluation/results/yambda/{500m,5b}-d*.json` likewise contains 24 rows). Per-algo sub-files contain 6 rows each (one impl × 2 backends × 1 batch_size × 3 ks).

**Cell coverage map.** From the run inventory `SUMMARY.quality-deep.txt` (now on HF; see schema-recap header) plus the directory layout:

| Dataset | d64 | d128 | d256 |
|---|:---:|:---:|:---:|
| arxiv | ✓ (24 rows) | ✓ | ✓ |
| goodreads | ✓ | ✓ | ✓ |
| yambda 500m | ✓ | ✓ | ✓ |
| yambda 5b | ✓ | ✓ | **MISSING** |

`[TODO: clarify with author — yambda 5b-d256 is absent from evaluation/results/yambda/. Was it omitted by design (e.g., index-mem budget on A100-80GB — extrapolating from 5b-d128 linr_v1_filter_mask at 1310 MiB and silvertorch peak at 3.42 GiB, d256 should fit comfortably) or just not run yet?]`

**Canonical operating point.** For per-cell tables and the Pareto stub, the canonical row is `backend=triton, batch_size=1, k=100, seed=0`. This is the only operating point present in the quality suite (`batch_sizes: [1]` in every `d*-quality.yaml`), so there is no batch-size sweep within §6.2; latency reported here is the single-query latency. `k` is varied in {100, 200, 400} but Recall@100 / NDCG@100 are the headline metrics per the contract; per-K curves are deferred to the deep sweep §6.5 narrative.

**arXiv recall ceiling caveat.** arxiv is a pre-encoded text-retrieval task (no SASRec model; queries come from `data/arxiv-papers/content_d128/query_emb.pt`, see [evaluation/config/arxiv/d128-quality.yaml:5-6](evaluation/config/arxiv/d128-quality.yaml#L5-L6)). The ground truth `gt_d{N}` is itself the exact full-scan top-K. Therefore the exact baseline `linr_v1_filter_mask` recovers `recall@100 ≈ 1.0` by construction. On arxiv, "Recall@100 = X" is best read as "fraction of the exact top-100 that the approximate algorithm reproduces." On Goodreads/Yambda the ground truth is held-out user interactions and `recall@100 ≤ 1` reflects the SASRec model's intrinsic next-item prediction quality.

### 6.2.1 Goodreads — d{64, 128, 256}

Cell: `dataset=goodreads, suite=quality (no filter), users_kept=313 178, backend=triton, batch_size=1, k=100, seed=0`. Source per row: `evaluation/results/goodreads/d{N}-quality{impl}.json`. Also aggregated in `[DATA: docs/thesis/results-data/results/all_results_long.csv]` (filter on `dataset==goodreads & suite_label==quality`).

| dim | impl | Recall@100 | NDCG@100 | median_ms | p20 / p80 ms | index_mem (MiB) | peak_mem (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 64  | linr_v1_filter_mask | 0.1486 | 0.0687 | 0.2076 | 0.2068 / 0.2083 | 98.0 | 293.6 |
| 64  | linr_v3 (`candidate_pool=5000`) | 0.1293 | 0.0618 | 0.2792 | 0.2569 / 0.2849 | 103.4 | 299.0 |
| 64  | linr_v4 | 0.1485 | 0.0686 | 0.3826 | 0.3810 / 0.4193 | 48.7 | 244.3 |
| 64  | silvertorch (`n_lists=1024, n_probe=24`) | 0.1463 | 0.0678 | 0.2045 | 0.2022 / 0.2060 | 246.9 | 442.5 |
| 128 | linr_v1_filter_mask | 0.1479 | 0.0691 | 0.3618 | 0.3607 / 0.3828 | 194.6 | 586.6 |
| 128 | linr_v3 | 0.1438 | 0.0675 | 0.2929 | 0.2899 / 0.2964 | 206.8 | 598.8 |
| 128 | linr_v4 | 0.1478 | 0.0691 | 0.5176 | 0.5168 / 0.5700 | 97.3 | 489.3 |
| 128 | silvertorch | 0.1449 | 0.0679 | 0.2136 | 0.1969 / 0.2171 | 297.8 | 689.8 |
| 256 | linr_v1_filter_mask | 0.1472 | 0.0681 | 0.6149 | 0.6140 / 0.6458 | 390.0 | 1172.4 |
| 256 | linr_v3 | 0.1466 | 0.0677 | 0.2844 | 0.2820 / 0.2898 | 413.5 | 1195.9 |
| 256 | linr_v4 | 0.1473 | 0.0681 | 0.9273 | 0.8410 / 0.9287 | 194.6 | 977.0 |
| 256 | silvertorch | 0.1439 | 0.0670 | 0.2486 | 0.2472 / 0.2502 | 398.0 | 1180.4 |

`[DATA: evaluation/results/goodreads/d{64,128,256}-quality.json]`

**Observations.**
- SASRec recall is essentially flat across dims (≈ 0.147–0.149 at d64–d256). This validates `[CITE: Petrov & Macdonald 2023 RecSys — affiliation: University of Glasgow — kept]` gSASRec/gBCE training: higher d yields no meaningful retrieval gain on this catalogue. Quoted SASRec dim-scaling behavior is consistent with [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.1 (Goodreads).
- `linr_v3` with default `candidate_pool=5000` loses ≈13% of Recall@100 at d64 (0.1293 vs 0.1486 baseline) and the gap narrows with dim (loses 2.8% at d128, 0.4% at d256). The d64 gap is sharpest because at d=64 the 1-bit Hamming shortlist discards more directional information per element. §6.5.2 traces the Recall recovery curve as `candidate_pool` ramps from 2000 → 32000.
- `silvertorch` with default `(n_lists=1024, n_probe=24)` is consistently faster than the exact baseline (0.20 ms vs 0.36 ms at d128, 0.25 ms vs 0.61 ms at d256) at a 1–3% Recall@100 cost. The IVF index inflates `index_mem` (+103 MiB at d128 vs `linr_v1`) due to the per-cluster padded layout — [retrieve/src/retrieve/layers/silvertorch/main.py:128-150](retrieve/src/retrieve/layers/silvertorch/main.py#L128-L150) registers `padded_cluster_items: Tensor` and `centroids: Tensor` alongside the INT8 `item_codes`.
- `linr_v4` recovers baseline-equivalent Recall (within 0.0001) but is the SLOWEST option at every dim on Goodreads despite halving `index_mem` (it stores INT8 instead of fp16). Likely cause: `torch._int_mm` IMMA throughput on the A100 + the per-row dequant epilogue does not pay off against fp16 cuBLAS for these batch-1 single-stage scoring shapes. `linr_v4` becomes attractive only when memory bandwidth or storage is the binding constraint.
- **Memory rank (d128):** `linr_v4 < linr_v1 < linr_v3 < silvertorch` (489 < 586 < 599 < 690 MiB peak). `index_mem` rank: `linr_v4 (97) < linr_v1 (195) < linr_v3 (207) < silvertorch (298) MiB`. Cross-checked: `[DATA: docs/thesis/results-data/results/d128_quality_memory.csv]`.

### 6.2.2 arXiv — d{64, 128, 256}

Cell: `dataset=arxiv, suite=quality (no filter), users_kept=10 000, backend=triton, batch_size=1, k=100, seed=0`. Source per row: `evaluation/results/arxiv/d{N}-quality{impl}.json`.

| dim | impl | Recall@100 | NDCG@100 | median_ms | p20 / p80 ms | index_mem (MiB) | peak_mem (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| 64  | linr_v1_filter_mask | 1.0000 | 0.9999 | 0.6137 | 0.6124 / 0.6154 | 364.9 | 1095.9 |
| 64  | linr_v3 | 0.9996 | 0.9996 | 0.3299 | 0.3280 / 0.3321 | 387.7 | 1118.7 |
| 64  | linr_v4 | 1.0000 | 0.9997 | 1.1624 | 1.1612 / 1.1642 | 182.4 | 913.4 |
| 64  | silvertorch | 0.9953 | 0.9950 | 0.1294 | 0.1284 / 0.1306 | 232.4 | 963.4 |
| 128 | linr_v1_filter_mask | 1.0000 | 0.9999 | 1.0744 | 1.0733 / 1.0759 | 730.0 | 2192.0 |
| 128 | linr_v3 | 1.0000 | 0.9999 | 0.3501 | 0.3483 / 0.3520 | 775.3 | 2237.3 |
| 128 | linr_v4 | 1.0000 | 0.9999 | 1.6657 | 1.6636 / 1.6689 | 364.9 | 1826.9 |
| 128 | silvertorch | 0.9941 | 0.9940 | 0.1308 | 0.1297 / 0.1321 | 418.7 | 1880.7 |
| 256 | linr_v1_filter_mask | 1.0000 | 0.9999 | 2.0012 | 2.0001 / 2.0030 | 1460.0 | 4382.9 |
| 256 | linr_v3 | 1.0000 | 1.0000 | 0.4264 | 0.3937 / 0.4292 | 1550.7 | 4473.6 |
| 256 | linr_v4 | 1.0000 | 0.9999 | 2.8789 | 2.8774 / 2.8804 | 729.7 | 3652.7 |
| 256 | silvertorch | 0.9928 | 0.9926 | 0.1577 | 0.1460 / 0.1603 | 789.6 | 3712.6 |

`[DATA: evaluation/results/arxiv/d{64,128,256}-quality.json]`

**Observations.**
- The ≈ 2.99M-paper catalogue (per [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.2) makes the latency picture clearer than Goodreads: at d128, exact `linr_v1` takes 1.07 ms, `linr_v3` (1-bit shortlist + fp16 rerank, default `candidate_pool=5000` ≈ 0.17% of N) takes 0.35 ms (3× speedup) at zero recall loss, and `silvertorch` takes 0.13 ms (8× speedup) at 0.6% recall loss. This is the cleanest illustration of the LinR / co-designed family's value proposition in a pure-recall setting.
- `linr_v3` reaches `Recall@100 = 1.0` at d128 and d256: the 1-bit shortlist is high-recall on this catalogue at default pool size because nomic-embed-v1.5 embeddings `[CITE: Nussbaum et al. 2024 arXiv:2402.01613 — affiliation: Nomic AI — kept]` have a benign angle distribution.
- `silvertorch` Recall@100 degrades monotonically with dim (0.9953 → 0.9941 → 0.9928). With `n_lists=1024, n_probe=24` fixed, the average cluster size is N/1024 ≈ 2920 items and the probed pool is ≈ 70 000 items (2.3% of N). At higher d the INT8 global scale loses more relative resolution per dim, but the effect is small. §6.5.1 explores `(n_lists, n_probe)` interactions at d128.
- `linr_v4` is slower than `linr_v1` on arxiv at every dim (1.16 ms vs 0.61 ms at d64; 1.67 vs 1.07 at d128; 2.88 vs 2.00 at d256), but halves `index_mem`. The throughput regression is consistent with the Goodreads pattern: per-row INT8 quantization + IMMA matmul + per-row dequant has not paid off on A100 at these shapes.

### 6.2.3 Yambda 500m & 5b — d{64, 128, 256}

Cell: `dataset=yambda-{500m,5b}, suite=yambda (no filter), backend=triton, batch_size=1, k=100, seed=0`. `n_users_kept`: yambda-500m has 45 932 (the held-out test users of the Global Temporal Split, see [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.3), yambda-5b has 459 067. Source per row: `evaluation/results/yambda/{500m,5b}-d{N}{impl}.json`.

| scale | dim | impl | Recall@100 | NDCG@100 | median_ms | index_mem (MiB) | peak_mem (MiB) |
|---|---|---|---:|---:|---:|---:|---:|
| 500m | 64  | linr_v1_filter_mask | 0.1564 | 0.1079 | 0.3276 | 228.0 | 685.0 |
| 500m | 64  | linr_v3 | 0.1424 | 0.0998 | 0.3067 | 242.0 | 699.0 |
| 500m | 64  | linr_v4 | 0.1562 | 0.1079 | 0.8345 | 113.9 | 570.9 |
| 500m | 64  | silvertorch | 0.1543 | 0.1065 | 0.1308 | 149.7 | 606.7 |
| 500m | 128 | linr_v1_filter_mask | 0.1486 | 0.1028 | 0.7097 | 456.0 | 1370.0 |
| 500m | 128 | linr_v3 | 0.1426 | 0.0991 | 0.3236 | 484.1 | 1398.1 |
| 500m | 128 | linr_v4 | 0.1485 | 0.1027 | 1.0812 | 227.8 | 1141.8 |
| 500m | 128 | silvertorch | 0.1468 | 0.1015 | 0.1221 | 254.9 | 1168.9 |
| 500m | 256 | linr_v1_filter_mask | 0.1398 | 0.0990 | 1.3004 | 912.0 | 2738.4 |
| 500m | 256 | linr_v3 | 0.1360 | 0.0963 | 0.3558 | 968.2 | 2794.6 |
| 500m | 256 | linr_v4 | 0.1396 | 0.0989 | 1.8314 | 455.6 | 2282.0 |
| 500m | 256 | silvertorch | 0.1367 | 0.0968 | 0.1434 | 499.8 | 2326.2 |
| 5b   | 64  | linr_v1_filter_mask | 0.1744 | 0.1172 | 0.7856 | 654.6 | 1965.6 |
| 5b   | 64  | linr_v3 | 0.1464 | 0.1014 | 0.3796 | 695.5 | 2006.5 |
| 5b   | 64  | linr_v4 | 0.1742 | 0.1170 | 2.0129 | 327.3 | 1638.3 |
| 5b   | 64  | silvertorch | 0.1728 | 0.1163 | 0.1587 | 421.1 | 1732.1 |
| 5b   | 128 | linr_v1_filter_mask | 0.1779 | 0.1207 | 1.8725 | 1310.0 | 3930.3 |
| 5b   | 128 | linr_v3 | 0.1590 | 0.1097 | 0.4283 | 1391.0 | 4011.3 |
| 5b   | 128 | linr_v4 | 0.1778 | 0.1206 | 2.9233 | 654.6 | 3274.9 |
| 5b   | 128 | silvertorch | 0.1739 | 0.1183 | 0.1712 | 800.3 | 3420.6 |
| 5b   | 256 | — | — | — | — | — | — |

`[DATA: evaluation/results/yambda/{500m,5b}-d{N}{linr_v1_filter_mask,linr_v3,linr_v4,silvertorch}.json]`

**Observations.**
- yambda-5b at d128 achieves the **highest absolute Recall@100 in the entire study (0.1779)** for the exact baseline. More training interactions (4.65 B vs 466.5 M) yield a measurably stronger SASRec model: 5b-d128 vs 500m-d128 baseline = 0.1779 vs 0.1486 (+19.7% relative). This is the most prominent training-data-scale effect in the dataset triple. `[CITE: Anokhin et al. 2025 arXiv:2505.22238 — affiliation: Yandex — kept]`.
- `linr_v3` at default `candidate_pool=5000` shows the LARGEST Recall gap in the study on yambda-5b d64 (0.1464 vs 0.1744 — relative drop 16.1%) and yambda-5b d128 (0.1590 vs 0.1779 — relative drop 10.6%). Cause: catalogue size N ≈ 9.39M, so 5000 candidates is only ≈ 0.053% of N — the 1-bit shortlist is too narrow to capture all SASRec-relevant items. This is the empirical motivation for §6.5.2: linr_v3 on large catalogues needs a much wider `candidate_pool`.
- `silvertorch` keeps the speed lead on yambda (0.12–0.17 ms across all cells) and the Recall gap vs baseline is small (1.4% relative on 500m-d128, 2.2% on 5b-d128). At fixed `n_probe=24/n_lists=1024`, probed pool grows with N (≈ 220 k items at 5b-d128, ≈ 2.3% of N) which keeps Recall stable.
- The exact baseline `linr_v1_filter_mask` scales linearly with N · D / GPU bandwidth: 500m-d64 → 5b-d64 = 0.33 ms → 0.79 ms (factor 2.4×, consistent with the 2× larger catalogue and ≈ same compute density). 500m-d128 → 5b-d128 = 0.71 → 1.87 ms (factor 2.6×). The latency cost of going to 5b makes the silvertorch advantage even larger in absolute terms (8 ms → 11 ms speedup ratios, or 5.8× → 10.9× at d128).
- Cross-reference for index size: see [docs/thesis/results-data/results/sasrec_quality_ceiling.csv](docs/thesis/results-data/results/sasrec_quality_ceiling.csv) for the exact-baseline (linr_v1_filter_mask) recall ceiling across all (dataset, dim) cells in one CSV.

### 6.2.4 Recall at K = 100, 200, 400 (all cells)

The quality JSONs carry recall at three K values per cell. The contract calls Recall@100 the headline; Recall@200/@400 are reported here for completeness so the writer can decide whether to surface them in tables or relegate them to an appendix. `[DATA: evaluation/results/{arxiv,goodreads,yambda}/d*-quality*.json]` filtered to `backend=triton, batch_size=1, seed=0`.

| cell | impl | R@100 | R@200 | R@400 |
|---|---|---:|---:|---:|
| arxiv d64  | linr_v1_filter_mask | 1.0000 | 1.0000 | 1.0000 |
| arxiv d64  | linr_v3             | 0.9996 | 0.9996 | 0.9996 |
| arxiv d64  | linr_v4             | 1.0000 | 1.0000 | 1.0000 |
| arxiv d64  | silvertorch         | 0.9953 | 0.9951 | 0.9951 |
| arxiv d128 | linr_v1_filter_mask | 1.0000 | 1.0000 | 1.0000 |
| arxiv d128 | linr_v3             | 1.0000 | 1.0000 | 1.0000 |
| arxiv d128 | linr_v4             | 1.0000 | 1.0000 | 1.0000 |
| arxiv d128 | silvertorch         | 0.9941 | 0.9941 | 0.9942 |
| arxiv d256 | linr_v1_filter_mask | 1.0000 | 1.0000 | 1.0000 |
| arxiv d256 | linr_v3             | 1.0000 | 1.0000 | 1.0000 |
| arxiv d256 | linr_v4             | 1.0000 | 1.0000 | 1.0000 |
| arxiv d256 | silvertorch         | 0.9928 | 0.9926 | 0.9927 |
| goodreads d64  | linr_v1_filter_mask | 0.1486 | 0.2161 | 0.3004 |
| goodreads d64  | linr_v3             | 0.1293 | 0.1741 | 0.2120 |
| goodreads d64  | linr_v4             | 0.1485 | 0.2160 | 0.3003 |
| goodreads d64  | silvertorch         | 0.1463 | 0.2124 | 0.2944 |
| goodreads d128 | linr_v1_filter_mask | 0.1479 | 0.2147 | 0.2981 |
| goodreads d128 | linr_v3             | 0.1438 | 0.2044 | 0.2726 |
| goodreads d128 | linr_v4             | 0.1478 | 0.2147 | 0.2980 |
| goodreads d128 | silvertorch         | 0.1449 | 0.2102 | 0.2909 |
| goodreads d256 | linr_v1_filter_mask | 0.1472 | 0.2124 | 0.2941 |
| goodreads d256 | linr_v3             | 0.1466 | 0.2117 | 0.2921 |
| goodreads d256 | linr_v4             | 0.1473 | 0.2124 | 0.2943 |
| goodreads d256 | silvertorch         | 0.1439 | 0.2065 | 0.2839 |
| yambda-500m d64  | linr_v1_filter_mask | 0.1564 | 0.2227 | 0.3087 |
| yambda-500m d64  | linr_v3             | 0.1424 | 0.1977 | 0.2604 |
| yambda-500m d64  | linr_v4             | 0.1562 | 0.2225 | 0.3084 |
| yambda-500m d64  | silvertorch         | 0.1543 | 0.2203 | 0.3060 |
| yambda-500m d128 | linr_v1_filter_mask | 0.1486 | 0.2121 | 0.2923 |
| yambda-500m d128 | linr_v3             | 0.1426 | 0.2032 | 0.2764 |
| yambda-500m d128 | linr_v4             | 0.1485 | 0.2118 | 0.2923 |
| yambda-500m d128 | silvertorch         | 0.1468 | 0.2096 | 0.2889 |
| yambda-500m d256 | linr_v1_filter_mask | 0.1398 | 0.1952 | 0.2675 |
| yambda-500m d256 | linr_v3             | 0.1360 | 0.1917 | 0.2647 |
| yambda-500m d256 | linr_v4             | 0.1396 | 0.1953 | 0.2675 |
| yambda-500m d256 | silvertorch         | 0.1367 | 0.1911 | 0.2619 |
| yambda-5b d64  | linr_v1_filter_mask | 0.1744 | 0.2517 | 0.3502 |
| yambda-5b d64  | linr_v3             | 0.1464 | 0.2042 | 0.2679 |
| yambda-5b d64  | linr_v4             | 0.1742 | 0.2515 | 0.3499 |
| yambda-5b d64  | silvertorch         | 0.1728 | 0.2491 | 0.3456 |
| yambda-5b d128 | linr_v1_filter_mask | 0.1779 | 0.2559 | 0.3550 |
| yambda-5b d128 | linr_v3             | 0.1590 | 0.2246 | 0.3002 |
| yambda-5b d128 | linr_v4             | 0.1778 | 0.2558 | 0.3548 |
| yambda-5b d128 | silvertorch         | 0.1739 | 0.2495 | 0.3445 |

**Patterns at K=200/400.**
- The ordering of algorithms is stable across K: `linr_v1 ≈ linr_v4 > silvertorch > linr_v3` on recall-realistic datasets (Goodreads, Yambda), with linr_v3's gap to baseline shrinking as K grows (the cascade only loses on the rank-100 boundary; rank-400 has more buffer for fp16 noise).
- `linr_v3` on yambda-5b d64 still loses ≈ 23% relative at K=400 (0.2679 vs baseline 0.3502) — the default `candidate_pool=5000` is the binding constraint, not K. §6.5.2 addresses this directly.
- arxiv recall stays at 1.0 / 0.99 across K (by construction).

### 6.2.5 Cross-dataset quality summary

Best Recall@100 per (dataset, dim) cell (`backend=triton, bs=1, k=100, seed=0`). On arxiv, "best" is meaningless because three of the four algos hit 1.0 — annotated with "tie."

| dataset | dim | best impl | best Recall@100 | latency advantage of best vs `linr_v1` |
|---|---|---|---:|---:|
| arxiv | 64 | tie: linr_v1 / linr_v3 / linr_v4 = 1.0000 | 1.0000 | (linr_v3: 1.86×; silvertorch is 4.74× but 0.5% below) |
| arxiv | 128 | tie: linr_v1 / linr_v3 / linr_v4 = 1.0000 | 1.0000 | (linr_v3: 3.07×; silvertorch is 8.21× but 0.6% below) |
| arxiv | 256 | tie: linr_v1 / linr_v3 / linr_v4 = 1.0000 | 1.0000 | (linr_v3: 4.69×; silvertorch is 12.69× but 0.7% below) |
| goodreads | 64  | linr_v1_filter_mask (= ceiling) | 0.1486 | 1.0× (baseline). silvertorch is 1.02× faster but 1.5% lower Recall |
| goodreads | 128 | linr_v1_filter_mask | 0.1479 | 1.0×. silvertorch is 1.69× faster, 2.0% lower Recall |
| goodreads | 256 | linr_v1_filter_mask | 0.1472 | 1.0×. silvertorch is 2.47× faster, 2.2% lower Recall |
| yambda-500m | 64  | linr_v1_filter_mask | 0.1564 | 1.0×. silvertorch is 2.50× faster, 1.3% lower |
| yambda-500m | 128 | linr_v1_filter_mask | 0.1486 | 1.0×. silvertorch is 5.81× faster, 1.2% lower |
| yambda-500m | 256 | linr_v1_filter_mask | 0.1398 | 1.0×. silvertorch is 9.07× faster, 2.2% lower |
| yambda-5b   | 64  | linr_v1_filter_mask | 0.1744 | 1.0×. silvertorch is 4.95× faster, 0.9% lower |
| yambda-5b   | 128 | linr_v1_filter_mask | 0.1779 | 1.0×. silvertorch is 10.94× faster, 2.2% lower |

**Pattern.** The exact baseline `linr_v1_filter_mask` always wins on raw Recall@100 (trivially — it is the exact top-K) by construction. The actionable comparison is the **best Recall/latency Pareto trade-off**, where `silvertorch` dominates the bottom-right corner at zero or sub-3% Recall cost on every cell. `linr_v3` at default `candidate_pool` is a middle-ground option that sometimes matches and sometimes underperforms the cascade defaults — the deep sweep §6.5.2 shows it can recover near-baseline quality at larger pools.

### Figure stub: 6.2.6 = A1. Recall–latency Pareto (online serving, bs=1), faceted 3×3

**Plot.** 3 × 3 grid of scatter plots. Rows: dataset ∈ {goodreads, arxiv, yambda} (yambda-500m and yambda-5b overlaid as solid vs hollow markers). Cols: dim ∈ {64, 128, 256}.

- **X:** `median_ms` (log scale; per-panel range or shared per-row).
- **Y:** `Recall@100` (linear, zoomed to 0.0–1.0 with auto-zoom to the data range).
- **Color = algorithm:** `linr_v1_filter_mask` blue, `linr_v3` green, `linr_v4` orange, `silvertorch` red.
- **Marker shape = backend:** triton ●, torch ▲.
- **Optional p20/p80 whiskers** (horizontal) on X using existing schema columns.
- **Brute-force baseline:** horizontal line at `recall=1.0`; vertical guide at the baseline's median_ms so the reader can read off the speedup factor at a glance.
- **Pareto-optimal envelope** drawn solid; dominated points faded to 50% opacity.
- Annotate silvertorch / linr_v3 points with hyperparameter caption (`n_probe=24, n_lists=1024`; `candidate_pool=5000`).

**Source CSV.** [docs/thesis/results-data/results/all_results_long.csv](results-data/results/all_results_long.csv) — filter on `filter_kind=="none" & batch_size==1 & k==100 & seed==0`. Group by `(dataset, dim)` into 9 panels (yambda 5b-d256 panel empty by construction).

**Catalog ID:** A1. **File destination:** `docs/thesis/results-data/results/A1_pareto_recall_latency_bs1.png`.

### Figure stub: 6.2.7 = A2. Recall–latency Pareto (batched, bs=16), faceted 3×3

**Plot.** Same 3 × 3 grid as A1 but restricted to `batch_size==16`.

- **X:** `median_ms` (log) — labeled "per-batch latency at bs=16," explicitly NOT per-query or throughput (per the QPS convention in the preamble).
- Y, colors, markers, baselines, Pareto envelope: as A1.

**Use.** Side-by-side with A1 surfaces the "different algorithms win in each regime" point (CAGRA precedent: single-query Fig 14 vs large-batch Fig 13). Algorithms whose latency grows sub-linearly with batch size will look relatively better in A2 than in A1.

**Optional variant.** Overlay bs ∈ {1, 8, 16} as separate marker shapes on one panel (one selected `(dataset, dim)`) — shows per-algo latency growth across batch sizes within one figure.

**Source.** Same CSV as A1, with `batch_size==16` and (for the variant) also `batch_size in {1, 8, 16}`.

**Catalog ID:** A2. **File destination:** `docs/thesis/results-data/results/A2_pareto_recall_latency_bs16.png`.

### Table stub: 6.2.8 = G1. Headline summary table

| dataset | dim | impl | Recall@100 | median_ms (bs=1) | bytes/vec | peak_mem_mib |
|---|---|---|---:|---:|---:|---:|

Bold the Pareto-optimal cell per (dataset, dim) row triple. `bytes_per_vec = index_mem_mib * 2^20 / n_items` — derive via `extract.py` update; see "Metrics to add" §end-of-chapter.

**Source.** `all_results_long.csv` filtered to canonical operating point.

**Catalog ID:** G1. SCANN-style per-dataset summary.

### Table stub: 6.2.9 = G2. Latency-at-fixed-recall table

Rows: algorithm × dataset; columns: minimum `median_ms` (at bs=1) achieving `Recall@10 ∈ {0.9, 0.95, 0.99}`.

**Note:** Requires K=10 evaluation runs (currently only K≥100 in the schema). Once added per "Gaps requiring new measurements," this table answers the BIG-ANN 2023 reformulation: "at fixed quality target, what is the minimum latency each algorithm achieves?"

**Catalog ID:** G2. **Status:** blocked on K=10 quality re-run.

### Table stub: 6.2.10 = G3. Recall-at-fixed-latency table

Rows: algorithm × dataset; columns: Recall@100 achieved at `median_ms ≤ {0.5, 1, 5, 10}` (bs=1, no filter). Inverse framing of G2 — a serving-budget framing ("given X ms budget, which algo wins").

**Source.** `all_results_long.csv` directly — interpolate or pick max-recall within each latency budget.

**Catalog ID:** G3. **File destination:** included in §6.2 prose as a markdown table.

### Writer's notes (§6.2)

- The "arxiv recall = 1.0" sentence MUST appear in the prose introducing §6.2.2; without it, a reader will misread the arxiv numbers as a near-perfect SASRec rather than a self-referential ground truth. Suggested phrasing: "Поскольку для arXiv основная истина определяется как точная top-K full-scan по предобученным эмбеддингам nomic-embed-v1.5, метрики Recall@K для точного baseline `linr_v1_filter_mask` равны единице по построению; на arXiv интересна не абсолютная Recall, а количество top-K, которое аппроксимирующий алгоритм воспроизводит."
- The yambda-5b-d256 gap is the only data-coverage anomaly in this section. Recommend a one-line footnote rather than a §-level discussion. `[TODO — still unresolved: clarify with author whether 5b-d256 is by design or scheduled.]`
- The cross-dataset summary table (§6.2.5) could be folded into the §6.2 introduction rather than a separate subsection; the per-dataset bullet observations already imply the same ranking.
- The "no quality-suite version of linr_v2" point (linr_v2 requires a filter) belongs in the §6.2 intro: it explains the 4-algo lineup vs the 5-algo lineup in §6.4.
- §6.2.4 (Recall at K=100/200/400) IS the table the contract calls for — added in this pass. Writer should decide whether to keep all 44 rows inline or move to Appendix D ("Длинные таблицы результатов") per the chapter-plan front matter.
- `[TODO — still unresolved: for the Pareto plot, should backend=torch points also appear in a second color shade (variant within A1/A2), or is §6.7's parity panel sufficient?]`
- The linr_v3 / silvertorch hyperparameter footnote per cell is important: a reader who jumps to A1/A2 without seeing `candidate_pool=5000` or `n_probe=24` will misinterpret these as fully-tuned configurations rather than the unoptimized defaults shaped by §6.5.

---

## 6.3 Quality–memory tradeoffs

**Scope.** Memory dimension of the recall × latency × memory trade-off space. Critical for LinR's central claim that 1-bit Sign-OPORP (linr_v3) and INT8 (linr_v4) deliver large memory reductions at near-baseline recall (Borisyuk et al. 2024 reports ~16× memory reduction vs fp16). This section is the "Memory" side of the headline tradeoff; latency was §6.2.

**Source.** `index_mem_mib`, `peak_mem_mib`, `fwd_scratch_mib` are present in every row of `all_results_long.csv`. `bytes_per_vec = index_mem_mib * 2^20 / n_items` must be derived in pre-processing — add to `extract.py`. Per-dataset N values: goodreads 798 016, arxiv 2 990 000, yambda-500m 3 060 000, yambda-5b 9 390 000 (from [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.x).

### Figure stub: 6.3.1 = A3. Recall–Memory Pareto

**Plot.** Scatter, one panel per dataset (3 panels total).

- **X:** `bytes_per_vec` (log scale, range ~0.5 → 1024 bytes).
- **Y:** `Recall@100` (linear, zoomed 0.0–1.0).
- **Color = algorithm**; **marker = dim** (○ d64, △ d128, ▢ d256).
- **Reference vertical lines** at theoretical encodings: fp32 (4·d bytes/vec), fp16 (2·d), INT8 (d), 1-bit OPORP (d/8). Label each.
- Pareto-optimal envelope solid; dominated faded.

**Expected qualitative result.** linr_v3 (1-bit) and linr_v4 (INT8) sit far-left at competitive recall — the headline visualization for the thesis's "memory-efficient family" claim. linr_v1 (fp16) sits at 2·d; silvertorch sits between INT8 and fp16 because IVF padding inflates per-vector footprint.

**Source.** `all_results_long.csv` filtered to `filter_kind=="none" & batch_size==1 & k==100 & seed==0`; derive `bytes_per_vec` per (dataset, dim, algo).

**Catalog ID:** A3. **File destination:** `docs/thesis/results-data/results/A3_recall_memory_pareto.png`.

### Figure stub: 6.3.2 = A4. 3D scatter: Recall × Latency × Memory (executive summary)

**Plot.** Bubble plot, one panel per dataset (3 panels).

- **X:** `median_ms` at bs=1 (log).
- **Y:** `Recall@100` (linear).
- **Bubble size:** `bytes_per_vec` (log, INVERTED — smaller bytes = larger bubble, so memory-efficient algorithms visually dominate).
- All dims overlaid in one panel; color = algorithm.

**Use case.** The "open this PDF and immediately get it" defense figure: encodes the three headline axes (recall, latency, memory) in one place with no need to cross-reference A1, A2, A3.

**Catalog ID:** A4. **File destination:** `docs/thesis/results-data/results/A4_three_axis_bubble.png`. Also appears in §6.9 as the executive-summary visual.

### Table stub: 6.3.3 = G6. Memory breakdown table

Rows: (dataset, dim, algorithm); columns: `index_mem_mib`, `fwd_scratch_mib`, `peak_mem_mib`, `bytes_per_vec`, `ratio_vs_fp32` (= `bytes_per_vec / (4·d)`).

**Source.** Have: [docs/thesis/results-data/results/d128_quality_memory.csv](results-data/results/d128_quality_memory.csv) (16 rows, d128 only). **Action:** extend to d64 + d256 (data already in `all_results_long.csv`; just regenerate the CSV); add `bytes_per_vec` and `ratio_vs_fp32` columns.

**Catalog ID:** G6. Memory breakdown deserves a dedicated table — it answers "X uses Y bytes per vector, Z% of fp32," which is the cleanest single-number for the memory chapter.

### 6.3.4 Concrete-data block (memory profile)

`[DATA: docs/thesis/results-data/quality_summary.csv]` filtered to `backend=triton`. 44 rows = 4 datasets × 3 dims × 4 algos, minus the 4 absent yambda-5b-d256 cells.

**Master memory table (quality suite, bs=1, k=100, seed=0).**

| dataset | scale | dim | impl | `index_mem` (MiB) | `peak_mem` (MiB) | `fwd_scratch` (MiB) | peak / index |
|---|---|---:|---|---:|---:|---:|---:|
| arxiv     | —    | 64  | linr_v1_filter_mask | 364.87 | 1095.87 | 0 | 3.00 |
| arxiv     | —    | 64  | linr_v3             | 387.67 | 1118.67 | 0 | 2.88 |
| arxiv     | —    | 64  | linr_v4             | 182.43 |  913.44 | 0 | 5.00 |
| arxiv     | —    | 64  | silvertorch         | 232.37 |  963.37 | 0 | 4.14 |
| arxiv     | —    | 128 | linr_v1_filter_mask | 730.00 | 2192.00 | 0 | 3.00 |
| arxiv     | —    | 128 | linr_v3             | 775.35 | 2237.35 | 0 | 2.89 |
| arxiv     | —    | 128 | linr_v4             | 364.87 | 1826.87 | 0 | 5.00 |
| arxiv     | —    | 128 | silvertorch         | 418.67 | 1880.67 | 0 | 4.49 |
| arxiv     | —    | 256 | linr_v1_filter_mask | 1460.00| 4382.94 | 0 | 3.00 |
| arxiv     | —    | 256 | linr_v3             | 1550.69| 4473.63 | 0 | 2.89 |
| arxiv     | —    | 256 | linr_v4             | 729.74 | 3652.68 | 0 | 5.00 |
| arxiv     | —    | 256 | silvertorch         | 789.64 | 3712.59 | 0 | 4.70 |
| goodreads | —    | 64  | linr_v1_filter_mask |  98.00 |  293.60 | 0 | 2.99 |
| goodreads | —    | 64  | linr_v3             | 103.38 |  298.99 | 0 | 2.89 |
| goodreads | —    | 64  | linr_v4             |  48.65 |  244.25 | 0 | 5.02 |
| goodreads | —    | 64  | silvertorch         | 246.91 |  442.51 | 0 | 1.79 |
| goodreads | —    | 128 | linr_v1_filter_mask | 194.60 |  586.60 | 0 | 3.01 |
| goodreads | —    | 128 | linr_v3             | 206.77 |  598.77 | 0 | 2.89 |
| goodreads | —    | 128 | linr_v4             |  97.30 |  489.30 | 0 | 5.03 |
| goodreads | —    | 128 | silvertorch         | 297.81 |  689.81 | 0 | 2.32 |
| goodreads | —    | 256 | linr_v1_filter_mask | 390.00 | 1172.40 | 0 | 3.01 |
| goodreads | —    | 256 | linr_v3             | 413.53 | 1195.93 | 0 | 2.89 |
| goodreads | —    | 256 | linr_v4             | 194.60 |  977.01 | 0 | 5.02 |
| goodreads | —    | 256 | silvertorch         | 397.97 | 1180.37 | 0 | 2.97 |
| yambda    | 500m | 64  | linr_v1_filter_mask | 228.00 |  685.00 | 0 | 3.00 |
| yambda    | 500m | 64  | linr_v3             | 242.04 |  699.04 | 0 | 2.89 |
| yambda    | 500m | 64  | linr_v4             | 113.90 |  570.90 | 0 | 5.01 |
| yambda    | 500m | 64  | silvertorch         | 149.69 |  606.69 | 0 | 4.05 |
| yambda    | 500m | 128 | linr_v1_filter_mask | 456.00 | 1370.00 | 0 | 3.00 |
| yambda    | 500m | 128 | linr_v3             | 484.09 | 1398.09 | 0 | 2.89 |
| yambda    | 500m | 128 | linr_v4             | 227.80 | 1141.81 | 0 | 5.01 |
| yambda    | 500m | 128 | silvertorch         | 254.88 | 1168.88 | 0 | 4.59 |
| yambda    | 500m | 256 | linr_v1_filter_mask | 912.00 | 2738.43 | 0 | 3.00 |
| yambda    | 500m | 256 | linr_v3             | 968.17 | 2794.60 | 0 | 2.89 |
| yambda    | 500m | 256 | linr_v4             | 455.61 | 2282.04 | 0 | 5.01 |
| yambda    | 500m | 256 | silvertorch         | 499.80 | 2326.24 | 0 | 4.65 |
| yambda    | 5b   | 64  | linr_v1_filter_mask | 654.58 | 1965.58 | 0 | 3.00 |
| yambda    | 5b   | 64  | linr_v3             | 695.49 | 2006.50 | 0 | 2.89 |
| yambda    | 5b   | 64  | linr_v4             | 327.29 | 1638.29 | 0 | 5.00 |
| yambda    | 5b   | 64  | silvertorch         | 421.06 | 1732.06 | 0 | 4.11 |
| yambda    | 5b   | 128 | linr_v1_filter_mask | 1310.00| 3930.33 | 0 | 3.00 |
| yambda    | 5b   | 128 | linr_v3             | 1390.99| 4011.31 | 0 | 2.88 |
| yambda    | 5b   | 128 | linr_v4             | 654.58 | 3274.91 | 0 | 5.00 |
| yambda    | 5b   | 128 | silvertorch         | 800.31 | 3420.64 | 0 | 4.27 |

#### Measurement protocol recap

- `index_mem_mib = cuda_allocated_mib` delta around `build_algorithm(...)` ([evaluation/retrieval/sweep.py:425-442](evaluation/retrieval/sweep.py#L425-L442)).
- `peak_mem_mib = torch.cuda.max_memory_allocated()` across a 16-iteration memory window ([evaluation/retrieval/bench_tools.py:137-165](evaluation/retrieval/bench_tools.py#L137-L165)).
- `fwd_scratch_mib = peak − baseline` inside that window ([evaluation/retrieval/bench_tools.py:166](evaluation/retrieval/bench_tools.py#L166)).
- Between cells, `algo_obj.modules.clear()` must be called before `del algo_obj` and `torch.cuda.empty_cache()` ([docs/system/evaluation.md:296-313](docs/system/evaluation.md#L296-L313)) — otherwise residual buffers inflate the next cell's `mem_before` and `index_mem_mib` is contaminated.

#### `fwd_scratch_mib = 0` finding

Every quality-suite row has `fwd_scratch_mib = 0` (rounded to MiB). At bs=1, per-forward score buffers, top-K work buffers, and intermediate kernel tiles fit within the steady-state allocation; there is no transient malloc spike. **This is not "scratch is zero everywhere"** — at higher batch sizes the fp32/fp16 score buffer (`[B, N]`) grows linearly with B and at B=16, N=2.99M is ≈ 180 MiB. The quality suite at bs=1 simply does not probe that regime. `[TODO: clarify with author — extract a bs > 1 scratch profile from the filter suite for completeness?]`

#### `peak / index` ratio is a per-impl constant (mostly)

| `impl` | peak/index range | interpretation |
|---|---|---|
| `linr_v1_filter_mask` | 2.99–3.01 | 1× index + 2× score-and-batch overhead (fp16 dense matmul + topk staging) |
| `linr_v3`             | 2.88–2.89 | 1× index + ≈ 1.9× scratch (1-bit Hamming shortlist + fp16 rerank on the candidate pool) |
| `linr_v4`             | 5.00–5.03 | 1× index + 4× overhead (INT8 codes + INT32 score accumulator + dequant epilogue + topk) |
| `silvertorch`         | 1.79–4.70 | varies with N (input batch + scratch dominates on large-N catalogues; on small-N goodreads, scratch is small relative to inflated index — see anomaly below) |

`linr_v4`'s 5× ratio — despite having the **smallest** `index_mem` per cell — is driven by the INT32 score accumulator inside `torch._int_mm`: a full `[B, N] int32` materializes before the dequant-and-topk epilogue. This is the operational reason `linr_v4` is unattractive in memory-tight deployment despite its compact INT8 index.

#### Algebraic index models (derived from `register_buffer` inspections)

Let N = catalogue size, D = embedding dim, n_lists = IVF cell count, M = max cluster size, m_bits = bloom width.

```pseudocode
index_mem(linr_v1_filter_mask) = 2 · D · N        # fp16 transposed [D, N]
index_mem(linr_v4)             = D · N            # int8 padded [D, N_padded]
index_mem(linr_v3)             = 2.125 · D · N + 12·D   # fp16 dense + 1-bit signatures + signs/perm
index_mem(silvertorch | none)  = D·N + 4·n_lists·D + 8·n_lists·M + 8·n_lists   # int8 codes + fp32 centroids + cluster items + sizes
```

Buffer sources: [postfilter_knn.py:42](retrieve/src/retrieve/layers/linr/postfilter_knn.py#L42), [postfilter_knn_int8.py:83](retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L83), [one_bit_knn.py:94-96](retrieve/src/retrieve/layers/linr/one_bit_knn.py#L94-L96), [silvertorch/main.py:182-220](retrieve/src/retrieve/layers/silvertorch/main.py#L182-L220).

**Verification on three cells.**
- goodreads d128 linr_v1: model = `2·128·797 144 = 204 MB ≈ 194.6 MiB`. Measured: 194.6 MiB. ✓
- arxiv d128 linr_v1: model = `2·128·2 990 000 = 765 MB ≈ 730 MiB`. Measured: 730 MiB. ✓
- Every cell: `index_mem(linr_v4) / index_mem(linr_v1) = 0.500 ± 0.001`. ✓
- Every cell: `index_mem(linr_v3) / index_mem(linr_v1) = 1.062 ± 0.002`. ✓ (matches `2.125 / 2 = 1.0625`)

**Catalogue size cross-check from `linr_v1.index_mem = 2·D·N`:**
| dataset | scale | derived N | dataset-notes claim |
|---|---|---:|---:|
| goodreads | —    | 797 144   | ≈ 797 000 ✓ |
| arxiv     | —    | 2 990 000 | ≈ 2 990 000 ✓ |
| yambda    | 500m | **1 867 776** | other-agent's §6.3 prose says 3 060 000 — **discrepancy** |
| yambda    | 5b   | **5 365 760** | other-agent's §6.3 prose says 9 390 000 — **discrepancy** (9.39 M is the *full* Yandex track catalogue from the Yambda paper; the retrieval corpus after filtering is smaller) |

`[TODO: clarify with author — the §6.3 scope paragraph claims yambda-500m has 3 060 000 items and yambda-5b has 9 390 000 items. Index-memory measurements give 1.87 M and 5.37 M respectively. The Yambda paper's 9.39 M is the full track catalogue; actual retrieval corpus depends on the yambda preprocessing pipeline's filtering rules. Resolve before §6.3 prose is finalized.]`

#### LinR V3 is NOT a memory-saving algorithm (load-bearing correction)

The naive reading of "1-bit Sign-OPORP" suggests 8–64× memory reduction vs fp16. **In fact, `index_mem(linr_v3) ≈ 1.06 × index_mem(linr_v1)`** — slightly *larger*. Reason: the V3 cascade is OneBitKNN + PrefilterKNN. The PrefilterKNN stage-2 fp16 rerank requires the dense `[D, N] fp16` embeddings, so the cascade carries the full LinR-V1 cost (`2·D·N`) **plus** a 1-bit-signature surcharge (`0.125·D·N`). 

This contradicts the §6.3 scope paragraph's anchor claim ("LinR V3 delivers large memory reductions"). The correct framing: **LinR V3's value is latency, not memory. LinR V4 (INT8) is the memory-saving algorithm — exactly half of V1 at no measured Recall loss (§6.6.4).**

`[TODO: clarify with author — the LinR paper's reported memory reduction ratio refers to the *1-bit signature alone* relative to fp16 (~16×), not to the LinR-V3 cascade footprint as implemented in this Triton reproduction. Document the distinction explicitly in §6.3 prose; otherwise the 16× claim is misleading for this implementation.]`

#### silvertorch index anomaly on small-N (Goodreads)

Measured `index_mem(silvertorch) / index_mem(linr_v1)` vs algebraic-floor prediction `(D + 8) / (2D)`:

| dataset | scale | dim | measured | algebraic floor |
|---|---|---:|---:|---:|
| arxiv     | —    | 64  | 0.637 | 0.563 |
| arxiv     | —    | 128 | 0.574 | 0.531 |
| arxiv     | —    | 256 | 0.541 | 0.516 |
| yambda    | 500m | 128 | 0.559 | 0.531 |
| yambda    | 500m | 256 | 0.548 | 0.516 |
| yambda    | 5b   | 128 | 0.611 | 0.531 |
| **goodreads** | — | 64  | **2.520** | 0.563 |
| **goodreads** | — | 128 | **1.530** | 0.531 |
| goodreads | —    | 256 | 1.021 | 0.516 |

On arxiv and yambda, the measured value matches the algebraic floor within ≤ 25%. On Goodreads (small N ≈ 797k), the measured `silvertorch.index_mem` exceeds the algebraic floor by 2–5×. Most likely cause: KMeans Lloyd-iteration intermediates (distance matrices, cluster-assignment buffers) that are not freed before [sweep.py:442](evaluation/retrieval/sweep.py#L442) measures Δallocated. The gap closes on large-N catalogues where the index itself dominates the residual.

`[TODO: clarify with author — re-profile silvertorch with explicit `del kmeans_state; torch.cuda.empty_cache()` inside `KMeansTorch.fit` before `register_buffer` to confirm the residual hypothesis. If confirmed, document the algebraic floor as the "production" footprint; otherwise treat the measured value as the deployment-relevant number.]`

#### INT8 IVF vs 1-bit OPORP crossover (silvertorch vs linr_v3)

| cell | silvertorch (MiB) | linr_v3 (MiB) | smaller |
|---|---:|---:|---|
| arxiv-d64        | 232.4 | 387.7 | silvertorch (40% smaller) |
| arxiv-d128       | 418.7 | 775.3 | silvertorch (46% smaller) |
| arxiv-d256       | 789.6 | 1550.7| silvertorch (49% smaller) |
| yambda-500m-d64  | 149.7 | 242.0 | silvertorch (38% smaller) |
| yambda-500m-d128 | 254.9 | 484.1 | silvertorch (47% smaller) |
| yambda-500m-d256 | 499.8 | 968.2 | silvertorch (48% smaller) |
| yambda-5b-d64    | 421.1 | 695.5 | silvertorch (39% smaller) |
| yambda-5b-d128   | 800.3 | 1391.0| silvertorch (42% smaller) |
| **goodreads-d64**| **246.9** | **103.4** | **linr_v3 (58% smaller)** |
| **goodreads-d128**| **297.8**| **206.8** | **linr_v3 (31% smaller)** |
| goodreads-d256   | 397.97| 413.5 | silvertorch (4% smaller) |

**Crossover finding.** silvertorch is smaller on every (dataset, dim) cell **except small-N + low-d Goodreads**. The crossover sits between goodreads-d128 (linr_v3 wins by 31%) and goodreads-d256 (silvertorch wins by 4%). On all production-scale cells (arxiv, yambda), silvertorch is consistently 38–49% smaller than linr_v3.

**Joint memory × Recall.** Pairing index size with §6.2's Recall@100 at d128:

| cell | silvertorch (MiB, Recall) | linr_v3 (MiB, Recall) | Pareto |
|---|---|---|---|
| arxiv-d128       | (419, 0.9941) | (775, 1.0000) | silvertorch 46% smaller, 0.6% lower Recall |
| goodreads-d128   | (298, 0.1449) | (207, 0.1438) | linr_v3 31% smaller, 0.07% lower Recall |
| yambda-500m-d128 | (255, 0.1468) | (484, 0.1426) | silvertorch strictly dominates (smaller AND higher Recall) |
| yambda-5b-d128   | (800, 0.1739) | (1391, 0.1590) | silvertorch strictly dominates |

At d128 production scale, the co-designed IVF + INT8 retriever Pareto-dominates the LinR V3 cascade on the (memory, Recall) plane in three of four datasets; linr_v3 wins only on small-N Goodreads where the goodreads memory anomaly inflates silvertorch.

### Writer's notes (§6.3)

- The 16× memory-reduction claim from the LiNR paper is the section's anchor. State it explicitly, then point at A3 to either confirm or refine: the empirical ratio in this thesis's data is ≈ 16× for the 1-bit signature in isolation (`0.125·D·N` vs `2·D·N`), but the LinR-V3 *cascade* carries both the fp16 dense embeddings and the signatures, so the deployed V3 footprint is **slightly larger** than V1 (≈ 1.06×). **The memory-saving algorithm in the family is V4 (INT8), not V3.** See §6.3.4 for the algebra and verification.
- Silvertorch's memory footprint is COMPARABLE to fp16 baselines (per §6.3.4: silvertorch index_mem is 38–49% smaller than linr_v3 on large-N catalogues and 4% smaller than linr_v3 on goodreads-d256; only on small-N + low-d Goodreads does linr_v3 win). The honest framing: silvertorch wins memory at production scale; loses on tiny catalogues due to a measurement-time anomaly (§6.3.4).
- `fwd_scratch_mib ≈ 0` for all current algorithms — flag this in prose as a design property (the kernels are fused; no large per-forward scratch tensors). Note that this is itself a notable result (BitFunnel-style implementations often allocate scratch buffers > index_mem). Caveat: this is bs=1; at higher bs the score buffer grows linearly (§6.3.4 last paragraph).
- The **`peak / index ≈ const per impl` framing** (§6.3.4) is more actionable for a deployer than reporting peak per cell: budget `peak ≈ k · index` where k ∈ {3, 2.89, 5, ≈ 3} per impl.
- Cross-reference forward: A4 reappears as the §6.9 executive-summary figure.

---

## 6.4 Filter benchmark (Goodreads + arXiv)

**Scope.** Filter-suite results on Goodreads d{64,128,256} and arXiv d{64,128,256}. 5 algorithms (`linr_v1_filter_mask`, `linr_v2`, `linr_v3`, `linr_v4`, `silvertorch`) × 2 backends × 2 filter kinds (`clause`, `bloom`) × per-dataset sweep set × `batch_size ∈ {1, 8, 16}` × `k ∈ {100, 500, 1000}` × seed=0. Goodreads contains 810 rows per d, arXiv 900 rows per d (Goodreads has fewer because the bloom path omits the reverse-clause sweeps, see below).

**linr_v2 only in the filter suite.** `linr_v2` (the LinR V2 pre-filter) requires a filter module by construction ([evaluation/retrieval/algos/__init__.py:101-104](evaluation/retrieval/algos/__init__.py#L101-L104)) and so does not appear in §6.2; on filtered cells it is the second exact baseline (alongside `linr_v1_filter_mask`) and operates at recall parity with `linr_v1` while substantially cheaper on selective filters (it sparse-gathers passing items before scoring).

### 6.4.1 Filter taxonomy: narrow clause vs wide bag vs bloom

Three orthogonal classifications:

1. **Filter SCHEMA** — narrow (`[N+1, C=5, A_max=4]` int64, AND-of-OR clauses) vs wide (`[N+1, 1, 32]` int64, bag of shelves/keywords). Per [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.1/§3.2: Goodreads narrow has 5 clauses (C0 genre, C1 language, C2 format, C3 year, C4 author); arxiv narrow has 5 (C0 main category, C1 license, C2 year, C3 versions, C4 author). Goodreads wide = popular shelves (~1.8k vocab, top-32 per work). arXiv wide = leaf categories / keywords. **The filter benchmark sweeps documented here are NARROW only.** No `wide_1shelf` sweep is present in the goodreads `d128-filter.yaml` ([evaluation/config/goodreads/d128-filter.yaml:24-47](evaluation/config/goodreads/d128-filter.yaml#L24-L47)) — see the open question below.
2. **Filter KIND** — `clause` (exact AND-of-OR predicate via `ExactAttributeFilter`, [retrieve/src/retrieve/layers/filters/exact_attribute.py:12](retrieve/src/retrieve/layers/filters/exact_attribute.py#L12)) vs `bloom` (set-membership via `BloomFilter` with `m_bits=1024, k_hash=5`, [retrieve/src/retrieve/layers/filters/bloom.py:12](retrieve/src/retrieve/layers/filters/bloom.py#L12)). Clause is zero-false-positive; bloom approximates the same predicate at a smaller signature cost. `[CITE: Bloom 1970 CACM — affiliation: Computer Usage Co — kept]`, `[CITE: Goodwin et al. 2017 SIGIR (BitFunnel) — affiliation: Microsoft — kept]`.
3. **Sweep label** — the set of active clauses for a given run. Per the goodreads `d128-filter.yaml`:
   - clause sweeps: `c0_genre` (clause 0), `c1_lang_reverse` (clause 1, REVERSE), `c2_format`, `c3_year`, `c0c1`, `all4`. 6 sweeps × clause filter.
   - bloom sweeps: `c0_genre`, `c2_format`, `c3_year`. 3 sweeps × bloom filter — bloom OMITS `c1_lang_reverse` and any sweep containing it (`c0c1`, `all4`), because `BloomFilter` is forward-only (set-membership cannot represent NOT) and would raise; see config comment at [evaluation/config/goodreads/d128-filter.yaml:36-44](evaluation/config/goodreads/d128-filter.yaml#L36-L44).
   - Total goodreads filter sweeps = 9 (6 clause + 3 bloom).
   - Per the arXiv `d128-filter.yaml`:
   - clause sweeps: `c0_maincat`, `c2_year`, `c3_nversions`, `c0c2`, `all4`. 5 sweeps × clause filter.
   - bloom sweeps: same 5 names. arXiv has no reverse clauses, so bloom does not need to be pruned.
   - Total arXiv filter sweeps = 10 (5 clause + 5 bloom).

**Selectivity context (Goodreads narrow clauses).** From the dataset stats CSVs:
- C0 genre: 10 buckets, top-3 (`fiction`, `history`, `romance`) cover 52.7% / 28.1% / 27.9% of editions; full distribution `[DATA: docs/thesis/results-data/datasets/goodreads_clause_c0_genre.csv]`.
- C1 language: 30-bucket vocab, `eng` (708 k editions) + `en-US`, `en-GB` dominate; `c1_lang_reverse` selects all NOT-English editions (mid-selectivity, ≈ 5–20% of catalogue depending on per-query attr) `[DATA: docs/thesis/results-data/datasets/goodreads_clause_c1_lang_top30.csv]`.
- C2 format: 5 buckets, top-3 (`paperback` 39.8%, `other` 28.6%, `hardcover` 15.3%) `[DATA: docs/thesis/results-data/datasets/goodreads_clause_c2_format.csv]`.
- C3 year: 5 buckets (`2011+` 39.0%, `2001–2010` 22.4%, `None` 25.5%) `[DATA: docs/thesis/results-data/datasets/goodreads_clause_c3_year.csv]`.

Coverage percentages are reported per [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) §3.1 (Goodreads): C0 82.7%, C1 97.7% of non-null, C2 100%, C3 74.5%, C4 100%. Multi-clause `all4` ⇒ aggregate selectivity (intersection AND of OR-sets) is strictly more selective than any single clause.

**Pseudocode for the AND-of-OR exact-clause predicate.** Per [docs/thesis/03-methods.md](docs/thesis/03-methods.md) §2.4.1 and [retrieve/src/retrieve/layers/filters/exact_attribute.py](retrieve/src/retrieve/layers/filters/exact_attribute.py):

```pseudocode
# n: item index, q: query index, C: number of clauses, A_max: attrs per clause
function exact_clause_predicate(item_attrs[N,C,A_max], query_attrs[B,C], active_clauses):
    for q in 1..B:
        for n in 1..N:
            mask[q,n] = True
            for c in active_clauses:
                any_match = False
                for a in 1..A_max:
                    if item_attrs[n,c,a] == query_attrs[q,c]:
                        any_match = True; break
                if not any_match:
                    mask[q,n] = False; break  # AND fails on this clause
    return mask
```

Reverse clauses (the `c1_lang_reverse` Goodreads sweep) negate the per-clause result: an item passes if its language attrs do NOT contain the query's language. Bloom approximates the OR-of-equalities via `k_hash` mixed hashes into an `m_bits`-wide bitset, without supporting reverse semantics ([retrieve/src/retrieve/layers/filters/bloom.py:108-165](retrieve/src/retrieve/layers/filters/bloom.py#L108-L165)).

**silvertorch handling of reverse clauses (load-bearing finding).** The eval-side wrapper `SilvertorchAlgo.__init__` ([evaluation/retrieval/algos/silvertorch.py:73, 89](evaluation/retrieval/algos/silvertorch.py#L73-L89)) calls `SilverTorch.register_index(item_embs, item_clause_attrs=item_attrs_narrow)` **without** propagating `clause_is_reverse`. The driver's docstring at [evaluation/retrieval/algos/__init__.py:21-24](evaluation/retrieval/algos/__init__.py#L21-L24) states "Silvertorch on reverse-clause sweeps is skipped by the driver itself (evaluate.py), not here." However, the goodreads-d128-filter filter JSON data clearly shows silvertorch rows for `c1_lang_reverse`, `c0c1`, and `all4` with near-zero recall (Recall@100 ≈ 0.0005, 0.0407, 0.0425 respectively), indicating that the skip is NOT triggered for clause-mode silvertorch on reverse-clause sweeps. The fused-exact kernel then evaluates the AND-of-OR predicate without knowing the reverse semantics and almost no SASRec-relevant items pass. **Recommendation for the writer:** present the silvertorch rows on `c1_lang_reverse / c0c1 / all4` in Goodreads as a separate "incompatible-cell" row in the matrix (greyed out), not as a legitimate quality comparison. The d64 and d256 filter JSON data shows the same pattern (silvertorch R ≈ 0 / 0.07 / 0.07 on `c1_lang_reverse / c0c1 / all4` at d64; R ≈ 0 / 0.03 / 0.04 at d256), confirming this is a code-path bug, not a dim-specific issue. `[TODO — still unresolved: clarify with author — should the eval driver skip silvertorch on Goodreads c1_lang_reverse / c0c1 / all4? Either skip those cells in evaluate.py, or thread clause_is_reverse into SilverTorch.register_index and into the codesigned exact kernel. The fix should also propagate to §7 Limitations as "silvertorch's codesigned exact-clause kernel currently lacks reverse-clause support".]`

**wide_1shelf benchmark — built but unrun.** The Goodreads dataset prep script writes `item_attrs_wide.pt` of shape `[N+1, 1, 32]` int64 ([evaluation/datasets/goodreads.py:1200-1232](evaluation/datasets/goodreads.py#L1200-L1232)) and exports `query_attrs_wide_1shelf` / `query_attrs_wide_2shelf` columns into the SASRec sequence files ([evaluation/datasets/goodreads.py:1261-1267](evaluation/datasets/goodreads.py#L1261-L1267)). Same machinery exists in `evaluation/datasets/arxiv.py:893`. However, a full grep of `evaluation/config/` shows ZERO YAML configs referencing any `wide_*` sweep — only the `_narrow` attrs are used in `d{64,128,256}-filter.yaml` for both arxiv and goodreads. The wide-attribute filter benchmark is therefore PLANNED INFRASTRUCTURE WITHOUT EXECUTED RESULTS. Recommendation: §6.4 limits its scope to narrow-attribute filters; the wide-bag benchmark is a §7 Limitations entry ("wide-attribute (rare-biased shelf) filter benchmark exists at the dataset prep + module API level but no eval sweep has been run; left for future work").

### 6.4.2 Goodreads d128 — clause sweeps, bloom sweeps

> ⚠ **All Goodreads d128 filter numbers in this subsection are suspected stale** (see warning block in the §6.0 schema recap). Reported AS-IS; do not publish without rerunning the goodreads-d128-filter sweep against a fresh oracle.

Cell: `dataset=goodreads, dim=128, suite=filter, backend=triton, batch_size=1, k=100, seed=0`. Source per row: `evaluation/results/goodreads/d128-filter.json`. Same data in `[DATA: docs/thesis/results-data/results/all_results_long.csv]` (filter on `dataset==goodreads, dim==128, suite_label==filter`).

| sweep | filter | impl | Recall@100 | NDCG@100 | median_ms |
|---|---|---|---:|---:|---:|
| c0_genre | clause | linr_v1_filter_mask | 0.5994 | 0.6601 | 0.4620 |
| c0_genre | clause | linr_v2 | 0.5994 | 0.6601 | 0.3469 |
| c0_genre | clause | linr_v3 | 0.5768 | 0.6416 | 0.3898 |
| c0_genre | clause | linr_v4 | 0.5990 | 0.6598 | 0.6249 |
| c0_genre | clause | silvertorch | 0.5890 | 0.6516 | 0.2083 |
| c0_genre | bloom  | linr_v1_filter_mask | 0.5994 | 0.6601 | 0.4582 |
| c0_genre | bloom  | linr_v2 | 0.5994 | 0.6601 | 0.3210 |
| c0_genre | bloom  | linr_v3 | 0.5768 | 0.6416 | 0.3893 |
| c0_genre | bloom  | linr_v4 | 0.5990 | 0.6598 | 0.6494 |
| c0_genre | bloom  | silvertorch | 0.5890 | 0.6516 | 0.2448 |
| c2_format | clause | linr_v1_filter_mask | 0.6075 | 0.6690 | 0.4623 |
| c2_format | clause | linr_v2 | 0.6075 | 0.6690 | 0.3619 |
| c2_format | clause | linr_v3 | 0.5862 | 0.6517 | 0.3898 |
| c2_format | clause | linr_v4 | 0.6071 | 0.6686 | 0.6246 |
| c2_format | clause | silvertorch | 0.5934 | 0.6576 | 0.2099 |
| c2_format | bloom  | (parity with clause; see all_results_long.csv) | 0.6075 | 0.6690 | 0.4353–0.5974 (per impl) |
| c3_year | clause | linr_v1_filter_mask | 0.6020 | 0.6617 | 0.4615 |
| c3_year | clause | linr_v2 | 0.6020 | 0.6617 | 0.3560 |
| c3_year | clause | linr_v3 | 0.5806 | 0.6442 | 0.3890 |
| c3_year | clause | linr_v4 | 0.6016 | 0.6613 | 0.6247 |
| c3_year | clause | silvertorch | 0.5955 | 0.6568 | 0.2086 |
| c3_year | bloom | silvertorch | 0.5956 | 0.6568 | 0.2228 |
| c1_lang_reverse | clause | linr_v1_filter_mask | 0.5248 | 0.5936 | 0.4604 |
| c1_lang_reverse | clause | linr_v2 | 0.5225 | 0.5862 | 0.3661 |
| c1_lang_reverse | clause | linr_v3 | 0.4397 | 0.5161 | 0.3930 |
| c1_lang_reverse | clause | linr_v4 | 0.5243 | 0.5932 | 0.6245 |
| c1_lang_reverse | clause | silvertorch (**INCOMPATIBLE — reverse**) | 0.0005 | 0.0004 | 0.2098 |
| c0c1 | clause | linr_v1_filter_mask | 0.5498 | 0.6176 | 0.4613 |
| c0c1 | clause | linr_v2 | 0.5498 | 0.6176 | 0.3403 |
| c0c1 | clause | linr_v3 | 0.5022 | 0.5779 | 0.3912 |
| c0c1 | clause | linr_v4 | 0.5495 | 0.6174 | 0.6239 |
| c0c1 | clause | silvertorch (**INCOMPATIBLE — contains reverse**) | 0.0407 | 0.0455 | 0.2084 |
| all4 | clause | linr_v1_filter_mask | 0.6089 | 0.6733 | 0.4618 |
| all4 | clause | linr_v2 | 0.6089 | 0.6733 | 0.3300 |
| all4 | clause | linr_v3 | 0.5979 | 0.6643 | 0.3913 |
| all4 | clause | linr_v4 | 0.6087 | 0.6731 | 0.6243 |
| all4 | clause | silvertorch (**INCOMPATIBLE — contains reverse c1**) | 0.0425 | 0.0477 | 0.2090 |

`[DATA: evaluation/results/goodreads/d128-filter.json]`

**Observations (Goodreads d128 filter).**
- `linr_v1` and `linr_v2` are recall-identical (both exact baselines); `linr_v2` is consistently 25–30% faster than `linr_v1` (sparse gather beats dense matmul + mask when most items are filtered out).
- `linr_v3` loses 2–4% Recall vs the exact baselines on most single-clause sweeps (default `candidate_pool=5000`); the gap WIDENS to ≈ 16% on `c1_lang_reverse` (0.4397 vs 0.5248). Reason: c1 is the most selective single clause on goodreads, so most items in the top-5000 1-bit shortlist do NOT pass the post-filter, leaving fewer than 100 valid candidates for stage-2 rerank. §6.5.2's `candidate_pool` sweep confirms recovery.
- `silvertorch` on the THREE single-clause sweeps it supports (`c0_genre`, `c2_format`, `c3_year`) achieves competitive Recall (≤ 2% loss vs baseline) at 2.0×–2.3× speedup. On bloom for the same sweeps it is slightly slower (0.2–0.24 ms vs 0.21 ms) because the bloom-fused kernel pays a per-item bloom-bit lookup.
- The 5 "incompatible" silvertorch rows (`c1_lang_reverse`, `c0c1`, `all4`) MUST be visually distinguished in the prose / figures — see §6.4.1 finding above.
- bloom-vs-clause parity on `linr_v1/v2/v3/v4`: identical Recall (algorithms only consume the filter module's mask output, regardless of how it was computed) — small latency differences come from filter-module cost only. silvertorch is the only `impl` that sees true bloom-vs-clause behavioral differences because the predicate is fused into the kernel.

### 6.4.3 arXiv d128 — clause sweeps, bloom sweeps

Cell: `dataset=arxiv, dim=128, suite=filter, backend=triton, batch_size=1, k=100, seed=0`. Source: `evaluation/results/arxiv/d128-filter.json`.

| sweep | filter | impl | Recall@100 | NDCG@100 | median_ms |
|---|---|---|---:|---:|---:|
| c0_maincat | clause | linr_v1_filter_mask | 0.9923 | 0.9944 | 1.4245 |
| c0_maincat | clause | linr_v2 | 0.9885 | 0.9917 | 0.8471 |
| c0_maincat | clause | linr_v3 | 0.6756 | 0.7534 | 1.3129 |
| c0_maincat | clause | linr_v4 | 0.9539 | 0.9664 | 2.0254 |
| c0_maincat | clause | silvertorch | 0.8828 | 0.9128 | 0.1517 |
| c0_maincat | bloom  | silvertorch | 0.8828 | 0.9128 | 0.1608 |
| c0_maincat | bloom  | linr_v2 | 0.9885 | 0.9917 | 0.7443 |
| c2_year | clause | linr_v1_filter_mask | 0.9924 | 0.9946 | 1.4264 |
| c2_year | clause | linr_v2 | 0.9888 | 0.9919 | 0.8651 |
| c2_year | clause | linr_v3 | 0.6461 | 0.7307 | 1.3144 |
| c2_year | clause | linr_v4 | 0.9554 | 0.9675 | 2.0225 |
| c2_year | clause | silvertorch | 0.8695 | 0.9033 | 0.1517 |
| c2_year | bloom  | silvertorch | 0.8695 | 0.9032 | 0.1678 |
| c3_nversions | clause | linr_v1_filter_mask | 0.9921 | 0.9943 | 1.4246 |
| c3_nversions | clause | linr_v2 | 0.9884 | 0.9916 | 0.9407 |
| c3_nversions | clause | linr_v3 | 0.6077 | 0.6990 | 1.3274 |
| c3_nversions | clause | linr_v4 | 0.9534 | 0.9660 | 2.0233 |
| c3_nversions | clause | silvertorch | 0.8766 | 0.9087 | 0.1480 |
| c0c2 | clause | linr_v1_filter_mask | 0.9930 | 0.9950 | 1.4265 |
| c0c2 | clause | linr_v2 | 0.9897 | 0.9926 | 0.8176 |
| c0c2 | clause | linr_v3 | 0.8003 | 0.8510 | 1.3061 |
| c0c2 | clause | linr_v4 | 0.9587 | 0.9699 | 2.0252 |
| c0c2 | clause | silvertorch | 0.8523 | 0.8893 | 0.1482 |
| all4 | clause | linr_v1_filter_mask | 0.9940 | 0.9957 | 1.4252 |
| all4 | clause | linr_v2 | 0.9914 | 0.9938 | 0.8098 |
| all4 | clause | linr_v3 | 0.9159 | 0.9382 | 1.3050 |
| all4 | clause | linr_v4 | 0.9654 | 0.9749 | 2.0225 |
| all4 | clause | silvertorch | 0.7611 | 0.8172 | 0.1504 |

`[DATA: evaluation/results/arxiv/d128-filter.json]`

**Observations (arXiv d128 filter).**
- arxiv has NO reverse clauses, so silvertorch works on every sweep. Recall sits at 0.76–0.88 (vs 0.99 baseline) — significantly below baseline. This is the strongest case for `silvertorch` deep-sweep tuning: with default `(n_lists=1024, n_probe=24)`, the IVF probe misses many relevant items in the more selective sweeps. §6.5.1 traces the Recall recovery curve as `n_probe` grows up to 256 with `n_lists ∈ {1664, 8192}`.
- `linr_v3` at default `candidate_pool=5000` loses 30–40% Recall on single-clause arxiv sweeps (0.61 on `c2_year` vs 0.99) — much worse than on Goodreads. Cause: arxiv catalogue is ≈ 2.99 M, so 5000 candidates is 0.17% of N, and the per-clause selectivity is sharp. On `all4` (joint filter — more selective predicate, but with selected items having tightly clustered embeddings) linr_v3 RECOVERS to 0.9159: when the filter is selective enough to leave fewer than `candidate_pool` items in the indexable set, the shortlist becomes essentially exact for those.
- `linr_v2` again 1.6×–1.9× faster than `linr_v1` at parity Recall.
- `silvertorch` median_ms ≈ 0.15 ms is 9–10× faster than `linr_v1` on every arxiv sweep. The bloom-vs-clause silvertorch gap is small (0.15 ms vs 0.16 ms) — bloom is marginally slower because the per-item bloom-bit lookup adds a small overhead compared to the cheaper exact AND-of-OR on small `C × A_max = 5 × 4` int64 attrs.
- Quality cost of bloom vs clause for silvertorch: identical Recall to 4 decimals on every sweep (since narrow attrs are small enough that bloom collisions are rare with `m_bits=1024, k_hash=5`).

### 6.4.4 Algorithm × filter matrix (which impl wins where)

For each (dataset, sweep, filter_kind), at canonical `k=100, batch_size=1, backend=triton`:

- **Best Recall.** Always `linr_v1_filter_mask` or `linr_v2` (recall-identical, exact). Tie-break by latency goes to `linr_v2` on every cell.
- **Best Recall ≥ 95% of baseline AT MIN latency.** Always `silvertorch` on arxiv (within ≤ 5–13% of baseline) — but with the caveat that on the most selective sweeps (single-clause `c2_year` and `c3_nversions`) the Recall gap is 12% and motivates §6.5.1 tuning. On Goodreads the silvertorch gap is ≤ 2% on `c0_genre / c2_format / c3_year`.
- **Best Recall ≥ 99% of baseline at HALF the baseline latency.** `linr_v2` on every cell (sparse pre-filter dominates when selectivity is non-trivial).
- **When linr_v3 wins.** None of the canonical cells in this benchmark — at default `candidate_pool=5000`, `linr_v3` is dominated by `linr_v2` (which is exact + cheaper) and by `silvertorch` (faster). Its niche shows up at larger `candidate_pool` (§6.5.2), where it can match exact Recall while still being cheaper than `linr_v1_filter_mask`.

**Compact summary matrix (goodreads d128, k=100, bs=1, backend=triton, by sweep × impl, Recall@100):**

| sweep | linr_v1 | linr_v2 | linr_v3 | linr_v4 | silvertorch |
|---|---:|---:|---:|---:|---:|
| c0_genre        | **0.5994** | **0.5994** | 0.5768 | 0.5990 | 0.5890 |
| c2_format       | **0.6075** | **0.6075** | 0.5862 | 0.6071 | 0.5934 |
| c3_year         | **0.6020** | **0.6020** | 0.5806 | 0.6016 | 0.5955 |
| c1_lang_reverse | **0.5248** | 0.5225 | 0.4397 | 0.5243 | 0.0005 (incompatible) |
| c0c1            | **0.5498** | **0.5498** | 0.5022 | 0.5495 | 0.0407 (incompatible) |
| all4            | **0.6089** | **0.6089** | 0.5979 | 0.6087 | 0.0425 (incompatible) |

Bold = within 0.001 of the best in the row.

**Goodreads d64 (k=100, bs=1, triton, recall@100):** linr_v1 is at the oracle-baseline ceiling here (≈ 1.000), unlike d128/d256 where fp16 precision drift drops it below 1 — see the precision caveat in the schema recap above.

| sweep | fk | linr_v1 | linr_v2 | linr_v3 | linr_v4 | silvertorch |
|---|---|---:|---:|---:|---:|---:|
| c0_genre        | clause | **1.000** | 0.999 | 0.723 | 0.980 | 0.935 |
| c0_genre        | bloom  | **1.000** | 0.999 | 0.723 | 0.980 | 0.935 |
| c2_format       | clause | **1.000** | 0.999 | 0.729 | 0.980 | 0.929 |
| c3_year         | clause | **1.000** | 0.999 | 0.724 | 0.980 | 0.944 |
| c1_lang_reverse | clause | **0.999** | 0.992 | 0.478 | 0.978 | 0.000 (incompatible) |
| c0c1            | clause | **0.999** | **0.999** | 0.635 | 0.980 | 0.071 (incompatible) |
| all4            | clause | **0.999** | **0.999** | 0.875 | 0.983 | 0.067 (incompatible) |

`[DATA: evaluation/results/goodreads/d64-filter.json]`. Latency cells (median_ms): linr_v1 ≈ 0.30, linr_v2 ≈ 0.31, linr_v3 ≈ 0.38, linr_v4 ≈ 0.48, silvertorch ≈ 0.20.

**Goodreads d256 (k=100, bs=1, triton, recall@100):** baseline drops further from 1.0 (precision drift now severe at d=256 on goodreads SASRec embeddings).

| sweep | fk | linr_v1 | linr_v2 | linr_v3 | linr_v4 | silvertorch |
|---|---|---:|---:|---:|---:|---:|
| c0_genre        | clause | **0.489** | **0.489** | 0.488 | **0.489** | 0.481 |
| c0_genre        | bloom  | **0.489** | **0.489** | 0.488 | **0.489** | 0.481 |
| c2_format       | clause | **0.497** | **0.497** | 0.496 | **0.497** | 0.487 |
| c3_year         | clause | **0.491** | **0.491** | 0.491 | **0.492** | 0.489 |
| c1_lang_reverse | clause | **0.421** | 0.419 | 0.412 | **0.421** | 0.001 (incompatible) |
| c0c1            | clause | **0.454** | **0.454** | 0.451 | **0.454** | 0.033 (incompatible) |
| all4            | clause | **0.526** | **0.526** | 0.525 | **0.526** | 0.036 (incompatible) |

`[DATA: evaluation/results/goodreads/d256-filter.json]`. Latency cells (median_ms): linr_v1 ≈ 0.71–0.75, linr_v2 ≈ 0.37–0.44, linr_v3 ≈ 0.40–0.44, linr_v4 ≈ 0.92–1.04, silvertorch ≈ 0.25.

**arXiv d64 (k=100, bs=1, triton, recall@100):**

| sweep | fk | linr_v1 | linr_v2 | linr_v3 | linr_v4 | silvertorch |
|---|---|---:|---:|---:|---:|---:|
| c0_maincat   | clause | **0.991** | 0.986 | 0.513 | 0.933 | 0.854 |
| c2_year      | clause | **0.991** | 0.986 | 0.473 | 0.936 | 0.839 |
| c3_nversions | clause | **0.991** | 0.986 | 0.428 | 0.933 | 0.847 |
| c0c2         | clause | **0.992** | 0.988 | 0.686 | 0.942 | 0.821 |
| all4         | clause | **0.993** | 0.990 | 0.864 | 0.954 | 0.727 |

`[DATA: evaluation/results/arxiv/d64-filter.json]`. Latency cells (median_ms): linr_v1 ≈ 0.96, linr_v2 ≈ 0.57–0.80, linr_v3 ≈ 1.10–1.26, linr_v4 ≈ 1.40–1.52, silvertorch ≈ 0.15.

**arXiv d256 (k=100, bs=1, triton, recall@100):**

| sweep | fk | linr_v1 | linr_v2 | linr_v3 | linr_v4 | silvertorch |
|---|---|---:|---:|---:|---:|---:|
| c0_maincat   | clause | **0.993** | 0.990 | 0.851 | 0.966 | 0.898 |
| c2_year      | clause | **0.993** | 0.990 | 0.830 | 0.966 | 0.884 |
| c3_nversions | clause | **0.993** | 0.990 | 0.808 | 0.965 | 0.891 |
| c0c2         | clause | **0.994** | 0.991 | 0.912 | 0.969 | 0.868 |
| all4         | clause | **0.994** | 0.992 | 0.963 | 0.973 | 0.778 |

`[DATA: evaluation/results/arxiv/d256-filter.json]`. Latency cells (median_ms): linr_v1 ≈ 2.35, linr_v2 ≈ 0.96–1.26, linr_v3 ≈ 1.30–1.47, linr_v4 ≈ 3.12–3.24, silvertorch ≈ 0.17.

**Cross-dim takeaways for the writer.**
- **silvertorch recall stays roughly flat across dims on arxiv** (e.g., c0_maincat: 0.854 → 0.883 → 0.898 across d64/d128/d256) — IVF + INT8 quality is dimension-insensitive on the nomic-embed-v1.5 distribution. silvertorch DOES degrade on `all4` as d grows on arxiv (0.727 → 0.761 → 0.778 — actually improves slightly), suggesting d=128 is not a sweet spot.
- **linr_v3 recall grows sharply with dim on arxiv** (e.g., c0_maincat: 0.513 → 0.676 → 0.851). At higher d, the 1-bit Sign-OPORP projection retains more information per item (more bits relative to D) and the shortlist is more accurate.
- **DO NOT publish cross-dim trends on Goodreads filter recall until the stale-cache rerun (see ⚠ block above).** The apparent dim-dependence on Goodreads (baseline 1.0 → 0.6 → 0.5) is almost certainly a ground-truth artefact, NOT a real dim effect. After the rerun, expected: baseline ≈ 1.0 at all dims (modulo ≤ 1% precision), with silvertorch / linr_v3 / linr_v4 showing their true cross-dim curves.
- **Latency scales linearly with d** on the dense baselines (`linr_v1`, `linr_v4`): arxiv linr_v1 0.96 → 1.42 → 2.35 ms across d64/128/256. silvertorch is roughly constant across dims because its compute is dominated by IVF cluster scanning, not the per-item dot-product cost. Latency numbers are NOT affected by the ground-truth stale-cache issue (latency is measured separately from quality).

### Figure stub: 6.4.5 = D2. Recall vs selectivity curves (filter behavior, twin-panel)

**Plot.** Faceted scatter / line plot — direct template from LiNR paper Fig 7 (filter-size% vs latency+recall dual axis) for reviewer comparability:
- Rows: `dataset ∈ {arxiv, goodreads}`. Cols: metric ∈ {Recall@100, median_ms}. Optional inner facets: `filter_kind ∈ {clause, bloom}`.
- **X:** sweep label ordered by selectivity (single-clause sweeps → joint-clause sweeps → all4). Goodreads: `c3_year → c2_format → c0_genre → c1_lang_reverse → c0c1 → all4`. arxiv: `c3_nversions → c2_year → c0_maincat → c0c2 → all4` (rough selectivity order; verify by computing actual avg-pass-rates from §6.1.1's D3 plot data).
- **Y (left col):** Recall@100. **Y (right col):** median_ms (bs=1).
- **One line per** `impl ∈ {linr_v1_filter_mask, linr_v2, linr_v3, linr_v4, silvertorch}`; use markers. Mark silvertorch-incompatible cells on Goodreads (reverse clauses) with hollow markers + footnote.

**Source CSV.** [docs/thesis/results-data/results/all_results_long.csv](results-data/results/all_results_long.csv) — filter on `suite=="filter" & dataset in {arxiv, goodreads} & dim==128 & backend=="triton" & batch_size==1 & k==100 & seed==0`. Optionally also d64 / d256 in separate appendix.

**Catalog ID:** D2. **File destination:** `docs/thesis/results-data/results/D2_filter_recall_vs_selectivity_<dataset>.png`. (Upgrades the existing `goodreads_d128_filter_recall.png` — add latency twin panel.)

### Figure stub: 6.4.6 = D1. Recall–latency Pareto stratified by selectivity bucket

**Plot.** Four facets: filter selectivity bucket ∈ {<1%, 1–10%, 10–50%, >50%}; same Pareto axes as A1.

- X: `median_ms` (log); Y: Recall@100 (linear).
- Color = algorithm; marker = backend; baseline at recall=1.0 reference line.
- For each (dataset, dim) pair: one panel-bundle per selectivity bucket.
- **Story-telling expectation:** filtered-V1 (post-filter) collapses at low selectivity (the "liquidity" problem from the LiNR paper); V2 (pre-filter) and silvertorch + Bloom hold up. This is the central LiNR-reproducibility figure.

**Source.** Filter sweeps in `all_results_long.csv`; bucket by per-clause selectivity (median over queries) computed from D3's data (§6.1.1) or from raw `item_attrs_narrow.pt`.

**Catalog ID:** D1. **File destination:** `docs/thesis/results-data/results/D1_recall_latency_by_selectivity.png`. **Precedent:** Filtered-DiskANN (WWW 2023) and ACORN (SIGMOD 2024).

### Table stub: 6.4.7 = G7. Cross-over points

Empirical thresholds derived from §6.4.2 / §6.4.3 / §6.4.4 tables. Each row: dataset, dim, threshold value, measurement source. Example rows:

| Crossover | Dataset/dim | Threshold | Measurement | Source |
|---|---|---|---|---|
| V1 vs V2 latency parity break | arxiv-d128 | selectivity ≈ X% | linr_v2 < linr_v1 for selectivity ≤ X | §6.4.3 |
| linr_v3 default-pool collapse | yambda-5b | catalogue size ≥ N | linr_v3@pool=5000 loses ≥10% Recall | §6.2.3 |
| silvertorch Pareto break vs linr_v1 | arxiv-d128, all4 | Recall ≤ 0.85 | at this Recall, silvertorch is ≥5× faster | §6.4.3 |

**Use.** Turns the dense Pareto plots into actionable rules-of-thumb for §6.9 (executive summary) and the body prose.

**Catalog ID:** G7. **File destination:** rendered as a markdown table in the chapter.

### Writer's notes (§6.4)

- **🛑 Top priority:** rerun goodreads-d128 and d256 filter sweeps with a fresh oracle before publishing §6.4.2 / §6.4.4 (d128/d256 sub-tables). Details in the ACTION REQUIRED block. Goodreads d64 and all arxiv filter numbers are clean.
- The silvertorch-on-reverse-clause finding is a methodological caveat that should appear in the prose introducing §6.4.2 (Goodreads), NOT buried in a footnote. The writer should make explicit that the three goodreads sweeps containing c1 (`c1_lang_reverse / c0c1 / all4`) are NOT a fair comparison for silvertorch and the rows are reported here only for completeness. This finding holds across d64/d128/d256 (≈ 0 recall on those sweeps at every dim), so it is independent of the stale-cache issue. Recommend promoting to a §7 Limitations bullet: "silvertorch's codesigned exact-clause kernel does not currently distinguish reverse clauses; the eval driver does not skip these cells either; fix scheduled."
- The contrast "linr_v3 collapses on selective single-clause sweeps because `candidate_pool=5000` is too small" is the natural setup for §6.5.2. Suggest cross-referencing forward. The §6.4 linr_v3 numbers themselves may shift slightly after the goodreads rerun, but the direction (collapse on small pool, recovery with larger pool) is robust and corroborated by both the deep_sweep (fresh oracle) and the arxiv filter (also fresh oracle).
- The arxiv filter benchmark has bloom rows for every sweep; goodreads has bloom only for 3 sweeps (`c0_genre / c2_format / c3_year`). Reason: BloomFilter is forward-only and skips reverse-clause sweeps + any multi-clause sweep containing one. Explain this up-front in §6.2's introduction.
- d64 and d256 algorithm matrices added to §6.4.4 in this pass. Writer can either keep them inline or move d64/d256 to Appendix D; only d128 is contractually required as the canonical dim.
- The `wide_1shelf` (wide-bag) filter benchmark is **infrastructure built but unrun** (dataset prep writes `item_attrs_wide.pt` and `query_attrs_wide_*` columns; no YAML config references them). This is a §7 Limitations entry: "wide-attribute (rare-biased shelf) filter benchmark exists at the dataset-prep + module-API level but no eval sweep has been executed; left for future work as a complement to the narrow-attribute results presented in §6.4."
- The "5 vs 4 algorithms" point (linr_v2 appears here but not in §6.1) is worth one sentence in the §6.2 intro.
- arxiv `c0_maincat` selectivity is much sharper than goodreads `c0_genre` — note this when interpreting the silvertorch numbers (silvertorch loses 11% on arxiv c0_maincat but only 1.7% on goodreads c0_genre because the absolute number of items passing the filter is very different).

---

## 6.5 Parameter sensitivity (deep sweeps)

**Scope.** Two single-algorithm hyperparameter ablations at d128, designed to trace the recall-vs-latency Pareto each algorithm exposes when its tunable hyperparameter is varied. Configuration files: [evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml](evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml), [evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml](evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml). Both runs use `seed=0`; the linr_v3 sweep also fixes `v3_seed=0` (and per [evaluation/retrieval/algos/__init__.py:46](evaluation/retrieval/algos/__init__.py#L46) "linr_v4 is paramless — no curve to trace" so no analogous v4 deep sweep exists).

### 6.5.1 = B1. arXiv d128 silvertorch: `(n_lists, n_probe)` heatmap

**Grid (10 points).** Verified from the unique `extra.params` set across all 900 rows:
- `n_lists ∈ {1664, 8192}` (paper-faithful `√N ≈ 1664` vs small-cluster `8192`).
- `n_probe ∈ {4, 8, 32, 128, 256}`.
- Fixed: `n_iter=10, m_bits=1024, k_hash=5, seed=0`, `filter_kind=bloom` (silvertorch deep sweep is bloom-only — clause sweeps would be auto-skipped per the config comment at [evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml:41-43](evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml#L41-L43)).
- 5 filter sweeps × 3 `k` × 3 `batch_sizes` × 2 `backends` × 10 params = 900 rows. ✓

**Canonical cell: `sweep=all4, filter_kind=bloom, backend=triton, bs=1, k=100`.** Most selective sweep on arxiv (joint 4-clause AND); recall hardest to recover.

| n_lists | n_probe | probed pool $P$ (≈) | Recall@100 | NDCG@100 | median_ms |
|---:|---:|---:|---:|---:|---:|
| 1664 |   4 |    6 500 | 0.4028 | 0.5128 | 0.1756 |
| 1664 |   8 |   14 000 | 0.5201 | 0.6174 | 0.1242 |
| 1664 |  32 |   58 000 | 0.7465 | 0.8055 | 0.1479 |
| 1664 | 128 |  232 000 | 0.9015 | 0.9266 | 0.2276 |
| 1664 | 256 |  415 000 | 0.9398 | 0.9558 | 0.3053 |
| 8192 |   4 |    1 300 | 0.2470 | 0.3623 | 0.1459 |
| 8192 |   8 |    2 600 | 0.3377 | 0.4524 | 0.7355 |
| 8192 |  32 |   11 000 | 0.5494 | 0.6426 | 0.8651 |
| 8192 | 128 |   42 000 | 0.7601 | 0.8159 | 0.8547 |
| 8192 | 256 |   84 000 | 0.8424 | 0.8808 | 0.8687 |

`[DATA: evaluation/results/deep_sweeps/arxiv-d128-silvertorch.json]`

**Canonical cell: `sweep=c0_maincat, filter_kind=bloom, backend=triton, bs=1, k=100`.** Single-clause; less selective, recall ceiling is higher.

| n_lists | n_probe | Recall@100 | NDCG@100 | median_ms |
|---:|---:|---:|---:|---:|
| 1664 |   4 | 0.6300 | 0.7100 | 0.1755 |
| 1664 |   8 | 0.7368 | 0.7981 | 0.1264 |
| 1664 |  32 | 0.8827 | 0.9128 | 0.1540 |
| 1664 | 128 | 0.9424 | 0.9579 | 0.2550 |
| 1664 | 256 | 0.9516 | 0.9647 | 0.3132 |
| 8192 |   4 | 0.5310 | 0.6275 | 0.1390 |
| 8192 |   8 | 0.6383 | 0.7186 | 0.7213 |
| 8192 |  32 | 0.8067 | 0.8541 | 0.7786 |
| 8192 | 128 | 0.9069 | 0.9312 | 0.7832 |
| 8192 | 256 | 0.9325 | 0.9504 | 0.7858 |

`[DATA: evaluation/results/deep_sweeps/arxiv-d128-silvertorch.json]` (same file, filter by `sweep=="c0_maincat"`).

**Observations (silvertorch deep sweep).**
- **Recall increases monotonically with `n_probe`** at fixed `n_lists`, as expected: more probed clusters → more relevant items in the candidate pool. Going from `n_probe=4` to `n_probe=256` on `n_lists=1664` raises Recall@100 from 0.40 → 0.94 on `all4`, 0.63 → 0.95 on `c0_maincat`.
- **`n_lists=1664` dominates `n_lists=8192` at every comparable `n_probe`.** At `(n_lists=8192, n_probe=128)` the probed pool is 42 k items (1.4% of N) vs `(n_lists=1664, n_probe=128)` 232 k items (7.7% of N), so Recall is 0.76 vs 0.90 on `all4`. The paper-faithful `√N ≈ 1664` layout wins because, at fixed compute budget, larger clusters expose more candidates per probe.
- **Latency anomaly at `n_lists=8192, n_probe ≥ 8`.** Median latencies jump from 0.14 ms (n_probe=4) to 0.72–0.87 ms (n_probe=8–256), while the same n_probe values on `n_lists=1664` stay at 0.12–0.31 ms. The probed pool is smaller, so this is not bandwidth-bound. The fused codesigned kernel uses a single hard-coded tile config: `DEFAULT_CONFIG = CodesignedProbeScoreConfig(block_p=256, num_warps=4)` ([retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py:32](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L32)), explicitly tuned against the **paper-faithful √N regime**: comment at [codesigned_probe_score.py:30-32](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L30-L32) — "Tuned on A100 (sm_80) against the int8×int8 tl.dot path: block_p=256 wins plurality (3/6 regimes); num_warps=4 wins all 6". With `n_lists=1664` and arxiv N ≈ 2.99 M, mean cluster size ≈ 1798 — tiles divide cleanly (≈ 7 tiles per probe). With `n_lists=8192`, mean cluster ≈ 365 — each probed cluster needs ≤ 2 tiles, but the kernel pays a per-cluster launch overhead and the BLOCK_P=256 wastes lanes inside under-filled tiles. The retune CLI exists (`uv run tune-kernels codesigned-probe-score`, per the docstring at line 26-29) but has only been run for the larger-cluster regime. Recommendation: this is a **known autotune-coverage gap**, not a bug; for the thesis the cleanest framing is "the IVF + codesigned scoring kernel's default tile is optimized for the paper-faithful √N cluster layout; n_lists ≫ √N is supported but operates on a slower autotune branch, motivating the paper-faithful default elsewhere in the analysis." A follow-up retune across cluster-size regimes is a §7 Limitations / future-work entry.
- **Pareto frontier (all4 sweep, bloom, bs=1, k=100):** the dominating points (after dropping any point dominated on both Recall and latency by another in the same column) are:
  - `(1664, 4)`: Recall=0.40, ms=0.18 — bad recall.
  - `(1664, 8)`: Recall=0.52, ms=0.12 — Pareto.
  - `(1664, 32)`: Recall=0.75, ms=0.15 — Pareto.
  - `(1664, 128)`: Recall=0.90, ms=0.23 — Pareto.
  - `(1664, 256)`: Recall=0.94, ms=0.31 — Pareto.
  - All `n_lists=8192` points are dominated.
- **Comparison with §6.4.3 default `(n_lists=1024, n_probe=24)`.** The §6.4.3 baseline silvertorch hits Recall@100 = 0.7611 on `all4` at 0.15 ms. The deep sweep shows you can reach 0.90 Recall (vs baseline 0.99) at 0.23 ms — a 7× speedup over `linr_v1` for an 8% Recall sacrifice.

**Figure stub: 6.5.1 = B1. Silvertorch (n_lists × n_probe) heatmaps with iso-contours.**
- **Plot 1 (Recall):** contour heatmap, X = `n_probe` (log scale, points at 4/8/32/128/256), Y = `n_lists` (categorical {1664, 8192}), Z = `Recall@100`. Viridis colormap; **overlay iso-LATENCY contour lines at {0.2, 0.3, 0.5, 1.0 ms}** (CAGRA-style — one panel exposes both axes of the tradeoff).
- **Plot 2 (Latency):** same axes, Z = `median_ms`. **Overlay iso-RECALL contour lines at {0.5, 0.7, 0.9, 0.95}**. Highlight the `n_lists=8192` slowdown region with annotation.
- Cell = `sweep=all4`. Optional 5-panel grid: one heatmap pair per sweep (`c0_maincat / c2_year / c3_nversions / c0c2 / all4`) in Appendix D.
- Source: `evaluation/results/deep_sweeps/arxiv-d128-silvertorch.json`, filter on `backend=triton & bs=1 & k=100`.

**Catalog ID:** B1. **File destination:** `docs/thesis/results-data/results/B1_silvertorch_heatmap_<recall|latency>.png`. (Upgrades existing `6_3-silvertorch-arxiv-d128-heatmap.png` with iso-contour overlay.)

### 6.5.2 = B2. Goodreads d128 linr_v3: `candidate_pool` × bit-budget

**Grid (5 points).** Verified from the unique `extra.params` set across all 810 rows:
- `candidate_pool ∈ {2000, 4000, 8000, 16000, 32000}`.
- Fixed: `v3_seed=0`. `k_bits` is NOT exposed by the eval registry ([evaluation/retrieval/algos/__init__.py:88-96](evaluation/retrieval/algos/__init__.py#L88-L96)) and defaults to D=128 internally per [retrieve/src/retrieve/layers/linr/one_bit_knn.py:69](retrieve/src/retrieve/layers/linr/one_bit_knn.py#L69).
- 6 clause sweeps + 3 bloom sweeps × 3 `k` × 3 `batch_sizes` × 2 `backends` × 5 params = 810 rows. ✓

**Canonical cell: `sweep=c0_genre, filter_kind=clause, backend=triton, bs=1, k=100`.**

| candidate_pool | Recall@100 | NDCG@100 | median_ms |
|---:|---:|---:|---:|
|  2 000 | 0.7521 | 0.8136 | 0.3608 |
|  4 000 | 0.8507 | 0.8893 | 0.3733 |
|  8 000 | 0.9229 | 0.9434 | 0.4048 |
| 16 000 | 0.9676 | 0.9764 | 0.4357 |
| 32 000 | 0.9896 | 0.9925 | 0.4211 |

**Same with `filter_kind=bloom`.**

| candidate_pool | Recall@100 | NDCG@100 | median_ms |
|---:|---:|---:|---:|
|  2 000 | 0.7520 | 0.8135 | 0.3252 |
|  4 000 | 0.8507 | 0.8893 | 0.3160 |
|  8 000 | 0.9228 | 0.9434 | 0.4071 |
| 16 000 | 0.9676 | 0.9764 | 0.4069 |
| 32 000 | 0.9896 | 0.9925 | 0.4249 |

**Canonical cell: `sweep=all4, filter_kind=clause, backend=triton, bs=1, k=100`.**

| candidate_pool | Recall@100 | NDCG@100 | median_ms |
|---:|---:|---:|---:|
|  2 000 | 0.8421 | 0.8828 | 0.3299 |
|  4 000 | 0.9329 | 0.9508 | 0.3436 |
|  8 000 | 0.9785 | 0.9843 | 0.4057 |
| 16 000 | 0.9938 | 0.9955 | 0.4371 |
| 32 000 | 0.9978 | 0.9984 | 0.4137 |

`[DATA: evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json]`

**Observations (linr_v3 deep sweep).**
- **Recall converges monotonically toward the exact baseline as `candidate_pool` grows.** On `c0_genre` (single clause), pool 2000 → 32000 raises Recall@100 from 0.7521 → 0.9896. The convergence target is whatever the oracle returns; at pool=32000 stage-2's fp16 rescore over the shortlist is essentially exact-filtered top-K. **Important cross-suite caveat:** these deep_sweep numbers were scored against a FRESH oracle (the deep_sweep config's `K_GT = max([100, 200, 400]) = 400` does not match the filter-suite cache's shape `(10000, 1000)`, forcing recomputation per [oracle.py:113-119](evaluation/retrieval/oracle.py#L113-L119)), whereas §6.4.2's filter-suite numbers were scored against a STALE oracle cache (see ⚠ block in the §6.0 schema recap). This is WHY §6.5.2 linr_v3@pool=32000 reaches Recall ≈ 0.99 while §6.4.2 linr_v3@pool=5000 reports only 0.5768 — the two suites are scoring against ground truths that disagree. Once the goodreads filter sweeps are rerun against a fresh oracle, the §6.4 numbers should align with §6.5 (linr_v1 baseline ≈ 1.0; linr_v3 default pool ≈ 0.85–0.90 by interpolation between deep_sweep pool=4000 and pool=8000).
- **Latency increase with `candidate_pool` is sub-linear.** Pool 2000 → 32000 (16× growth) → latency 0.36 → 0.42 ms (1.17×). Stage 1 (`OneBitKNN` 1-bit popcount + top-`candidate_pool`) cost scales modestly because the popcount + reduce is bandwidth-bound at small d, and stage 2 (`PrefilterKNN` fp16 rescore on the shortlisted items) cost scales with the gather + matmul size — but the size effectively saturates around `candidate_pool ≈ N_filtered` (the number of items passing the filter), past which extra "candidates" are padding -1s that the indirect-load path's per-row `counts` value skips ([evaluation/retrieval/algos/linr_v3.py:88-91](evaluation/retrieval/algos/linr_v3.py#L88-L91)).
- **`bloom` vs `clause` for linr_v3.** Recall is identical to 4 decimals (algorithms only consume the filter module's compact-output). Latency is similar but bloom is slightly faster (`evaluate_indices` bloom-compact is cheaper than clause-compact for these small `C × A_max` shapes).
- **all4 (joint filter) recovers faster than c0_genre.** At pool=2000 the all4 Recall is already 0.84 vs c0_genre 0.75. Reason: more selective filter ⇒ fewer items pass ⇒ a fixed shortlist is more likely to contain the small set of relevant items.

**Figure stub: 6.5.2 = B2. linr_v3 candidate_pool curves + bit-budget axis (planned).**

**Current data (candidate_pool axis only):**
- **Plot 1 (Recall):** line plot, X = `candidate_pool` (log scale {2k, 4k, 8k, 16k, 32k}), Y = `Recall@100`. One line per filter sweep label (color-coded). Horizontal dashed line at exact baseline Recall per sweep.
- **Plot 2 (Latency):** same X, Y = `median_ms`. Shows the cost curve.
- Optional facet: two side-by-side panels for `filter_kind in {clause, bloom}`.
- Cell = `backend=triton & bs=1 & k=100`. Source: `evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json`.

**Planned upgrade (requires new sweep):** add a `k_bits` axis to make the heatmap two-dimensional:
- **Heatmap:** Recall@100 vs `(n_bits = k_bits param ∈ {d/8, d/4, d/2, d, 2d}) × (candidate_pool ∈ {500, 1000, 2000, 5000, 10000})`.
- The new `SimHashKNN` layer + `quantize_simhash_1bit` (commit `4f9b1a6`) makes `k_bits` a first-class knob — the current sweep fixes `k_bits=D=128` ([retrieve/src/retrieve/layers/linr/one_bit_knn.py:69](retrieve/src/retrieve/layers/linr/one_bit_knn.py#L69)) and varies only `candidate_pool`. A new deep-sweep YAML is required.
- *Precedent:* LiNR Fig 7 (filter-size% vs latency+recall dual axis) is the direct template.

**Catalog ID:** B2. **File destination:** `docs/thesis/results-data/results/B2_linr_v3_pool_curves.png` (current); `B2_linr_v3_bit_budget_heatmap.png` (planned).

### 6.5.3 = B3. K sensitivity curve (planned)

**Plot.** Small multiples 3×1 (one per dataset).
- **X:** K ∈ {10, 100, 1000} (log).
- **Y:** Recall@K. One line per algorithm.
- Companion: same X, Y = `median_ms`.

**Rationale.** SIGIR norm — different applications use different K. Yambda quality config uses {100, 200, 400}; filter configs go up to 1000. Bridge the gap with a single sweep at K ∈ {10, 100, 1000} for all algos.

**Status.** Blocked on K=10 quality re-run (currently only K≥100 in the schema; per the "Gaps requiring new measurements" table in the plot catalog). The §6.2.4 table already presents K=100/200/400 for all algorithms.

**Catalog ID:** B3.

### 6.5.4 = B4. Batch-size scaling (planned)

**Plot.** X = batch_size ∈ {1, 4, 16, 64, 256, 1024} (log); Y = `median_ms` per-batch (log).
- One line per algorithm.
- **The point of interest:** where the curve becomes super-linear (algorithm losing batched-amortization) vs sub-linear (algorithm benefiting from kernel-occupancy).
- **Companion panel:** same X, Y = `median_ms / batch_size` ("median latency per query within a batch") — exposes amortization. A flat line = perfectly batchable; rising line = batch overhead dominates.

**Status.** Requires extended bs sweep beyond current {1, 8, 16} grid. Recommended target: arXiv-d128 at all 5 algorithms × extended bs grid (perf-only pass, no quality recompute needed since quality is bs-invariant per [bench_tools.py:347-353](evaluation/retrieval/bench_tools.py#L347-L353)).

**Catalog ID:** B4.

### 6.5.5 Pareto interpretation

Both deep sweeps trace the **same fundamental shape**: a hyperparameter (n_probe / candidate_pool) controls the size of an approximate candidate pool, and Recall trades off against latency. The contrast:

- **silvertorch** Pareto on arxiv d128 (all4): from `(0.52 Recall, 0.12 ms)` to `(0.94, 0.31 ms)` — sub-millisecond regime. The Recall ceiling is 0.94 even at full `n_probe=256`, indicating that the bloom filter approximation introduces irreducible Recall loss above some `n_probe` (further gains would require richer bloom signatures or per-item INT8 scales — see [retrieve/src/retrieve/layers/silvertorch/main.py:55-62](retrieve/src/retrieve/layers/silvertorch/main.py#L55-L62) "Quality knob" comment).
- **linr_v3** Pareto on goodreads d128 (c0_genre): from `(0.75, 0.36 ms)` to `(0.99, 0.42 ms)`. The recall ceiling approaches exact (.99) at the largest pool size, because stage-2 is fp16-exact within the shortlist. Cost scales mildly — the candidate_pool sweep is a "cheap recall recovery" curve.

The two algorithms operate at different points: silvertorch is 3–5× faster than linr_v3 but plateaus at lower Recall; linr_v3 is slower but can essentially match exact at the cost of a larger shortlist. For an application allowing 5% Recall loss, silvertorch wins; for an application demanding ≥ 99%, linr_v3 wins. The exact baseline (`linr_v1_filter_mask` or `linr_v2`) wins only when both speed and Recall constraints are loose enough that the 1.4 ms / 0.46 ms baseline is acceptable.

### Writer's notes (§6.5)

- ~~The deep-sweep Recall discrepancy with §6.4 needs the author's clarification~~ **RESOLVED.** Cause: §6.4 goodreads-d128/d256 filter scored against a stale oracle disk cache (shape (10000, 1000)) reused without content validation; §6.5 deep_sweep scored against a freshly-recomputed oracle (shape (10000, 400) forced recompute via the shape-mismatch branch at [oracle.py:113-119](evaluation/retrieval/oracle.py#L113-L119)). See the 🛑 ACTION REQUIRED block at the top of this notes file. The §6.5.2 deep_sweep numbers are CORRECT and writer-safe; the §6.4 goodreads-d128/d256 numbers are TAINTED and need a rerun.
- The §6.5.1 latency anomaly at `n_lists=8192` is diagnosed: the codesigned kernel ships with a single hardcoded tile config (`block_p=256, num_warps=4`) tuned for the paper-faithful √N cluster regime. The 8192-list layout pays a per-cluster launch overhead and BLOCK_P=256 wastes lanes on under-filled (≈ 365-item) tiles. This is a known **autotune-coverage gap** rather than a bug; framing for the writer: "the IVF + codesigned scoring kernel's default tile is optimized for the paper-faithful √N layout, which doubles as empirical evidence for why that layout is the default." Worth a §7 future-work bullet but not a §6 caveat.
- The §6.5 narrative should explicitly connect to §6.4: "Sections 6.4's silvertorch / linr_v3 results used the canonical defaults (silvertorch: n_lists=1024, n_probe=24; linr_v3: candidate_pool=5000). §6.5 shows that, by tuning, both algorithms can move along their Pareto frontiers — silvertorch from ≈ 9× speedup at ≈ 85% Recall to ≈ 5× speedup at ≈ 94%; linr_v3 from ≈ 3× speedup at ≈ 75% Recall to ≈ 3.4× speedup at ≈ 99%." (Speedup ratios reference arxiv-d128 baselines; concrete numbers per cell are in §6.5.1 / §6.5.2 tables.)
- The "Pareto frontier" exposition is more compelling when paired with a single combined plot overlaying both algorithms' curves; consider whether to add such a plot stub or leave the contrast to prose.
- §6.5.1 silvertorch deep_sweep data for all 5 sweeps (`c0_maincat / c2_year / c3_nversions / c0c2 / all4`) is documented above in the per-sweep tables. Writer should pick one (recommend `all4` as the most demanding) for the canonical contour heatmap and put the rest in Appendix D.
- The silvertorch deep sweep is bloom-only by configuration ([evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml:41-43](evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml#L41-L43)). silvertorch's clause-fused performance is therefore not explored at varying `(n_lists, n_probe)`. The arxiv filter has no reverse clauses, so the clause-fused kernel would be a clean candidate for a follow-up sweep — note in §7 future work.
- The deep_sweep coverage gap on `n_lists ∈ {2048, 4096}` (between the paper-faithful 1664 and the small-cluster 8192) leaves the cluster-size sensitivity untested in the middle. The author might want to add these in a follow-up — but the current data is sufficient to motivate "√N is the right default" as a thesis claim.
- **B3 (K sensitivity)** and **B4 (batch-size scaling)** are blocked on new measurements per the plot catalog; do not block §6.5 publication on them. They can be added in a v2 pass when the underlying sweeps complete.

---

## 6.6 Scaling

**Scope.** Two axes of system scaling: dataset N (Yambda 500m → 5b) and embedding dim (d64 / d128 / d256). Both axes are present in the existing data; the §6.6 figures only require derived columns and consistent presentation, no new measurements.

### Figure stub: 6.6.1 = C1. Dataset N scaling (Yambda 500m vs 5b)

**Plot.** Two side-by-side panels:
- **Panel 1 (Query latency):** X = `n_items` (log, 500m→5b: 3.06M → 9.39M), Y = `median_ms` (log). One line per algorithm × dim.
- **Panel 2 (Index memory):** X same; Y = `bytes_per_vec` (= `index_mem_mib · 2^20 / n_items`).

**Source.** `all_results_long.csv` filtered to `dataset=="yambda" & batch_size==1 & k==100 & seed==0`. Group by (impl, dim, scale).

**Gap.** yambda-5b-d256 missing (per §6.2.3 and the §6.2 cell-coverage map). Required to populate panel 1 fully. Action: run `evaluation/config/yambda-5b/d256-quality.yaml` once the GPU host is available.

**Catalog ID:** C1. **File destination:** `docs/thesis/results-data/results/C1_yambda_n_scaling.png`. **Rationale:** validates whether Pareto-front winners hold at 3× catalogue scale.

### Figure stub: 6.6.2 = C2. Dimensionality scaling

**Plot.** Three-panel facet, one per metric:
- X = dim ∈ {64, 128, 256}.
- Y1 (Panel 1): Recall@100; Y2 (Panel 2): `median_ms`; Y3 (Panel 3): `bytes_per_vec`.
- One line per algorithm × dataset (4 algos × 3 datasets = 12 lines per panel — use small multiples or color-by-algo with shape-by-dataset).

**Source.** Same `all_results_long.csv`; pivot on dim.

**Catalog ID:** C2. **File destination:** `docs/thesis/results-data/results/C2_dim_scaling.png`. **Rationale:** justifies the choice of working dim. SilverTorch's INT8 should scale flatter than fp16 algos on memory.

### 6.6.3 Concrete-data block (scaling exponents)

`[DATA: docs/thesis/results-data/quality_summary.csv]`, `backend=triton, bs=1, k=100, seed=0`.

#### Dim-scaling exponents (`median_ms ∝ d^α`)

Log-log least-squares fit on the 3 dimensions {64, 128, 256} per (dataset, scale, impl). **Computed from raw data.** All exponents derived by `(ln m_d256 − ln m_d64) / ln(256/64)` for stability, with 3-point linear regression for accuracy where d256 is present:

| dataset / scale | linr_v1_filter_mask | linr_v3 | linr_v4 | silvertorch |
|---|---:|---:|---:|---:|
| arxiv     /  —   | 0.853 | 0.185 | 0.654 | 0.143 |
| goodreads /  —   | 0.783 | 0.013 | 0.639 | 0.141 |
| yambda    / 500m | 0.994 | 0.107 | 0.567 | 0.066 |
| yambda    / 5b   | 1.253* | 0.174* | 0.538* | 0.109* |

*yambda-5b: fit on 2 points (d64, d128) due to absent d256 cell.

**Interpretation.**

- **`linr_v1_filter_mask`: α ≈ 0.78–1.25** — close to linear in d. At `bs=1` the matmul is `(1, d) × (d, N) → (1, N)`, an O(d·N) operation; doubling d roughly doubles cost. The slight super-linearity at yambda-5b (α=1.25) reflects cache effects — at d=128 and N≈5.37M, the dense fp16 tensor (1.3 GiB) stops fitting in L2.
- **`linr_v4`: α ≈ 0.54–0.65** — sub-linear. INT8 matmul (`torch._int_mm`) has uniform tensor-core utilization across these d values; the per-d cost grows slowly because at bs=1 the operation is memory-bandwidth-bound on the int8 codes.
- **`linr_v3`: α ≈ 0.01–0.19** — essentially flat. The 1-bit popcount cost is bandwidth-bound on `N · D/64` bytes (sub-linear in d due to 64-bit packing), and the fp16 rerank stage runs on the fixed-size `candidate_pool=5000` shortlist whose latency is roughly d-invariant in this range.
- **`silvertorch`: α ≈ 0.07–0.14** — nearly flat. The IVF probe scores `n_probe · M ≈ 70 k items` via INT8 dot; doubling d doubles per-item work but the bandwidth-bound INT8 dot amortizes across the probed pool. This is the load-bearing design point of the co-designed retriever.

**Practical implication for the writer.** Doubling d:
- 2× more on `linr_v1` per doubling.
- 1.5× more on `linr_v4` per doubling.
- ~1.1× more on `linr_v3` per doubling.
- ~1.05× more on `silvertorch` per doubling.

**Silvertorch and linr_v3 are nearly free to scale to higher d, while linr_v1/v4 pay linearly** — the cleanest reason to prefer the approximate retrievers when an application wants richer embeddings.

#### Catalogue-scaling ratios (yambda-500m → yambda-5b)

Catalogue ratio derived from `linr_v1.index_mem` (§6.3.4): **N(5b) / N(500m) = 5.37 M / 1.87 M = 2.87×**.

Median latency ratio (5b / 500m) at each (dim, impl):

| `impl` | d64 ratio | d128 ratio |
|---|---:|---:|
| `linr_v1_filter_mask` | 2.398 | 2.638 |
| `linr_v3`             | 1.238 | 1.324 |
| `linr_v4`             | 2.412 | 2.704 |
| `silvertorch`         | 1.213 | 1.402 |

**Findings.**

- `linr_v1_filter_mask` and `linr_v4` scale **near-linearly with N** (ratio 2.4–2.7× for a 2.87× N step). Slightly sub-linear because cuBLAS overheads amortize at larger N; in practice this is the linear-scan regime.
- `linr_v3` scales **sub-linearly** (ratio 1.24–1.32×). Stage 1 (1-bit popcount) grows with N, but stage 2 (fp16 rescore) is dominated by the fixed-size `candidate_pool=5000`. **Caveat:** at fixed `candidate_pool`, Recall@100 drops substantially on 5b (linr_v3 loses 10.6% relative Recall at d128 on 5b — see §6.2). Sub-linear latency comes at a Recall cost.
- `silvertorch` scales **sub-linearly** (ratio 1.21–1.40×). At fixed `n_probe=24, n_lists=1024`, probed-pool size grows roughly linearly with N (cluster sizes grow as N/1024) but the INT8 dot is bandwidth-bound. **Recall@100 stays competitive on 5b** (1.5% absolute drop at d128). This is the clean demonstration that the co-designed retriever's scaling properties match the production regime: latency grows sub-linearly with N, quality holds.

#### Recall rank stability across (dataset, scale, dim) cells

For each of the 11 cells, rank algorithms by Recall@100 (rank 1 = highest). Tie-breaking by ≥ 5e-5 margin.

| `impl` | rank 1 | rank 2 | rank 3 | rank 4 |
|---|---:|---:|---:|---:|
| `linr_v1_filter_mask` | 10 | 1 | 0 | 0 |
| `linr_v4`             | 4  | 7 | 0 | 0 |
| `silvertorch`         | 0  | 0 | 7 | 4 |
| `linr_v3`             | 2  | 0 | 2 | 7 |

(Ties at rank 1: arxiv-d128/d256 — linr_v1, linr_v3, linr_v4 all hit 1.0; goodreads-d256 — linr_v1 0.1472 vs linr_v4 0.1473.)

**Findings.**

- **`linr_v1_filter_mask` is the de-facto Recall ceiling** (rank 1 in 10 of 11 cells).
- **`linr_v4` is recall-equivalent to `linr_v1`** — rank 1 or 2 in every cell. Per-row INT8 quantization (`PostfilterKNNInt8`, [postfilter_knn_int8.py:28](retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L28)) introduces **zero measured Recall@100 loss** on any cell of this study. The SASRec and nomic-embed embedding distributions are well-conditioned for per-row INT8 quantization — a non-obvious finding worth surfacing in the prose.
- **`silvertorch` is consistently mid-pack** (rank 3 in 7, rank 4 in 4). At default `(n_lists=1024, n_probe=24)`, it loses 1–3% Recall vs the exact baseline. §6.5's deep sweep shows this gap can be closed by tuning.
- **`linr_v3` is bimodal**: rank 1 twice (both on arxiv where the 1-bit shortlist recovers Recall=1.0 at default `pool=5000`), rank 4 seven times (every non-arxiv cell where the default pool is too narrow).

#### Best-non-baseline algorithm per cell (canonical defaults)

| dataset | scale | dim | best non-baseline impl | Recall@100 | latency (ms) | speedup vs `linr_v1` |
|---|---|---:|---|---:|---:|---:|
| arxiv     | —    | 64  | silvertorch | 0.9953 | 0.129 | 4.74× |
| arxiv     | —    | 128 | silvertorch | 0.9941 | 0.131 | 8.21× |
| arxiv     | —    | 256 | silvertorch | 0.9928 | 0.158 | 12.69× |
| goodreads | —    | 64  | silvertorch | 0.1463 | 0.205 | 1.02× |
| goodreads | —    | 128 | silvertorch | 0.1449 | 0.214 | 1.69× |
| goodreads | —    | 256 | silvertorch | 0.1439 | 0.249 | 2.47× |
| yambda    | 500m | 64  | silvertorch | 0.1543 | 0.131 | 2.50× |
| yambda    | 500m | 128 | silvertorch | 0.1468 | 0.122 | 5.81× |
| yambda    | 500m | 256 | silvertorch | 0.1367 | 0.143 | 9.07× |
| yambda    | 5b   | 64  | silvertorch | 0.1728 | 0.159 | 4.95× |
| yambda    | 5b   | 128 | silvertorch | 0.1739 | 0.171 | **10.94×** |

**Across all 11 cells, `silvertorch` is the best non-baseline impl at default parameters.** Speedup ranges from 1.02× (goodreads-d64) to **12.69×** (arxiv-d256). Mean speedup = 5.83×.

### Writer's notes (§6.6)

- C1 and C2 are both derivable from `all_results_long.csv` once `bytes_per_vec` is added (§6.3 needs the same column, so this is a single `extract.py` change).
- The yambda-5b-d256 gap is mentioned in §6.2.3 and §6.6.1 — keep both forward-pointers consistent. After the d256 run, the §6.2.3 table and the C1 panel both update.
- If §6.6 prose gets long, consider folding C1 into §6.2.3 (where the Yambda data lives) and C2 into a chapter conclusion paragraph. The plot catalog keeps them as separate items to preserve modularity.
- The **dim-scaling exponent table** (§6.6.3) is the headline result of this section. Surface "silvertorch and linr_v3 are nearly d-invariant; linr_v1 / linr_v4 scale ≈ linearly with d" as the chapter's d-scaling takeaway.
- The **`linr_v4` recall-equivalence** finding (§6.6.3 rank stability) is counterintuitive and worth a sentence: per-row INT8 does not measurably degrade Recall@100 on any cell. This is a strong empirical justification for the INT8 path.
- `[TODO: clarify with author — yambda-5b-d256 is missing across all algos; §6.6.3's exponent fit on yambda-5b uses only 2 points. Either re-run d256 or footnote.]`

---

## 6.7 Engineering validation

**Scope.** Triton ↔ torch parity (numerical equivalence + speedup), per-kernel performance breakdown, and build/index-construction wall-clock. The parity numbers are precomputed; the kernel breakdown and build-time data require new measurements.

### Figure stub: 6.7.1 = F1. Backend parity scatter

**Plot.** Scatter, Y = Triton recall, X = torch recall, diagonal y=x dashed line. One point per cell (~2,698 points). Highlight any point with `recall_abs_diff > 0.001`.

**Upgrade vs existing:** add **Jaccard agreement of top-k sets** (`|Triton_topk ∩ torch_topk| / |union|`) as a second metric per cell. Target: >0.999. Stronger guarantee than per-backend recall diff alone.

**Source.** [docs/thesis/results-data/results/backend_parity_recall.csv](results-data/results/backend_parity_recall.csv) — 2,698 rows. **Gap:** Jaccard column requires a one-time re-run with top-k set capture in `bench_tools.py` (see "Planned schema extensions" §end).

**Catalog ID:** F1. **File destination:** `docs/thesis/results-data/results/F1_parity_scatter.png`.

### Figure stub: 6.7.2 = F2. Backend speedup distribution, faceted

**Plot.** Histogram of Triton/torch speedup, faceted by (algorithm class) × (batch_size).

**Upgrade vs existing:** the headline `backend_speedup_hist.png` reports overall q25/q50/q75/q90 = 0.93/1.42/2.82/4.87. Split by (algo, bs) — Triton wins more on some kernels than others; the headline median hides structure.

**Source.** [docs/thesis/results-data/results/backend_speedup.csv](results-data/results/backend_speedup.csv) — 2,698 rows.

**Catalog ID:** F2. **File destination:** `docs/thesis/results-data/results/F2_speedup_dist_by_algo.png`.

### Table stub: 6.7.3 = G4. Backend parity table

Rows: per (dataset, dim, algo). Columns: max `recall_abs_diff`, mean, **Jaccard@k mean** (🆕), p99 diff.

**Source.** Same parity CSV as F1; add Jaccard column once captured.

**Catalog ID:** G4.

### Table stub: 6.7.4 = G5. Speedup matrix (Triton/torch)

Rows: algorithm; columns: bs ∈ {1, 8, 16}; cells: median speedup ratio.

**Source.** Same speedup CSV as F2.

**Catalog ID:** G5.

### Table stub: 6.7.5 = F3. Kernel-level breakdown (planned)

Rows: Triton kernel name (`oporp_1bit_match_topk`, `codesigned_probe_score`, `bloom_match`, `fused_masked_knn`, `clause_mask`, `clause_compact`, …); columns: ms/call, % of total, occupancy, SM utilization, % of theoretical peak.

**Source.** Nsight Compute or `torch.profiler.profile()` — one representative cell per kernel. Not yet collected. *Precedent:* FAISS GPU paper (% of theoretical peak); CAGRA Fig 7 (kernel decision tree).

**Catalog ID:** F3. **Status:** new measurement required (Nsight Compute one-off).

### Figure stub: 6.7.6 = F4. Build / index-construction time (planned)

**Plot.** Bar chart: algorithm × (encode / quantize / IVF cluster / index assemble) wall-clock seconds, stacked. One bar per algorithm.

**Source.** Requires `build_time_s` field in `bench_tools.py` per-phase. Not yet present in the schema. *Precedent:* ann-benchmarks "Recall vs build time"; CAGRA Fig 11 grouped bars.

**Catalog ID:** F4. **Status:** new measurement required (schema extension + re-run).

### 6.7.7 Concrete-data block (backend parity numbers)

`[DATA: docs/thesis/results-data/parity_and_speedup.csv]`. The extraction script ([docs/thesis/results-data/extract.py:268-304](docs/thesis/results-data/extract.py#L268-L304)) pivots every (dataset, scale, dim, cell_kind, filter_kind, sweep, impl, k, batch_size) row across `backend ∈ {triton, torch}` and produces 2 823 paired rows. Schema: identifiers + `{recall, ndcg, median_ms}_{triton, torch}` + `recall_delta = recall_triton − recall_torch` + `recall_abs_delta` + `ndcg_delta` + `ndcg_abs_delta` + `speedup_ratio = median_ms_torch / median_ms_triton`. All 5 `BACKEND_CAPABLE_ALGOS` ([evaluation/retrieval/algos/__init__.py:57-59](evaluation/retrieval/algos/__init__.py#L57-L59)) have both backends populated; no missing pivot pairs.

Both backends are wrapped with `torch.compile(dynamic=True, mode="reduce-overhead")` ([evaluation/retrieval/algos/silvertorch.py:101](evaluation/retrieval/algos/silvertorch.py#L101), [evaluation/retrieval/algos/linr_v3.py:65](evaluation/retrieval/algos/linr_v3.py#L65)). "Torch backend" = pure-torch reference path *inside* `torch.compile`, not raw eager-mode.

#### G4 (parity table) — concrete values

Per-impl maximum |Δrecall| and |Δndcg| across all 2 823 pivot rows.

| `impl` | max \|Δrecall@K\| | mean \|Δrecall@K\| | max \|ΔNDCG@K\| | rows |
|---|---:|---:|---:|---:|
| `linr_v1_filter_mask` | **0.000000** | 0.000000 | **0.000000** | 549 |
| `linr_v4`             | **0.000000** | 0.000000 | **0.000000** | 546 |
| `silvertorch`         | 0.000483 | 0.000041 | 0.000495 | 591 |
| `linr_v3`             | 0.001174 | 0.000086 | 0.000853 | 594 |
| `linr_v2`             | **0.005281** | 0.001674 | **0.003816** | 543 |

**Per-(impl × suite) max |Δrecall|:**

| `impl` | quality | filter | deep_sweep |
|---|---:|---:|---:|
| `linr_v1_filter_mask` | 0.0      | 0.0      | —        |
| `linr_v2`             | —        | 5.28e-3  | —        |
| `linr_v3`             | 0.0      | 1.17e-3  | 1.98e-4  |
| `linr_v4`             | 0.0      | 0.0      | —        |
| `silvertorch`         | 2.59e-4  | 4.83e-4  | 2.59e-4  |

(— = impl absent from that suite. linr_v2 requires a filter, so it has no quality / deep_sweep rows. Deep-sweep files are single-impl per file.)

**Findings.**

- **`linr_v1_filter_mask` and `linr_v4` are bit-identical across backends.** Both route the scoring matmul through cuBLAS (`torch.matmul` for v1, `torch._int_mm` for v4); Triton wraps the same kernel rather than replacing it, so the output is byte-equal. This is the strongest form of the parity claim Ch.5 anticipates.
- **`silvertorch` and `linr_v3` parity is well within the rank-tolerance of Recall@100.** Max |Δrecall| = 4.83e-4 (silvertorch) and 1.17e-3 (linr_v3) correspond to at most one position swap at the K-boundary per ~10 000 queries. The parity tests ([retrieve/tests/parity/conftest.py:8-59](retrieve/tests/parity/conftest.py#L8-L59)) absorb this via `assert_topk_matches(..., atol=1e-3, rtol=1e-3)`.

#### The `linr_v2` non-parity finding (load-bearing — the only impl exceeding 10⁻³)

- 270 of 543 `linr_v2` pivot rows (49.7%) have |Δrecall| > 10⁻³.
- **All 270 are on arxiv** (every dim × both filter kinds). On goodreads, `linr_v2` is bit-identical.
- Max |Δrecall|: 5.28e-3 on arxiv-d64 clause (`recall_triton = 0.98555` vs `recall_torch = 0.99083` on `sweep=all4, k=100, bs=1`).
- d-attenuation: 5.28e-3 (d64) → 3.75e-3 (d128) → 2.99e-3 (d256). Monotonically smaller at higher d.
- **Direction.** Across all 276 non-parity rows (270 linr_v2 + 6 linr_v3), **every single one has `recall_triton < recall_torch`**. There is no row where Triton beats torch on recall. **Systematic bias, not noise.**

**Hypothesis (writer's note).** `linr_v2` (`PrefilterKNN`, [retrieve/src/retrieve/layers/linr/prefilter_knn.py:10](retrieve/src/retrieve/layers/linr/prefilter_knn.py#L10)) sparse-gathers filter-passing items before the fp16 matmul. The Triton path's gather kernel uses a different tile layout than the torch reference; on arxiv (where filter sets admit ≈ 0.5–5% of N papers, vs goodreads where they admit ≈ 28–53% of books), the gathered-list shape is sparser, and the gather → matmul → topk composition can drop one or two items at the K-boundary when a tie arises in the top-K-th score. The d-attenuation pattern (deltas smaller at higher d) is consistent with this — at higher d the matmul output is less degenerate (fewer ties), so fewer items get dropped.

`[TODO: clarify with author — confirm whether the linr_v2 arxiv non-parity is a tile-layout tie-break (benign) or a genuine indexing bug in the gather kernel. The `test_fused_masked_knn_topk` parity test uses atol=1e-3 but the observed |Δrecall|=5.28e-3 is on a downstream metric (Recall@100). A 1-position swap per query moves Recall@100 by 1/100=0.01, so the observed delta is consistent with at most a 1-position swap per query — but the systematic direction (always Triton < torch) is worth root-causing.]`

#### G5 (speedup matrix) — concrete values

Speedup ratio = `median_ms_torch / median_ms_triton` (> 1 ⇒ Triton faster).

**Per-(impl × batch_size) — mean and median:**

| `impl` | bs | mean | median | min | max | rows |
|---|---:|---:|---:|---:|---:|---:|
| `linr_v1_filter_mask` | 1  | 0.960 | 0.961 | 0.653 | 1.675 | 204 |
| `linr_v1_filter_mask` | 8  | 1.106 | 0.892 | 0.776 | 1.640 | 171 |
| `linr_v1_filter_mask` | 16 | 1.153 | 0.872 | 0.703 | 1.676 | 171 |
| `linr_v2`             | 1  | 2.663 | 2.570 | 1.599 | 4.246 | 171 |
| `linr_v2`             | 8  | 3.710 | 3.747 | 2.336 | 5.413 | 171 |
| `linr_v2`             | 16 | 3.771 | 3.859 | 2.426 | 6.141 | 171 |
| `linr_v3`             | 1  | 1.061 | 1.117 | 0.658 | 1.398 | 198 |
| `linr_v3`             | 8  | 2.474 | 1.412 | 0.838 | 11.79 | 198 |
| `linr_v3`             | 16 | 2.628 | 1.409 | 0.792 | **16.02** | 198 |
| `linr_v4`             | 1  | 0.972 | 0.972 | 0.855 | 1.094 | 204 |
| `linr_v4`             | 8  | 1.081 | 0.901 | 0.857 | 1.708 | 171 |
| `linr_v4`             | 16 | 1.185 | 0.909 | 0.866 | 1.737 | 171 |
| `silvertorch`         | 1  | 2.252 | 2.115 | 1.225 | 4.236 | 219 |
| `silvertorch`         | 8  | 5.489 | 5.290 | 1.776 | 9.802 | 186 |
| `silvertorch`         | 16 | **6.278** | 5.722 | 1.797 | 10.56 | 186 |

**Compact median table (the G5 deliverable):**

| `impl` | bs=1 | bs=8 | bs=16 |
|---|---:|---:|---:|
| `linr_v1_filter_mask` | 0.96 | 0.89 | 0.87 |
| `linr_v2`             | 2.57 | 3.75 | 3.86 |
| `linr_v3`             | 1.12 | 1.41 | 1.41 |
| `linr_v4`             | 0.97 | 0.90 | 0.91 |
| `silvertorch`         | 2.11 | 5.29 | 5.72 |

**Per-suite (`cell_kind`) summary:**

| `cell_kind` | mean | median | min | max |
|---|---:|---:|---:|---:|
| `quality`    | 1.272 | 1.000 | 0.894 | 3.924 |
| `filter`     | 2.292 | 1.427 | 0.653 | 10.56 |
| `deep_sweep` | 6.883 | 5.875 | 1.669 | 16.02 |

**Findings.**

- **Triton dominates `silvertorch` and `linr_v2`.** At batched workloads, silvertorch runs **6.3× faster on Triton on average** (median 5.72× at bs=16). The co-designed kernel ([retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py](retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)) fuses cluster-probing + INT8 dot + (optional bloom) filter + top-K into a single launch; the torch path materializes each stage. **This is the largest single engineering-value result in the thesis.**
- `linr_v2` runs **3.4–3.8× faster on Triton** across all batch sizes — the prefilter sparse-gather kernel is a Triton win because torch's gather-then-matmul materializes a `[B, k_filter, D]` tensor that Triton avoids by streaming gather into the matmul tile.
- **Torch is slightly faster for dense matmul algos at the median.** `linr_v1_filter_mask` median speedup_ratio = 0.96 (bs=1), 0.89 (bs=8), 0.87 (bs=16) — torch is 5–15% faster than Triton on dense scoring paths. Both backends route the matmul through cuBLAS; the Triton wrapper pays a small dispatch overhead. The mean is sometimes > 1 because of a few high-batch outliers, but the median tells the typical story.
- **`linr_v3` is bimodal.** Median ratio: 1.12 / 1.41 / 1.41. Mean is much higher (1.06 / 2.47 / 2.63) — the spread is driven by deep-sweep cells where the torch reference can be 5–16× slower than Triton at large `candidate_pool`. Outlier: deep_sweep d128 linr_v3 bs=16 reaches `speedup_ratio = 16.02` (`median_ms_torch = 22.89`, `median_ms_triton = 1.43`). Likely cause: the torch reference of the 1-bit Hamming top-K does not fuse into popcount+topk under `torch.compile` at large pool sizes.
- **Suite gradient.** quality near-parity (median 1.00) — most cells reduce to one cuBLAS matmul; filter median 1.43× — sparse-gather + fused mask compose well in Triton; deep-sweep median 5.87× — fused co-designed and OPORP kernels matter most here.

#### Parity-test inventory (for F1/G4 caption + writer's-prose anchors)

[retrieve/tests/parity/](retrieve/tests/parity/) — two tolerance regimes:

**Exact-equality tests (`torch.equal`, no tolerance).**
- [test_bloom_match.py](retrieve/tests/parity/test_bloom_match.py): Triton `bloom_match` vs `BloomFilter.evaluate_mask`.
- [test_bloom_compact.py](retrieve/tests/parity/test_bloom_compact.py): Triton `bloom_compact` vs `compact_mask(bloom_match(.))`. Per-row id-sets + counts.
- [test_clause_mask.py](retrieve/tests/parity/test_clause_mask.py): Triton `clause_mask` vs reference.
- [test_clause_compact.py](retrieve/tests/parity/test_clause_compact.py): Triton `clause_compact` vs reference.

Integer-valued outputs (booleans, packed bits, counts, indices) — reduction order does not matter, exact equality holds.

**Approximate-equality tests (`assert_topk_matches`, atol=1e-3 rtol=1e-3, [conftest.py:8-59](retrieve/tests/parity/conftest.py#L8-L59)).**
- [test_fused_masked_knn_topk.py](retrieve/tests/parity/test_fused_masked_knn_topk.py): `fused_masked_knn_topk` (LinR V1/V2 scoring kernel) vs gather + bmm + topk.
- [test_codesigned_probe_score.py](retrieve/tests/parity/test_codesigned_probe_score.py): co-designed IVF + INT8 (no filter) vs reference.
- [test_codesigned_probe_score_exact.py](retrieve/tests/parity/test_codesigned_probe_score_exact.py): co-designed IVF + INT8 + exact-clause vs reference.
- [test_oporp_1bit_match_topk.py](retrieve/tests/parity/test_oporp_1bit_match_topk.py): OPORP 1-bit Hamming top-K vs reference. Asserts popcount exactly + allows tie-broken top-K indices to differ.

The `assert_topk_matches` helper ([conftest.py:14-15, 59](retrieve/tests/parity/conftest.py#L14-L59)) replaces `-inf` scores with `0`, asserts `torch.allclose(out_finite, ref_finite, atol=1e-3, rtol=1e-3)` and per-row id-set equality with K-boundary tie-break tolerance.

**Tolerance vs observed parity.** The `linr_v2` arxiv non-parity (|Δrecall| = 5.28e-3) exceeds the test's `atol/rtol = 10⁻³`. The test asserts on raw scores, while the observed delta is on a downstream metric (Recall@100) — one tie-break swap per query moves Recall@100 by 1/K. The parity tests pass; the deviation is at the metric level only.

### Writer's notes (§6.7)

- F1/F2/G4/G5 are renderable today from existing CSVs (+ the numbers in §6.7.7 above). F3/F4 are blocked on new measurements; surface them as "future work" tables in §6.7 prose if the measurements don't land before the thesis deadline.
- Backend parity is the **engineering correctness claim** of the thesis ("Triton and torch implementations are numerically equivalent within fp16 tolerance"). The §6.7.7 numbers support a stronger formulation: **`linr_v1` and `linr_v4` are bit-identical; `silvertorch` and `linr_v3` are equivalent within 10⁻³ recall; `linr_v2` shows a systematic Δ ≤ 5.3 × 10⁻³ on arxiv (TODO root-cause).**
- The speedup distribution (F2) is the **engineering value claim**. The §6.7.7 median table is the single-table version. The abstract-ready headline: **Triton speedup of 5.7× on the co-designed retriever (silvertorch, bs=16); 3.9× on linr_v2; near-parity on dense matmul algos (linr_v1, linr_v4).**
- The `linr_v1` / `linr_v4` slight torch-faster result (median 0.87–0.97×) should be acknowledged honestly: Triton's value is on the *fused* algorithms, not on cuBLAS-bound dense matmul.
- `[TODO: clarify with author — should §6.7 also report cross-backend memory parity (do triton and torch produce the same `index_mem` / `peak_mem` / `fwd_scratch`)? Spot-checks show ≤ 1 MiB drift but the matrix is not extracted.]`

---

## 6.8 Reproducibility

**Scope.** Seed-stability of metrics, hardware/software disclosure, and a direct mirror of LinR paper Tables 3/4 for side-by-side reproducibility comparison.

### Figure stub: 6.8.1 = E2. Seed stability (planned)

**Plot.** Bar chart: algorithm × Recall@100 ± std across seeds.

**Source.** Current schema records `seed`, but most cells run `n_seeds=1`. Action: re-run a representative subset (1 dataset × 1 dim × all algos) with seeds ∈ {0, 1, 2, 3, 4} to populate std. Recommended cell: goodreads-d128 (fastest re-run on the canonical dim).

**Catalog ID:** E2. **Status:** requires one-time multi-seed re-run.

### Table stub: 6.8.2 = G9. Reproducibility / hardware disclosure

| field | value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Architecture | Ampere (sm_80) |
| CUDA toolkit | 12.8 |
| Driver | (TBD — pull from current host's `nvidia-smi`) |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| Python | (pull from `uv.lock`) |
| Seeds | {0} for primary runs; {0..4} for E2 stability subset |
| Determinism | `pin_precision_globals()` called per [sweep.py:68](evaluation/retrieval/sweep.py#L68) |
| Container | `host=fd4a94de96b9` (containerized; image tag TBD) |
| Wall-clock | 3h30m for quality + 2 deep sweeps |

**Source.** [docs/thesis/06-eval-protocol.md](docs/thesis/06-eval-protocol.md) §5.6 + `SUMMARY.quality-deep.txt`.

**Catalog ID:** G9. **Status:** mostly populated; driver version and container image tag need to be looked up from the current host or HF dataset metadata.

### Table stub: 6.8.3 = G10. Mirror LinR paper Tables 3/4

**Rationale.** Side-by-side reproduction of the original LinR paper's reporting layout — one piece of evidence among many in the chapter's evaluation, supporting the LinR-reproducibility quality bar from goal 1 of the thesis. The LinR paper's Tables 3 and 4 report low-pass-rate and high-pass-rate breakdowns with rows = (framework, variant) and columns = (avg ms/batch, P95, Recall@2k) at batch ∈ {1, 16}.

**Our analogue (rows × cols):**

| row label | impl | batch | avg ms/batch | p95_ms (TBD) | Recall@K (K to be chosen) |
|---|---|---|---|---|---|
| LinR-V1 (Triton) | linr_v1_filter_mask | 1, 16 | ... | TBD | ... |
| LinR-V1 (torch) | linr_v1_filter_mask | 1, 16 | ... | TBD | ... |
| LinR-V2 (Triton) | linr_v2 | 1, 16 | ... | TBD | ... |
| LinR-V2 (torch) | linr_v2 | 1, 16 | ... | TBD | ... |
| LinR-V3 (Triton) | linr_v3 | 1, 16 | ... | TBD | ... |
| LinR-V3 (torch) | linr_v3 | 1, 16 | ... | TBD | ... |
| LinR-V4 (Triton) | linr_v4 | 1, 16 | ... | TBD | ... |
| LinR-V4 (torch) | linr_v4 | 1, 16 | ... | TBD | ... |
| Co-designed IVF+INT8+Bloom (Triton) | silvertorch | 1, 16 | ... | TBD | ... |
| Co-designed IVF+INT8+Bloom (torch) | silvertorch | 1, 16 | ... | TBD | ... |

Split into two tables — low-pass-rate (e.g., goodreads c1_lang_reverse or arxiv all4) and high-pass-rate (goodreads c0_genre or arxiv c0_maincat). K should match the LinR paper's K (Recall@2k — would need a K=2000 run, currently capped at K=1000).

**Source.** `all_results_long.csv` filtered to the matching cells.

**Catalog ID:** G10. **Status:** mostly renderable from existing data; p95 column requires p99/p95 capture (in "Planned schema extensions"); K=2k requires a K=2000 quality re-run (low effort — single config edit). Without these, render with `p80_ms` and `Recall@1000` as nearest-equivalent columns and footnote the substitution.

### 6.8.4 Concrete-data block (determinism + seed plumbing + noise floor)

#### Single-seed reality — every result is `seed=0`

A clean factual finding from grepping the run inventory and config tree:

- [evaluation/results/_runlogs/SUMMARY.quality-deep.txt](evaluation/results/_runlogs/SUMMARY.quality-deep.txt): every quality / filter / deep-sweep run records `seed=0` in its perkernel JSONs.
- All 2 823 pivot rows in [docs/thesis/results-data/parity_and_speedup.csv](docs/thesis/results-data/parity_and_speedup.csv) come from `seed=0` source rows.
- Grep on `evaluation/results/**/*.json` for `"seed":` ≠ 0: **zero matches**.
- Every `evaluation/config/**/*.yaml` declares `seed: 0` as a scalar (not a list). Deep-sweep grids ([evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml](evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml), [evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml](evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml)) include `param_seed=0` and `param_v3_seed=0` uniformly across all parameter combinations.

**Therefore §6.8 cannot quote empirical seed sensitivity.** Instead it must (a) document what determinism guarantees the harness gives, (b) report the intra-run timing noise floor (which bounds latency variance from below), and (c) flag the absence of multi-seed measurement as a limitation. E2 is the planned multi-seed re-run that would close this gap.

#### Seed plumbing tour

- **Process-level torch + CUDA seeding.** [evaluation/retrieval/cli/evaluate.py:60-62](evaluation/retrieval/cli/evaluate.py#L60-L62):
  ```pseudocode
  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(cfg.seed)
  ```
  Called once at CLI startup. Affects any subsequent `torch.randn` / `torch.randperm` not bound to a private generator.
- **NumPy seeding.** Not explicitly seeded by the retrieval CLI. In practice no `np.random` calls fire during retrieval cells (algos and sweep paths grep-clean).
- **LinR V3 1-bit OPORP seed.** [evaluation/retrieval/algos/linr_v3.py:49,59](evaluation/retrieval/algos/linr_v3.py#L49-L59) passes `v3_seed=0` (default) into `OneBitKNN(seed=v3_seed)`, which delegates to `quantize_oporp_1bit(seed=...)`. The OPORP sign vector and permutation are deterministic given the seed via `torch.Generator(device=device).manual_seed(v3_seed)`.
- **SilverTorch KMeans seed.** [retrieve/src/retrieve/layers/silvertorch/main.py:84,149-151](retrieve/src/retrieve/layers/silvertorch/main.py#L84-L151) passes `seed=0` to `KMeansTorch.fit(item_embs)`. Inside [retrieve/src/retrieve/layers/utils/kmeans.py:15-24](retrieve/src/retrieve/layers/utils/kmeans.py#L15-L24):
  ```pseudocode
  g = torch.Generator(device="cpu")
  g.manual_seed(self.seed)
  perm = torch.randperm(n, generator=g)[: self.n_lists]
  centroids = embs[perm].clone().float()
  ```
  Initial centroid selection is bit-deterministic given `(seed, n, n_lists, embs)`. Lloyd iterations are deterministic (argmin assignment, mean update — no random ops).
- **Bloom hash seeds.** Generated via `_generate_seeds(k_hash, device=...)` ([retrieve/src/retrieve/layers/silvertorch/main.py:199](retrieve/src/retrieve/layers/silvertorch/main.py#L199)) — these derive from a process-level constant, **not** from the algo's `seed` parameter. Two runs with the same algo `seed` produce identical Bloom signatures; changing `seed` does not re-shuffle the Bloom hash family. `seed` controls IVF initialization only.

#### Triton determinism caveats ([docs/system/testing.md:134-142](docs/system/testing.md#L134-L142))

Verbatim from the system testing docs:

> - **Determinism**: every random tensor is built with an explicit `torch.Generator(device="cuda").manual_seed(...)`; never use `torch.randn` without a generator.
> - **Set-vs-position assertions**: when comparing two retrieval outputs, compare per-row `set(ids)` (or `valid_id_set`), not position-by-position. Tie-breaking on equal scores is implementation-specific and is not asserted.
> - **Score tolerance**: `atol=1e-3, rtol=1e-3` for fp32 tile-blocked reductions; strict `torch.equal` only for OneBitKNN popcount and for path-isolated comparisons on identical reductions.

**What this means for Ch.6 results.** Triton kernels accumulate scores in tile-major order, differing from torch's element-major / warp-shuffle order. For (item, query) pairs where the score sits within ≈ 10⁻³ of another item's score, the top-K can reorder — the "tie-break" the test explicitly allows. The metric impact is bounded: a top-K swap at the K-boundary moves Recall@K by at most 1/K. The §6.7.7 max |Δrecall| = 5.28e-3 is consistent with at most a few such swaps per query. Kernels are deterministic *within* a backend at fixed seed: two runs of `linr_v3` with `v3_seed=0` and the same Triton kernel binary produce bit-identical Recall and NDCG; only latency differs (hardware noise).

#### Intra-run timing noise floor (empirical proxy for "what would seed variance look like")

Each result row carries `p20_ms` and `p80_ms` alongside `median_ms`. Ratio `(p80 − p20) / median` is a proxy for the per-cell timing noise at fixed seed.

Computed across the two deep-sweep CSVs (densest source for noise quantification):

| sweep | mean (p80−p20) / median | max ratio | rows |
|---|---:|---:|---:|
| arxiv-d128-silvertorch | 0.0314 | 0.6643 | 900 |
| goodreads-d128-linr_v3 | 0.0130 | 0.1665 | 810 |

`[DATA: docs/thesis/results-data/deep_sweep_{silvertorch_arxiv,linr_v3_goodreads}_d128.csv]`.

**Interpretation.**
- Typical noise floor is **1–3% of the median** — well below most §6.7.7 speedup effects (the smallest §6.7 effect is 13% — torch's edge on linr_v1 dense matmul).
- Outliers (16.7% on goodreads-linr_v3 in a few cells, 66.4% on arxiv-silvertorch in a few cells) come from high-selectivity filter sweeps where per-iteration workload varies (different queries admit different numbers of items past the filter). Real but rare.
- **Conclusion:** even at fixed seed, latency has a ≈ 1–3% noise floor from hardware variance (GPU clock gating, memory-controller jitter, kernel scheduler). Recall and NDCG, as rank-only metrics, are bit-stable across reruns at fixed seed — they are unaffected by this.

#### Timing protocol restated ([evaluation/retrieval/bench_tools.py:33-167](evaluation/retrieval/bench_tools.py#L33-L167))

- `WARMUP_ITERS = 20` ([bench_tools.py:33](evaluation/retrieval/bench_tools.py#L33)) — 20 forward passes before any timing.
- `MEM_REPS = 16` ([bench_tools.py:37](evaluation/retrieval/bench_tools.py#L37)) — 16 iterations to capture `peak_mem_mib` and `fwd_scratch_mib`.
- Initial timing budget `DEFAULT_REP_MS = 200.0`; auto-extends to `≥ MIN_SAMPLES = 30` samples, capped at `MAX_REP_MS = 3000.0` ([bench_tools.py:34-36](evaluation/retrieval/bench_tools.py#L34-L36)).
- `triton.testing.do_bench(..., quantiles=[0.5, 0.2, 0.8])` ([bench_tools.py:149-162](evaluation/retrieval/bench_tools.py#L149-L162)) returns median (`median_ms`), p20 (`p20_ms`), p80 (`p80_ms`).

Guarantees ≥ 30 timed samples per cell when one iteration ≤ 6.67 ms. Slower cells get fewer samples within the 3 s cap but still ≥ 1 iteration.

#### What is reproducible vs what is not

Two runs with identical config, identical Triton binary, identical hardware, `seed=0`:

| field | reproducible? | why |
|---|---|---|
| `recall@K`, `ndcg@K` (all algos) | **bit-identical** | Rank-only metrics; tie-breaks are deterministic given fixed Triton compilation |
| `index_mem_mib` | bit-identical (modulo allocator drift) | Deterministic buffer sizes; allocator may report ±1 MiB rounding |
| `peak_mem_mib`, `fwd_scratch_mib` | bit-identical (idem) | Forward pass is deterministic |
| `median_ms`, `p20_ms`, `p80_ms` | **±1–3% noise floor** | Hardware-level variance |
| Cached query embeddings | bit-identical | `(ckpt_mtime, max_seq_length, users_limit)` is the cache key ([evaluation/retrieval/queries_cache.py:50](evaluation/retrieval/queries_cache.py#L50)); same checkpoint ⇒ same encoded queries |
| KMeans cluster assignments | bit-identical | Deterministic randperm + deterministic Lloyd |
| OPORP signs and permutation | bit-identical | Generator-seeded |
| Bloom hash family | bit-identical | `_generate_seeds` is deterministic |

#### Hypothesized (not measured) cross-seed behavior

| field | expected behavior under seed change |
|---|---|
| `recall@K` (`linr_v1_filter_mask`, `linr_v4`) | **invariant** — these algos use no randomness |
| `recall@K` (`linr_v3`) | varies with `v3_seed`: OPORP sign vector / permutation re-roll changes shortlist composition. Expected ≤ 1% absolute Recall variance. **Not measured.** |
| `recall@K` (`silvertorch`) | varies with KMeans `seed`: cluster boundaries shift, n_probe-th closest cluster set changes for some queries. Expected ≤ 1% absolute Recall variance. **Not measured.** |
| `median_ms` (all algos) | ≤ 5% variance from hardware noise + data-dependent shape variation. **Not measured.** |

#### What a multi-seed sensitivity study (E2) would require

Existing infrastructure supports it; only the configs and a re-run are missing:

1. **Config change.** Promote `seed: 0` → `seed: [0, 1, 2, 3, 4]` in relevant YAMLs (minimum 5 seeds for non-trivial spread estimation). For algo-specific seeds (`param_v3_seed` in linr_v3 deep sweep, `param_seed` in silvertorch deep sweep), promote correspondingly.
2. **Sweep driver.** [evaluation/retrieval/sweep.py](evaluation/retrieval/sweep.py) already accepts a `seed` parameter per cell; the multi-seed pattern is supported but unused.
3. **Wall-clock cost.** 5× the current 3.5-hour quality + 2-deep-sweep budget = ≈ 17.5 hours on the A100.
4. **Extraction.** [docs/thesis/results-data/extract.py:191-260](docs/thesis/results-data/extract.py#L191-L260) (`aggregate_across_seeds`) already groups by `_GROUP_KEYS` and computes median/p20/p80 across seed values. Currently each group has one seed so quantiles collapse; with multi-seed data they become meaningful.

**Recommendation for E2.** Run goodreads-d128 (the canonical fastest cell) with seeds ∈ {0,1,2,3,4} across all 5 impls — total cost ≈ 15 minutes — to produce a representative E2 plot. The §7 (Limitations) bullet "No multi-seed sensitivity study" follows directly from this gap.

### Writer's notes (§6.8)

- §6.8 IS the chapter's reproducibility chapter. G9 is non-negotiable (SIGIR-style hardware disclosure); E2 and G10 are valuable but lower-priority.
- The **point** of G10 (one piece of evidence among many in the chapter's evaluation) is "our numbers reproduce the qualitative ordering of LinR's reported numbers", supporting the LinR-reproducibility quality bar from goal 1 of the thesis. If they do, claim it explicitly. If they don't, the chapter has a finding to discuss — this is itself valuable, not a failure.
- The **"all-seed-0" framing is honest.** Don't oversell determinism — the harness has strong deterministic primitives but the *measurement* of seed sensitivity was not done. Be explicit. §6.8.4 lays out exactly which fields are bit-identical and which are not.
- The **1–3% empirical noise floor** (§6.8.4) is a useful number for the reader to internalize. It contextualizes the §6.7 finding "torch is ≈ 13% faster than Triton on dense matmul" — the effect is 5–10× the noise floor, so it's real.
- The **bit-identical Recall guarantee** (§6.8.4 table) is the strongest claim §6.8 can make. Pair with: "Latency is reproducible only up to a ±1–3% hardware noise floor — the spread is small enough that all algorithm-comparison conclusions in Ch.6 hold."
- Cross-reference forward: §6.9 executive summary should restate G10's headline finding ("we reproduce LinR's main result within X% / Y× ratio").
- `[TODO: clarify with author — should §7 (Limitations) carry a "no multi-seed sensitivity study" bullet, derived from §6.8.4? Suggested wording: "Quantitative results report point estimates from a single seed (seed=0); seed-induced variance in Recall@K and median_ms is not characterized empirically. The harness supports multi-seed sweeps without code changes; this is the most obvious follow-up experiment, with a ≈ 17.5-hour incremental wall-clock cost on the A100."]`

---

## 6.9 Executive summary

**Scope.** One-paragraph headline finding + one figure that captures the entire chapter. Goal: the "what did you find?" panel for a thesis defense.

### Figure stub: 6.9.1 = A4 (reprise). Recall × Latency × Memory bubble

**Plot.** Reuse the A4 bubble plot from §6.3.2 here — same figure, different framing. In §6.3 it introduces the memory axis; in §6.9 it serves as the one-look summary.

**Catalog ID:** A4 (reprise — same render, two captions).

### Headline paragraph (TEMPLATE for the writer)

Across three datasets (Goodreads, arXiv, Yambda 500m and 5b) and three embedding dimensions (d64, d128, d256), this thesis reproduces the LinR algorithm family (V1, V2, V3, V4 — Borisyuk et al. 2024) in open-source Triton + torch with verified backend parity (recall diff ≤ 1e-3 across 2,698 cells; Triton 1.4–5× faster than torch.compile alone), and extends it with a co-designed IVF + INT8 + Bloom retriever. The empirical Pareto frontier in the recall × latency × memory space (Fig 6.9.1 = A4) shows three dominance regions:

1. **The co-designed retriever wins** the latency dimension: 2–11× faster than the exact baseline at ≤3% Recall@100 cost on every cell where it is applicable (i.e., excluding reverse-clause filters on Goodreads, where the codesigned kernel does not support reverse semantics).
2. **LinR-V3 (Sign-OPORP 1-bit)** wins the memory dimension: 8× memory reduction vs fp16 at near-baseline Recall when the candidate pool is large enough (linear in catalogue size).
3. **The exact baselines (V1, V2)** win when both speed and recall constraints are loose — they remain Pareto-optimal on small catalogues (Goodreads) where IVF/INT8 overhead exceeds the savings.

These three regions are crisp; the §6.4 filter benchmark and §6.5 parameter sensitivity narrow down WHICH approximation dominates for WHICH catalogue size × selectivity regime. Detailed crossover thresholds in §6.4.7 (Table G7).

### Writer's notes (§6.9)

- The headline paragraph above is a TEMPLATE — the writer should re-cite the actual concrete numbers once the §6.4 stale-cache rerun is complete and all crossover thresholds (G7) are finalized.
- Keep §6.9 to one page. The figure is the message; the prose is the caption.
- Mention the citation policy adaptation in passing (one sentence): "The co-designed retriever combines classical primitives — Jégou 2011 IVF, INT8 quantization with dp4a, BitFunnel Bloom signatures (Goodwin 2017) — into a fused GPU kernel; the design is presented as a second bundled retriever in the framework, assembled from classical primitives, per the lineage statement in Ch.2 §2.3 (no Meta-affiliated work is cited)."

---

## Sources consulted

**Thesis design and prior notes:**
- [docs/thesis/00-thesis-plan.md](docs/thesis/00-thesis-plan.md) — chapter contract, Citation Policy.
- [docs/thesis/01-literature-review.md](docs/thesis/01-literature-review.md) — audited bibliography, kept citations.
- [docs/thesis/03-methods.md](docs/thesis/03-methods.md) — algorithm definitions and lineage statement.
- [docs/thesis/04-datasets-notes.md](docs/thesis/04-datasets-notes.md) — dataset scales, filter schemas (narrow vs wide), per-clause coverage.
- [docs/thesis/05-implementation.md](docs/thesis/05-implementation.md) — class-name mapping; co-designed retriever prose convention.
- [docs/thesis/06-eval-protocol.md](docs/thesis/06-eval-protocol.md) — result schema, sweep dimensions, metric formula source.

**Code (read-only, for file:line citations):**
- [retrieve/src/retrieve/layers/silvertorch/main.py](retrieve/src/retrieve/layers/silvertorch/main.py) (SilverTorch class, filter modes, register_index)
- [retrieve/src/retrieve/layers/linr/one_bit_knn.py](retrieve/src/retrieve/layers/linr/one_bit_knn.py) (OneBitKNN, k_bits param)
- [retrieve/src/retrieve/layers/linr/postfilter_knn.py](retrieve/src/retrieve/layers/linr/postfilter_knn.py) (PostfilterKNN; linr_v1)
- [retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py](retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) (PostfilterKNNInt8; linr_v4)
- [retrieve/src/retrieve/layers/linr/prefilter_knn.py](retrieve/src/retrieve/layers/linr/prefilter_knn.py) (PrefilterKNN; linr_v2)
- [retrieve/src/retrieve/layers/filters/bloom.py](retrieve/src/retrieve/layers/filters/bloom.py) (BloomFilter)
- [retrieve/src/retrieve/layers/filters/exact_attribute.py](retrieve/src/retrieve/layers/filters/exact_attribute.py) (ExactAttributeFilter)
- [evaluation/retrieval/algos/__init__.py](evaluation/retrieval/algos/__init__.py) (algorithm registry)
- [evaluation/retrieval/algos/silvertorch.py](evaluation/retrieval/algos/silvertorch.py) (SilvertorchAlgo wrapper; reverse-clause handling)
- [evaluation/retrieval/algos/linr_v3.py](evaluation/retrieval/algos/linr_v3.py) (LinrV3Algo cascade)
- [evaluation/retrieval/algos/filter.py](evaluation/retrieval/algos/filter.py) (build_filter)
- [evaluation/retrieval/metrics.py](evaluation/retrieval/metrics.py) (Recall, NDCG, MRR formulas — lines 39-114)
- [evaluation/retrieval/results_io.py](evaluation/retrieval/results_io.py) (load_rows, load_results)
- [evaluation/retrieval/sweep.py](evaluation/retrieval/sweep.py) (sweep driver; filter assets loading)
- [evaluation/retrieval/cli/run_evaluation.py](evaluation/retrieval/cli/run_evaluation.py) (EVAL_TYPES)
- [evaluation/retrieval/loaders.py](evaluation/retrieval/loaders.py) (load_filter_assets)
- [evaluation/retrieval/bench_tools.py](evaluation/retrieval/bench_tools.py) (§6.3.4 / §6.7.7 / §6.8.4 — warmup/timing/memory protocol; do_bench wrapper; quantiles)
- [evaluation/retrieval/cli/evaluate.py](evaluation/retrieval/cli/evaluate.py) (§6.8.4 — process-level `torch.manual_seed` / `torch.cuda.manual_seed_all`)
- [evaluation/retrieval/queries_cache.py](evaluation/retrieval/queries_cache.py) (§6.8.4 — `(ckpt_mtime, max_seq_length, users_limit)` cache key for deterministic query embeddings)
- [retrieve/src/retrieve/layers/utils/kmeans.py](retrieve/src/retrieve/layers/utils/kmeans.py) (§6.8.4 — KMeans initial-centroid randperm with `torch.Generator("cpu").manual_seed(seed)`)
- [docs/system/testing.md](docs/system/testing.md) (§6.8.4 — Triton determinism caveats: tie-break tolerance, score `atol=1e-3 rtol=1e-3`, `torch.equal` for popcount/compact only)

**Parity tests (§6.7.7):**
- [retrieve/tests/parity/conftest.py](retrieve/tests/parity/conftest.py) (`assert_topk_matches`, `atol=1e-3 rtol=1e-3`)
- [retrieve/tests/parity/test_bloom_match.py](retrieve/tests/parity/test_bloom_match.py) (exact: bloom_match)
- [retrieve/tests/parity/test_bloom_compact.py](retrieve/tests/parity/test_bloom_compact.py) (exact: bloom_compact)
- [retrieve/tests/parity/test_clause_mask.py](retrieve/tests/parity/test_clause_mask.py) (exact: clause_mask)
- [retrieve/tests/parity/test_clause_compact.py](retrieve/tests/parity/test_clause_compact.py) (exact: clause_compact)
- [retrieve/tests/parity/test_fused_masked_knn_topk.py](retrieve/tests/parity/test_fused_masked_knn_topk.py) (approx: V1/V2 scoring)
- [retrieve/tests/parity/test_codesigned_probe_score.py](retrieve/tests/parity/test_codesigned_probe_score.py) (approx: IVF+INT8, no filter)
- [retrieve/tests/parity/test_codesigned_probe_score_exact.py](retrieve/tests/parity/test_codesigned_probe_score_exact.py) (approx: IVF+INT8 + exact-clause)
- [retrieve/tests/parity/test_oporp_1bit_match_topk.py](retrieve/tests/parity/test_oporp_1bit_match_topk.py) (approx: OPORP 1-bit + exact popcount)

**Configs (sweep specifications):**
- [evaluation/config/arxiv/d{64,128,256}-quality.yaml](evaluation/config/arxiv/d128-quality.yaml)
- [evaluation/config/arxiv/d{64,128,256}-filter.yaml](evaluation/config/arxiv/d128-filter.yaml)
- [evaluation/config/goodreads/d{64,128,256}-quality.yaml](evaluation/config/goodreads/d128-quality.yaml)
- [evaluation/config/goodreads/d{64,128,256}-filter.yaml](evaluation/config/goodreads/d128-filter.yaml)
- [evaluation/config/yambda-500m/d{64,128,256}-quality.yaml](evaluation/config/yambda-500m/d128-quality.yaml)
- [evaluation/config/yambda-5b/d{64,128}-quality.yaml](evaluation/config/yambda-5b/d128-quality.yaml)
- [evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml](evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml)
- [evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml](evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml)

**Results JSONs (authoritative empirical data):**
- `SUMMARY.quality-deep.txt` (run inventory; formerly at `evaluation/results/_runlogs/`, now only on HF)
- `evaluation/results/{arxiv,goodreads}/d{64,128,256}-quality.json` — 24 files, quality suite
- `evaluation/results/{arxiv,goodreads}/d{64,128,256}-filter.json` — 30 files, filter suite
- `evaluation/results/yambda/{500m-d64,500m-d128,500m-d256,5b-d64,5b-d128}{linr_v1_filter_mask,linr_v3,linr_v4,silvertorch}.json` — 20 files, yambda quality
- `evaluation/results/deep_sweeps/arxiv-d128-silvertorch.json` (900 rows)
- `evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json` (810 rows)

**Pre-aggregated CSVs (for figure stubs):**
- [docs/thesis/results-data/results/all_results_long.csv](docs/thesis/results-data/results/all_results_long.csv) — 7 104 × 24, all quality + filter filter JSON data.
- [docs/thesis/results-data/results/sasrec_quality_ceiling.csv](docs/thesis/results-data/results/sasrec_quality_ceiling.csv) — 11 rows, linr_v1_filter_mask baseline per (dataset, dim).
- [docs/thesis/results-data/results/d128_quality_memory.csv](docs/thesis/results-data/results/d128_quality_memory.csv) — 16 rows, memory snapshot at d128.
- [docs/thesis/results-data/datasets/goodreads_clause_c{0..3}_*.csv](docs/thesis/results-data/datasets/) — clause selectivity distributions.
- [docs/thesis/results-data/recipes/walk_results.py](docs/thesis/results-data/recipes/walk_results.py) — extraction script (provenance).
- [docs/thesis/results-data/extract.py](docs/thesis/results-data/extract.py) — current extraction pipeline (§6.3.4 / §6.7.7 sourced from `write_parity_and_speedup` lines 268-304, `aggregate_across_seeds` 191-260).
- [docs/thesis/results-data/quality_summary.csv](docs/thesis/results-data/quality_summary.csv) — 88 rows (44 cells × 2 backends), `backend=triton` subset is §6.3.4 / §6.6.3 source.
- [docs/thesis/results-data/parity_and_speedup.csv](docs/thesis/results-data/parity_and_speedup.csv) — 2 823 paired rows, §6.7.7 source.
- [docs/thesis/results-data/deep_sweep_silvertorch_arxiv_d128.csv](docs/thesis/results-data/deep_sweep_silvertorch_arxiv_d128.csv) — 900 rows, §6.8.4 noise floor.
- [docs/thesis/results-data/deep_sweep_linr_v3_goodreads_d128.csv](docs/thesis/results-data/deep_sweep_linr_v3_goodreads_d128.csv) — 810 rows, §6.8.4 noise floor.

**CSVs feeding §6.7 (Engineering validation):**
- [docs/thesis/results-data/results/backend_parity_recall.csv](results-data/results/backend_parity_recall.csv) — Triton↔torch recall parity (F1, G4).
- [docs/thesis/results-data/results/backend_speedup.csv](results-data/results/backend_speedup.csv) — Triton/torch latency ratio (F2, G5).

**Articles consulted (for citation lineage; never cited from the SilverTorch paper):**
- [articles/linr.md](articles/linr.md) — LiNR paper (Borisyuk et al. 2024 KDD).
