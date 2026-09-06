# Plans roadmap — the master plan

**Read this first.** This is the single ordered work queue for the
repository. Every other document in [docs/plans/](.) is detail for one
phase of this queue; none of them fixes an order, this file does. If you
are an agent starting a session: read [`../../CLAUDE.md`](../../CLAUDE.md),
then this file top to bottom, then only the plan section your step names.

**Conventions.** A plan lives in this directory while it is either a
live instruction set or a record of intent whose validation hasn't
signed off. Once neither is true it moves to [archive/](archive/). System
behaviour is documented in [../system/](../system/), never here — if you
want to know how something *works*, those are the maintained references;
a plan only tells you why it was built that way.

**How to keep this file true.** Each step below has a checkbox, the plan
section it executes, where it runs (`A100` = needs GPU time; `Mac` =
legacy label, needs no GPU time and runs on the box's CPUs — there is no
Mac target since 2026-09-06), its gate, and what it unblocks. When you finish a
step: run its gate, append the validation record to the *plan's own*
record section (the model is
[cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360)),
then flip the checkbox here with the date and commit. Do not start a
step whose dependencies are unchecked. Do not reorder steps without
rewriting the dependency notes. When a phase is fully checked, move its
plan to `archive/` and shorten its entry here to one line under *Done*.

---

## 0. Goal, scheduling rule, branch

**Goal.** A reproducibility paper on SilverTorch (Meta) and LiNR
(LinkedIn) built on this repository: our from-the-paper Triton
reimplementation, Meta's official kernels as the reference, one correct
benchmark harness, public datasets up to the papers' scale.
Plan: [reproducibility-paper.md](reproducibility-paper.md) (venue
research lives there; it does not order anything here).

**Scheduling rule.** There are no dates in this file. Every step is to
be done, in the order given; nothing is optional, deferred, or
conditional on a deadline. The A100 is the bottleneck, so GPU steps are
ordered first and Mac-side steps run in parallel with them.

**Branch.** `development` = `main` + the refactor track (formerly
`refactor/kernels-eval`) + the CUDA and CuTe SilverTorch backends
(formerly `feat/cute-dsl-scorer`) + today's plans; 24 commits ahead of
`main`, nothing merged. Those two feature branches were deleted on
2026-09-05 after confirming `development` contains them. Step A4 merges
`development` into `main`. After that, one branch per phase off `main`.

**Decisions already taken (2026-09-05), do not reopen.** Meta's
`meta-recsys/silvertorch` ops become the reference backend
(`backend="official"`); our CUDA C++ and CuTe DSL backends are deleted
after the official parity gate; Triton stays and gets the kernel effort;
the harness is rewritten rather than patched. Rationale in the four
plans and in *Done* below.

## 1. The queue

Effort is in focused days from the plans; "A100 h" is wall time on the
box. Plan section references: **H** =
[evaluation-harness-v2.md](evaluation-harness-v2.md), **O** =
[silvertorch-official-integration.md](silvertorch-official-integration.md),
**P** = [reproducibility-paper.md](reproducibility-paper.md) (gaps G1–G16
in its §B.3), **D** = [dataset-candidates.md](dataset-candidates.md).

### Phase A — unblock the tree

- [x] ~~**A0 — make the Mac able to run this repo's Python.**~~ **Dropped
  2026-09-06 (user decision): GPU environments are the only target; Mac
  support is not a goal.** Nothing below depends on it any more — every
  step runs on the GPU box, and the "Mac" label on a step now only means
  "needs no GPU time", i.e. it can run on the box while the GPU is busy.
- [x] **A1 — golden baseline on the old harness.** H §6 WP-0 (A100,
  0.5 d). Done 2026-09-06 on `dev/a1-golden`, commit `9856998`; golden
  JSONs and the run record are under `evaluation/golden/`, the validation record is in
  [evaluation-harness-v2.md](evaluation-harness-v2.md) §6 WP-0. The
  `users_limit` row-count fix landed, plus **two bugs the golden run
  found**, both of which had been hidden by steps 4–7 never having run:
  the fetched datasets are the pre-`3b1b5b3` 1-indexed `[N+1, …]`
  artifacts (loud on goodreads, *silent* on arxiv — `cos(query, target)`
  0.99 → 0.62), and K3's `common.clause_pass` is a `NameError` under
  inductor, which failed every compiled filter algo. Backends are
  `triton` + `torch` only; the cuda / cute columns in H's text are void
  (H's amendment). Of the harness half of
  [refactor-validation-handoff.md](refactor-validation-handoff.md),
  **steps 1, 4 and 7 passed**; **steps 5 and 6 are deferred 2026-09-06
  (user: heavy evals later)** — scripted and ready, see that file's
  status. Unblocks: A4, C4.
- [x] **A2 — pin and build the official package.** O §10 WP-0 (A100,
  0.5 d). `uv sync --extra official`; upstream suite green; sha, `nvcc`
  and build log in `docs/plans/official-silvertorch-artifacts/`. Gate:
  build < 5 min, `torch.ops.st.fused_kmean_ann` exists. Unblocks: A3, B2.
  **Done 2026-09-06, `d2b9248`** (`dev/a0-a3-deps-official`). Pin
  `21aa35e28b6dd9a91e9ee35efb0857715e86bda7`; build 2 m 17 s with nvcc 12.4
  (a *minor* mismatch against the cu128 wheel — warning, not an error, so
  O §11's 12.8 preference relaxes to 12.x); `torch.ops.st.fused_kmean_ann`
  present. Upstream `pytest silvertorch/` **fails at collection** on three
  files whose Buck `load_library` lines survived Meta's `@oss-disable`
  stripping; excluding them, 99 passed / 3 subtests passed, every CUDA test
  included — green for everything the OSS build ships. Two ops
  (`is_topk`, `take_top_k_and_gather_from_main_and_fresh`) are dead source:
  their `.cpp`/`.cu` are not in `setup.py`. The `cute` extra stays until B4
  (rule 5); `official` is added alongside. Record: O §13.1.
- [x] **A3 — official op facts.** O §10 WP-1 (A100, 0.5 d): bit-order
  probe, syncs and launches per op, graph-capture attempt, parse cost,
  the `per_embedding_scale` overflow. Gate: O §3 confirmed or corrected
  in O's record section. Unblocks: B1 (the adapter is written against
  measured facts, not read ones).
  **Done 2026-09-06, `137a3c5`** (`dev/a0-a3-deps-official`). **The answer
  B1 needs: the official bloom mask is HIGH-bit-first — document `d` is bit
  `63 - (d % 64)` of word `d // 64`** (three independent probes agree), so
  take the HIGH-first branch of the two the B1 note below asks for. O §4.2
  (i)–(iii) confirmed, including `per_embedding_scale` returning `inf` in
  every slot at D=128. Corrected: `fused_kmean_ann` costs 3 syncs and 19
  launches, not 2 and ≈ 12; `_with_partial_masks` 4 syncs;
  `bloom_index_search_batch` *captures* into a CUDA graph and then faults on
  replay, so it needs to be on the harness's not-capturable list explicitly
  (D7 unchanged). Parse cost at B=16: 58.7 µs/call, 3.67 µs/query. Record:
  O §13.2; script and raw output in
  [official-silvertorch-artifacts/](official-silvertorch-artifacts/README.md).
- [ ] **A4 — merge to `main`.** After A1 passes: merge `development`
  into `main` (library gates passed 2026-09-02, harness golden passed in
  A1). Everything below happens on phase branches off `main`, merged
  back through `development`.

### Phase B — official backend, parity, deletion

- [x] **B1 — adapter + tests.** O §5, §10 WP-2 (Mac, 2 d): `backend=
  "official"` in `SilverTorch`, `require_official`, parity tests T1–T7.
  Gate: `ruff` clean, suite collects and skips on the Mac. Needs A3.
  Authored 2026-09-06 on `dev/b1-official-adapter` (Mac gate green; A3's
  bit order pinned as `OFFICIAL_BIT_ORDER = "high_first"`, with the
  library review's "now" items — see
  [architecture-review-2026-09-06-library.md](architecture-review-2026-09-06-library.md)).
  **Done 2026-09-06, `aadc380`** (`dev/integration`): GPU gate — 43/43 in
  `test_official.py` on the A100 after 11 **test-side** fixes (T1-exact's
  CSR doc space, T3's mirrored-doc negative control, `-inf`-slot id
  normalisation in T6, the fd-2 sync instrument in T7); no adapter or
  kernel change. Record: O §14.3.
- [x] **B2 — parity gate.** O §10 WP-3 (A100, 0.5 d). Gate: phase-3
  scores `torch.equal` on the int32 path on every regime, bloom ⊇ check
  and FPR at matched memory recorded in O's record section. **Unblocks
  B4 (deletion) — never delete before this is green.**
  **Done 2026-09-06, `2b09f8e` + `0521a67` + `aadc380`** (`dev/integration`,
  nvcc 12.8 build, clocks unlocked — counts and bit comparisons only).
  T1 `torch.equal` vs the reference and vs Triton on all 8 regime cells +
  4 exact-mask cells; bloom ⊇ exact (AND) and ⊆ exact (NOT) with 0 false
  negatives / positives; FPR at matched memory (byte-exact at
  `b_multiplier = m_bits / (max_terms · 5)`) 0.0000 for both blooms on
  synthetic attrs — the real-attribute calibration stays D3; T7 syncs per
  forward 3 / 3 / 7 / 4 (none / exact / bloom-partial / bloom-full),
  identical with `cache_plans` on and off. Full suite **574 passed, 0
  failed, 127 skipped (all cute — extra not installed)**; upstream suite
  99/99 on the 12.8 build; the three pre-existing red cells fixed (fp32
  literal compare → `torch.equal` on the fp32 inputs, ungated cute cell
  gated); the review's deferred `argsort(stable=True)` applied after
  measuring it bit-identical on nine regimes. Record: O §14. **B4 is now
  unblocked.**
- [ ] **B3 — kernel head-to-head, Triton vs official.** O §9a/§9b, WP-4
  (A100, 1 d). Gate: JSON + tables appended to O. This is paper gap G2.
- [ ] **B4 — delete the CUDA C++ and CuTe backends.** O §7, WP-5 (Mac,
  1 d). Tag the parent commit `cuda-cute-backends-final`; move
  [cuda-silvertorch-handoff.md](archive/cuda-silvertorch-handoff.md),
  [cuda-silvertorch-phase2.md](archive/cuda-silvertorch-phase2.md),
  [cute-dsl-scorer.md](archive/cute-dsl-scorer.md) and
  [cute-dsl-scorer-artifacts/](archive/cute-dsl-scorer-artifacts/README.md) to
  `archive/`; rewrite the system docs O §7 lists. Gate: suite green on
  the A100, collect-only on the Mac, `git grep -il "cute\|codesigned_probe_score_cuda"`
  hits only `docs/plans/archive/`. Needs B2.
  **Authored 2026-09-06 on `dev/b4-delete-cuda-cute`, GPU suite pending**
  (`4d92432` deletion, `010681d` `Backend` split — review item 5, plus the
  docs commit): parent `41d4479` tagged `cuda-cute-backends-final`; 3,677
  lines of kernel/host/test code gone, `build_transposed_sigs` moved to
  `bloom_hash.py` for TF-1; CPU gates green (ruff, collect-only 504,
  evaluation suite 93/1, links 0, torch-backend outputs `torch.equal` to
  the parent). Record and the grep-rule reading: O §15. The coordinator
  flips this box after the A100 run.
- [x] **B5 — salt as a buffer.** O §8 TF-2 (Mac, 0.5 h; validate with
  `test_bloom_hash.py` on the A100). Do before any campaign timing.
  Authored 2026-09-06 on `dev/b1-official-adapter` (commit `2dbee72`).
  **Done 2026-09-06** — GPU gate passed in B2's full-suite run 3
  (`test_bloom_hash.py` incl. the buffer-vs-inline bit-equality on CUDA
  and CPU, and every bloom row of `test_silvertorch.py`, green; O §14.2).
  The raw-capture latency claim is unmeasured until WP-7.

### Phase C — harness v2 (Mac work in parallel with B; A100 gate at the end)

Backends everywhere in H are now `triton | torch | official` (H's
amendment). The official backend is eager-only (O D7), so H's `graph`
mode applies to `triton` and `torch` only. **H §8** amends H §2/§3 with
the findings of a survey of ann-benchmarks, big-ann-benchmarks,
VectorDBBench, MTEB, cuVS bench, MLPerf, asv, Criterion and the
results-as-data practice of ClickBench and db-benchmark
([artifacts](evaluation-harness-v2-artifacts/README.md)); §8.4 lists what
each WP below gains. Three change behaviour rather than schema: build-time
params (`n_lists`) sweep separately from query-time params (`n_probe`,
`candidate_pool`), which turns the `deep` sweep's 12 builds into 2; the
resume key includes the library subtree's tree hash, so a kernel change
invalidates a cell instead of silently reusing it; and the campaign's
process boundary moves down to `(dataset, dim, algo, backend)` (H §8.2 K,
user decision 2026-09-06) so no dynamo cache or CUDA graph pool outlives
the backend under test — cross-backend parity moves to a spill file, and
bit-exactness stays where it belongs, in B2's library parity suite.

- [ ] **C1 — `bench.py`, `metrics.py`, `algos.py` + tests.** H §6 WP-1
  (Mac, 2 d). Needs A1 (golden exists). Includes O §6.2 / WP-6: the
  `official` path in the `PATHS` table.
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU tests green,
  GPU gate C4 pending. `algos.py` landed as `algos_v2.py` (the old
  `algos/` package shadows the name until C3 deletes it); `metrics.py`
  rewritten in place with the old per-row API kept as wrappers.
- [ ] **C2 — `config.py`, `data.py`, `oracle.py` + tests.** H §6 WP-2
  (Mac, 1.5 d).
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU tests green,
  GPU gate C4 pending. `config.py` and `oracle.py` rewritten in place with
  the old API kept below a divider / as wrappers; `data.py` is new and
  imports `encode.py` (kept as the one `training.*` boundary). Oracle
  caches are now blob v4 with the fingerprint in the file name.
- [ ] **C3 — `run.py`, `cli.py`, deletions, docs.** H §6 WP-3 (Mac,
  1.5 d): delete the old harness files H §5 lists, rewrite
  [../system/evaluation.md](../system/evaluation.md) to H §2, archive
  [evaluation-refactor.md](archive/evaluation-refactor.md) and the harness half
  of the refactor runbook.
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU e2e green, GPU
  gate C4 pending. `run.py` (the cell loop, JSONL + samples sidecar,
  resume by key + `code_version`, parity spill, failures as `status:
  failed`), `cli.py` (`bench run` / `campaign` / `report` stub),
  `upload.py`; the old harness, its 19 YAMLs and the `evaluate` /
  `run-evaluation` / `stage-results` scripts deleted (H §5);
  `algos_v2.py` → `algos.py`; [../system/evaluation.md](../system/evaluation.md)
  rewritten; validation record in H §9.
- [ ] **C4 — GPU gate.** H §6 WP-4 (A100, 1 d). Gate: quality within
  1e-6 of the A1 golden, graph latency within 5 %, `cudagraph_skips ==
  0`, `jaccard_vs_first@100 == 1.0` torch-vs-triton, one `official` cell
  runs end to end on goodreads-d128 `c0_genre` with jaccard ≥ 0.99 (O
  WP-6's gate). Unblocks: D1.

### Phase D — campaign and baselines (A100)

- [ ] **D1 — full campaign rerun.** H §6 WP-5 + O WP-7 (A100 ≈ 24 h wall
  + 0.5 d): four datasets × dims × `{triton, torch, official}` × seeds
  {0, 1, 2} on headline sweeps, `n_probe ∈ {24, 32}`, the S9 co-design
  ablation cells. Closes in one pass: the goodreads oracle rerun (old
  open item 2), P gaps G3 (P99/QPS), G4 (seeds), G7, G8 (cross-dataset
  deep sweeps), old §4b items 1, 3, 4, 5, 7. Gate: H WP-5's.
- [ ] **D2 — external baselines.** P G5 (A100, 2–3 d): Faiss-GPU
  IVF-Flat, Faiss-CPU IVF-Flat, HNSW, cuBLAS brute-force floor at
  matched recall, as harness algos. Then P G13 (cuVS IVF-Flat / IVF-PQ /
  CAGRA with bitset prefilter) and G14 (Filtered-DiskANN, ACORN on the
  CPU box, or FANNBench's harness on one of our datasets).
- [ ] **D3 — bloom FPR and memory vs width.** P G6 (A100, 1–2 d), on
  both blooms (ours and official), real attributes.
- [ ] **D4 — `report.py`.** H §6 WP-6 (Mac, 1.5 d): thesis/paper tables
  and figures from the JSONL only, plus the methodology paragraph.

### Phase E — datasets (GPU box in parallel with D)

**Decision (2026-09-05, final).** The study's dataset set: **arXiv** and
**Goodreads** (rerun on the new harness, D1), **YFCC-10M**, **PubMed +
MedCPT**, **Semantic Scholar SPECTER2** (with **OpenAlex** as the fallback
if the S2 API key is not granted), **KuaiRand-27K** — three
semantic-search corpora beyond arXiv and one recsys corpus beyond
Goodreads, all with real filters and ready or locally-encodable query
encoders. Dropped: Amazon Reviews 2023 (out in general), Yambda-full and
Cohere Wikipedia (scale without meaningful filters, closed query
encoder; the existing Yambda unfiltered runs stay as they are). Survey
and ingestion plans: [dataset-candidates.md](dataset-candidates.md)
§3–§4.

- [ ] **E0 — request the Semantic Scholar API key** (D §3.7; free
  research partner form). First thing in this phase: it is the long
  pole for E3. If it is refused, E3 runs on OpenAlex.
- [ ] **E1 — YFCC-10M.** D §3.4 / §4.5 (download-bound, ≈ 3 GB; A100
  minutes). Retry the download with the exact URLs in D §3.4 (served on
  2026-09-05). Ingest: uint8 CLIP → fp16/int8 codes; tag bags → the
  narrow clause tensor (cap K per D §3.4 gotcha a); the shipped 100k
  queries with their tag predicates and filtered GT become the query
  set, so this is the one dataset whose filtered ground truth is *not*
  ours. Gate: our exact filtered oracle reproduces the shipped GT on the
  100k queries; one `none` + one filter cell run.
  *Status: download + parse + loader done 2026-09-06, GPU gate pending.*
  (`evaluation/eval_datasets/yfcc.py`, `yfcc_check_gt.py`,
  `config/yfcc10m/d192-filter.yaml`; 200-query CPU subset of the gate
  reproduces the shipped GT bit-exactly. Two deviations from D §3.4 to
  carry into the paper: the tag clause tensor is capped at 32 tags/item,
  which makes the *harness* predicate stricter than the shipped one on
  25.8 % of queries — the shipped GT is validated against the uncapped
  CSR instead — and the harness scores cosine while the shipped GT is
  squared L2. Both in [../system/datasets.md](../system/datasets.md#yfcc10m).)
- [ ] **E2 — PubMed + MedCPT, ~36 M articles.** D §3.8 / §4.1
  (download-bound: 102 GB of 768-d fp32 embeddings + 44 GB of per-PMID
  JSON from the NCBI FTP, public domain, no registration; no encoding).
  PCA to 256 / 128 / 64 for the dim sweep (fit on a 1 M sample, apply on
  GPU, ≈ 1 h); queries encoded locally with the open
  `ncbi/MedCPT-Query-Encoder` and projected with the same PCA:
  NFCorpus / TREC-style biomedical query sets plus item-as-query;
  attributes: MeSH descriptors (multi-valued, ~30 k), year, journal /
  language via the MEDLINE join. Gate: layout on disk, oracle built,
  one `none` + one filter cell.
  **Status: deferred 2026-09-06 (user decision): no PCA, native 768-d only,
  heavy ETL/evals later; loader skeleton on `dev/e2-pubmed`.** What exists:
  `evaluation/eval_datasets/pubmed.py` (download / verify / medline / convert /
  attrs / queries / encode_queries) with CPU tests on synthetic fixtures,
  `config/pubmed/d768-filter.yaml`, the `pubmed` → `pinkmeme/eval-pubmed`
  registry entry (unpublished) and the docs/system/datasets.md section. What is
  not done: no data staged (the ~198 GB raw mirror does not fit the 100 GiB
  `/workspace` quota — see the section for the budget), no PCA anywhere, no
  oracle, no cells. The `--dims 256,128,64` PCA plan in D §4.1 is superseded.
- [ ] **E3 — Semantic Scholar SPECTER2, ~50 M slice of 120 M.** D §3.7
  (`embeddings-specter_v2`: 30 files × 28 GB JSONL ≈ 840 GB for 120 M
  papers, 768-d; `papers` for year / venue / fields of study /
  publication type / open access / citation bucket; `citations` for
  citation-based relevance; SPECTER2 weights are open, so text queries
  encode locally). Stream the 30 files, keep a 50 M slice with
  abstracts + English + year ≥ 2000, PCA to 256 / 128 / 64 as in E2.
  Queries: held-out papers; relevance = cited papers *and* the exact
  filtered oracle; natural filters = year < query year, same field.
  **Fallback if E0 fails: OpenAlex** (D §3.9: same attribute shape,
  citation links, but ~670 GB snapshot pass + ≈ 9–14 A100 h of nomic
  encoding). Gate: layout on disk, oracle built, one `none` + one filter
  cell.
- [ ] **E4 — KuaiRand-27K, 32 M videos.** D §3.2 / §4.3. ETL to the
  harness layout; train gSASRec D=128 with a *shared* item table (two
  32 M-row tables are ~100 GB fp32 + Adam) or train on the 5-core
  subset while indexing all 32 M; attributes: video_type, upload_type,
  category hierarchy, tags, duration / upload-date buckets. Two filter
  protocols: target-derived (optimistic) and business-rule (exclude
  ads, duration bucket; pessimistic, LiNR-style pass-rate tiers). Gate:
  checkpoint on HF, layout on disk, one `none` + one filter cell.
- [ ] **E5 — campaign cells on E1–E4** on the D1 harness state; extend
  the report. Needs D1.

### Phase F — the paper (F1 and F3 can start any time)

- [ ] **F1 — reframe + deviations table.** P G1 (Mac, 1 d): QuantizedIVF
  is SilverTorch Algorithm 1; the deviations table incl. O's findings
  (official bloom hash ≠ ours, official eager-only, the
  `per_embedding_scale` overflow).
- [ ] **F2 — "official vs reimplementation" section.** O §10 WP-9 (1 d),
  from B3 + D1 numbers.
- [ ] **F3 — provenance + hardware/software disclosure.** P G9 (hours).
- [ ] **F4 — artifacts.** P §B.7: tagged `torchretrieve` release, Zenodo
  DOI (incl. the pinned official sdist), HF datasets + oracles + results,
  one-command `reproduce-paper`, anonymised mirror for review.
- [ ] **F5 — write the paper** per P §C.3 research questions and §C.4
  section plan, with every table produced by D4's `report.py`.

### Phase G — kernel follow-ups and extended experiments (after F)

- [ ] **G-a — Triton transposed bloom (TF-1) + retune (TF-3/4).** O §8,
  WP-8 (2 d + A100). Gate: parity bit-exact, bloom kernel-only within
  1.3× of official. Then rerun B3.
- [ ] **G-b — P gaps G10–G12, G15, G16** (synthetic scale ladder to
  240 M / 1 B, controlled pass-rate sweep, co-design ablation depth, V3
  bit width, extended batch grid). G13/G14 are in D2.
- [ ] **G-c — resource paper for the library itself** (P §A.1 lists the
  track), after F5.
- [ ] **G-d — deferred kernel optimizations** (`oporp_1bit_match_topk`
  hardware popcount; allocator hygiene in the LiNR kernels). No timeline.
- [ ] **G-e — parked feature plans**:
  [torch-export-refactor.md](torch-export-refactor.md),
  [live-update-api.md](live-update-api.md). Not started; both carry
  stale-anchor banners; re-scope before executing. Research menu:
  [future-work-and-research.md](future-work-and-research.md).

## 2. Dependencies at a glance

```
(A0 dropped 2026-09-06 — no Mac target)
A1 ─┬─> A4 (merge)
    └─> C1 ─> C2 ─> C3 ─> C4 ─┬─> D1 ─> D4 ─┐
A2 ─> A3 ─> B1 ─> B2 ─┬─> B4   │   D2, D3 ──┼─> F2, F5
                      └─> B3 ──┘            │
B5 (any time before D1)                     │
E0 first; E1–E4 ingest (any time) ─> E5 (after D1) ┘
F1, F3 (any time); F4 (after D1)
```

The A100 critical path is A1 → A2 → A3 → B2 → B3 → C4 → D1 → D2/D3 →
E5. Mac work (B1, B4, B5, C1–C3, D4, F1, F3) fills the gaps; E1–E4
ingestion runs on the box whenever it is otherwise idle.

### 2.1 Where each step runs

The A100 is the bottleneck, so this is the partition to schedule
against. **Mac** means it needs neither a GPU nor a GPU-produced number,
so it can run on the box's CPUs while the GPU is busy. (A0 was dropped
2026-09-06: there is no Mac target, every step runs on the GPU box.)

| GPU box only | Mac, startable after A0 | Mac, waiting on a GPU number |
|---|---|---|
| A1, A2, A3 | C1, C2, C3 (the harness rewrite, 5 d) | A4 — needs A1 |
| B2, B3 | B1 †, B5, D4 | B4 — needs B2 |
| C4, D1, D2, D3 | E0, E1 download + parse ‡, E1–E4 loader code + fixture tests | F2 — needs B3 + D1 |
| E1–E4 encode / PCA / gSASRec / oracle builds, E5 | F1, F3, G-e re-scope, G-a kernel authoring | F4 — needs D1; F5 — needs all |
| G-a validation and retune, G-b | | |

† B1's *shape* — the `backend="official"` branch, `require_official`, the
T1–T7 skeletons — is Mac work. A3 supplies the constants the tests
assert, so write them parameterised over the two candidate bit orders and
let A3 pick one.

‡ YFCC-10M is 2.9 GB in total (**D** §3.4) and ships its own filtered
ground truth, so the download, the `.spmat` parse and the narrow attrs
tensor are Mac work on a 588 GB-free disk; only E1's gate ("our exact
oracle reproduces the shipped GT") needs the box. The other three
datasets are 146 GB (PubMed) to 840 GB (Semantic Scholar) and belong on
the box's disk, but their `eval_datasets/<name>.py` modules are ordinary
Python that can be written and unit-tested here against fixtures —
`eval_datasets/` is 3,875 lines with **no tests at all** today, and Phase
E adds four more loaders to it.

CPU-side critical path, startable at once:
**C1 → C2 → C3** (5 d), with B1, B5, D4, F1, F3 and the E-phase
loaders as filler — ≈ 11 focused days that never touch the A100.

## 3. Superseded and parked

- **Refactor validation runbook** — library steps passed on the A100
  2026-09-02 (see *Done*); harness steps 4–7 are replaced by A1. The two
  implemented plans ([kernels-layers-design.md](kernels-layers-design.md),
  [evaluation-refactor.md](archive/evaluation-refactor.md)) archive with C3.
- **Thesis results expansion (old item 4)** — its 4a schema items are H
  §2/§3, its 4b reruns are D1; the section is gone from this file.
- **Goodreads oracle rerun (old item 2)** — D1.
- **Deferred kernel optimizations (old item 3)** — G-d.

---

## Done

### CuTe DSL SilverTorch backend — implemented 2026-09-02, deleted at B4 (2026-09-06)

`SilverTorch(backend="cute")`, a one-to-one port of the CUDA C++ backend into
NVIDIA's CuTe DSL, bit-exact against it and against Triton; its finding
(kernel-for-kernel parity with C++, the cost being DSL launch overhead,
gone under CUDA-graph replay) is in the archived plan
[archive/cute-dsl-scorer.md](archive/cute-dsl-scorer.md) §5 / §5.1 with
raw outputs in [archive/cute-dsl-scorer-artifacts/](archive/cute-dsl-scorer-artifacts/README.md).
Citable as `retrieve@cuda-cute-backends-final`.

### CUDA SilverTorch backend — implemented 2026-07-06, validated + tuned on A100 2026-09-02, deleted at B4 (2026-09-06)

`SilverTorch(backend="cuda")`, the paper's two-kernel design (transposed
cluster-major bloom index → 1-bit masks → masked `__dp4a` scoring) in
CUDA C++, bit-identical to Triton on scores; its A100 record (47/47 parity,
the memory-level-parallelism fix that took bloom scoring from 147.8 to
58.6 µs kernel-only against Triton's 124.6 µs at B=16, P=58k, D=128) is
[archive/cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360),
phase 2 in [archive/cuda-silvertorch-phase2.md](archive/cuda-silvertorch-phase2.md).
Deleted after Meta's official ops passed the parity gate (B2, O §14);
the transposed-index idea returns to Triton as O §8 TF-1. Citable as
`retrieve@cuda-cute-backends-final`.

### Refactor track — implemented 2026-07-06, library gates passed 2026-09-02, harness gates pending

Two structure-preserving cleanup plans written from a full audit of both
packages. They change no measured numbers and no op schemas. All code
phases plus the `filter=` → `filter_mode=` rename and the K9/E8.2 doc
sweeps are committed on `refactor/kernels-eval`.

- [kernels-layers-design.md](kernels-layers-design.md) — library-side
  dedup and API consistency: fix the broken `tune-kernels` subcommand
  (K1); shared host-wrapper prep/epilogue so validation reaches the
  production path, and the last four `@custom_op` kernels migrated to
  `@triton_op` (K2); shared `@triton.jit` predicate helpers in
  `kernels/common.py` (K3); `masked_topk` plus the `_PackedBitsKNN` base
  that absorbs `SimHashKNN`'s near-copy of `OneBitKNN` (K4); public
  bloom-hash core in `layers/filters/bloom_hash.py` (K5);
  `FilterModule.register_index` kw-only alignment and the minimal
  `RetrievalModule` ABC (K6); the `KernelTuneSpec` registry (K7); new
  tests (K8); doc sweep (K9).
- [evaluation-refactor.md](archive/evaluation-refactor.md) — harness leanness:
  delete the dead `torch_knn` algo, the unreachable CPU-timing path and
  three unused dependencies; quarantine the upload script behind
  `--repo-id`; rename `datasets` → `eval_datasets` (E1); replace the
  deep kwarg threading in `sweep.py` with `SweepContext` /
  `FilterAssets` (E2); typed `RetrievalAlgo` protocol and a declarative
  eligibility table instead of ValueError-as-control-flow (E3);
  `AlgoBase` for the duplicated compile tail (E4); split `bench_tools.py`
  into `measure.py` / `encode.py` / `passes.py` with `PerfStats` /
  `QualityStats` dataclasses (E5); **oracle cache keyed by content
  fingerprint** — the fix for the blocker in Open work item 2 (E6);
  config/loader hygiene (E7); unit tests and the `evaluation.md` rewrite
  (E8).

Both stay in this directory only until validation signs off; archive
them after.

### Stage 1 — Autotune separation (2026-05-22)

Every kernel exposes a `<Name>Config` dataclass + a single curated
`DEFAULT_CONFIG` next to its `@triton.jit` body. Overrides go through a
private `_<name>_impl(..., *, config=None)` so the public op keeps a
fixed schema. Offline tuning via
[tune.py](../../retrieve/src/retrieve/tune.py). `bloom_match` keeps a
hard-coded tile — its per-call width is dictated by `N`. Convention:
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation).

### Stage 2 — `torch.compile` fixes (2026-05-22)

Replaced `@torch._dynamo.disable` on every kernel host wrapper with
proper op registration, so each algo's
`compile(dynamic=True, mode="reduce-overhead")` captures one
cudagraph_trees graph across filter + index + cascade. The compact
kernels were refactored to return full-width `[B, N]` `(ids, counts)`,
removing the `counts.max().item()` host sync.

### Stage 2b — `custom_op` → `triton_op` (2026-05-23, completed by K2)

`@custom_op` is opaque: inductor and `torch.export` cannot see the
underlying `@triton.jit` kernel. Stage 2b flipped the silvertorch and
bit-KNN wrappers to `@torch.library.triton_op` with a textually-inline
`wrap_triton(...)` launch. The filter/compact kernels initially stayed
behind because their shape-branching prep didn't trace cleanly; K2 moved
that branching into the eager `_impl`s and finished the migration.

**All ten Triton kernel ops are `@triton_op`.** The two hand-written
SilverTorch backends added six `@torch.library.custom_op`s on top while
they lived (a C++ extension has no `@triton.jit` body for inductor to
see); B4 removed them, so the totals are back to 10 registered ops across
7 kernel files, all `@triton_op`.

The host-side `if actual_k < k: pad` tail was eliminated rather than
moved caller-side: `oporp_1bit_match_topk_indirect` widens its score
buffer so `topk(k)` always has ≥ k lanes, and the silvertorch wrappers
rely on index-build asserts. The pad branch was dead in production.

---

## Archived

[archive/](archive/) holds plans whose work fully landed and which are no
longer instructions for anyone. Currently: the SilverTorch reverse-clause
wrapper fix (shipped in `cc85d8f`; the remaining sweep rerun is tracked
as Open work item 2).

Also deleted along the way, subsumed by the system docs: the prototype
custom_op migration doc, the per-item research doc, the deferred
mask-compact-kernel doc, the Stage 1 autotune-separation plan, the Stage
2 `02-triton-op-migration.md` plan, and the `plans-silvertorch-backup/`
tree (which held the `ShardedSilverTorch` sketches and a shelved
native-CUDA experiment — that experiment later shipped as
`backend="cuda"` and was deleted again at B4; its plans are in
`archive/`).
