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
section it executes, where it runs (`Mac` = this CUDA-less dev machine,
`A100` = the GPU box), its gate, and what it unblocks. When you finish a
step: run its gate, append the validation record to the *plan's own*
record section (the model is
[cuda-silvertorch-handoff.md §13](cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360)),
then flip the checkbox here with the date and commit. Do not start a
step whose dependencies are unchecked. Do not reorder steps without
rewriting the dependency notes. When a phase is fully checked, move its
plan to `archive/` and shorten its entry here to one line under *Done*.

---

## 0. Goal, deadline, branch

**Goal.** A reproducibility paper on SilverTorch (Meta) and LiNR
(LinkedIn) built on this repository: our from-the-paper Triton
reimplementation, Meta's official kernels as the reference, one correct
benchmark harness, public datasets up to the papers' scale.
Plan: [reproducibility-paper.md](reproducibility-paper.md).

**Deadline.** ECIR 2027 Reproducibility track — abstract **12 Oct 2026**,
paper **19 Oct 2026**, 12 LNCS pages, notification 7 Dec 2026. The
minimal ECIR cut is the paper plan's P0 gap list (its §B.3 and §C.5);
everything marked *SIGIR* below is for the ≈ Feb 2027 full version. Six
weeks from 2026-09-05: the A100 is the bottleneck, so GPU steps are
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
the harness is rewritten rather than patched; the paper targets ECIR
first. Rationale in the four plans and in *Done* below.

## 1. The queue

Effort is in focused days from the plans; "A100 h" is wall time on the
box. Plan section references: **H** =
[evaluation-harness-v2.md](evaluation-harness-v2.md), **O** =
[silvertorch-official-integration.md](silvertorch-official-integration.md),
**P** = [reproducibility-paper.md](reproducibility-paper.md) (gaps G1–G16
in its §B.3), **D** = [dataset-candidates.md](dataset-candidates.md).

### Phase A — unblock the tree (A100, week 1)

- [ ] **A1 — golden baseline on the old harness.** H §6 WP-0 (A100,
  0.5 d). Commit the 3-line `users_limit` row-count fix; run the golden
  cells on goodreads-d128 `c0_genre` (all five algos, `triton` + `torch`)
  and arxiv-d128 `c0_maincat` (`silvertorch`, `triton` only — the cuda /
  cute columns in H's text are void, see H's amendment). Gate: golden
  JSONs committed under `evaluation/golden/`. Unblocks: A4, C4. Also
  closes the harness half of
  [refactor-validation-handoff.md](refactor-validation-handoff.md)
  (steps 4–7) — record the result there.
- [ ] **A2 — pin and build the official package.** O §10 WP-0 (A100,
  0.5 d). `uv sync --extra official`; upstream suite green; sha, `nvcc`
  and build log in `docs/plans/official-silvertorch-artifacts/`. Gate:
  build < 5 min, `torch.ops.st.fused_kmean_ann` exists. Unblocks: A3, B2.
- [ ] **A3 — official op facts.** O §10 WP-1 (A100, 0.5 d): bit-order
  probe, syncs and launches per op, graph-capture attempt, parse cost,
  the `per_embedding_scale` overflow. Gate: O §3 confirmed or corrected
  in O's record section. Unblocks: B1 (the adapter is written against
  measured facts, not read ones).
- [ ] **A4 — merge to `main`.** After A1 passes: merge `development`
  into `main` (library gates passed 2026-09-02, harness golden passed in
  A1). Everything below happens on phase branches off `main`, merged
  back through `development`.

### Phase B — official backend, parity, deletion (weeks 1–2)

- [ ] **B1 — adapter + tests.** O §5, §10 WP-2 (Mac, 2 d): `backend=
  "official"` in `SilverTorch`, `require_official`, parity tests T1–T7.
  Gate: `ruff` clean, suite collects and skips on the Mac. Needs A3.
- [ ] **B2 — parity gate.** O §10 WP-3 (A100, 0.5 d). Gate: phase-3
  scores `torch.equal` on the int32 path on every regime, bloom ⊇ check
  and FPR at matched memory recorded in O's record section. **Unblocks
  B4 (deletion) — never delete before this is green.**
- [ ] **B3 — kernel head-to-head, Triton vs official.** O §9a/§9b, WP-4
  (A100, 1 d). Gate: JSON + tables appended to O. This is paper gap G2.
- [ ] **B4 — delete the CUDA C++ and CuTe backends.** O §7, WP-5 (Mac,
  1 d). Tag the parent commit `cuda-cute-backends-final`; move
  [cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md),
  [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md),
  [cute-dsl-scorer.md](cute-dsl-scorer.md) and
  [cute-dsl-scorer-artifacts/](cute-dsl-scorer-artifacts/README.md) to
  `archive/`; rewrite the system docs O §7 lists. Gate: suite green on
  the A100, collect-only on the Mac, `git grep -il "cute\|codesigned_probe_score_cuda"`
  hits only `docs/plans/archive/`. Needs B2.
- [ ] **B5 — salt as a buffer.** O §8 TF-2 (Mac, 0.5 h; validate with
  `test_bloom_hash.py` on the A100). Do before any campaign timing.

### Phase C — harness v2 (Mac work in parallel with B; A100 gate in week 3)

Backends everywhere in H are now `triton | torch | official` (H's
amendment). The official backend is eager-only (O D7), so H's `graph`
mode applies to `triton` and `torch` only.

- [ ] **C1 — `bench.py`, `metrics.py`, `algos.py` + tests.** H §6 WP-1
  (Mac, 2 d). Needs A1 (golden exists). Includes O §6.2 / WP-6: the
  `official` path in the `PATHS` table.
- [ ] **C2 — `config.py`, `data.py`, `oracle.py` + tests.** H §6 WP-2
  (Mac, 1.5 d).
- [ ] **C3 — `run.py`, `cli.py`, deletions, docs.** H §6 WP-3 (Mac,
  1.5 d): delete the old harness files H §5 lists, rewrite
  [../system/evaluation.md](../system/evaluation.md) to H §2, archive
  [evaluation-refactor.md](evaluation-refactor.md) and the harness half
  of the refactor runbook.
- [ ] **C4 — GPU gate.** H §6 WP-4 (A100, 1 d). Gate: quality within
  1e-6 of the A1 golden, graph latency within 5 %, `cudagraph_skips ==
  0`, `jaccard_vs_first@100 == 1.0` torch-vs-triton, one `official` cell
  runs end to end on goodreads-d128 `c0_genre` with jaccard ≥ 0.99 (O
  WP-6's gate). Unblocks: D1.

**Deadline fallback.** If C4 is not green by the end of week 3, add
mean / p95 / p99 / QPS and per-query latency vectors to the *old*
harness's `measure.py` (H §2 protocol, ½ d) and run Phase D on it; the
rewrite then lands for the SIGIR version.

### Phase D — campaign and baselines (A100, weeks 3–4)

- [ ] **D1 — full campaign rerun.** H §6 WP-5 + O WP-7 (A100 ≈ 24 h wall
  + 0.5 d): four datasets × dims × `{triton, torch, official}` × seeds
  {0, 1, 2} on headline sweeps, `n_probe ∈ {24, 32}`, the S9 co-design
  ablation cells. Closes in one pass: the goodreads oracle rerun (old
  open item 2), P gaps G3 (P99/QPS), G4 (seeds), G7, G8 (cross-dataset
  deep sweeps), old §4b items 1, 3, 4, 5, 7. Gate: H WP-5's.
- [ ] **D2 — external baselines.** P G5 (A100, 2–3 d): Faiss-GPU
  IVF-Flat, Faiss-CPU IVF-Flat, HNSW, cuBLAS brute-force floor at
  matched recall, as harness algos. *ECIR: Faiss + HNSW suffice (P
  §C.5); cuVS / Filtered-DiskANN / ACORN are SIGIR (G13, G14).*
- [ ] **D3 — bloom FPR and memory vs width.** P G6 (A100, 1–2 d), on
  both blooms (ours and official), real attributes.
- [ ] **D4 — `report.py`.** H §6 WP-6 (Mac, 1.5 d): thesis/paper tables
  and figures from the JSONL only, plus the methodology paragraph.

### Phase E — datasets (GPU box in parallel with D)

**Decision (2026-09-05).** The study's dataset set is fixed: **arXiv**
and **Goodreads** (rerun on the new harness, D1), **YFCC-10M**,
**OpenAlex**, **KuaiRand-27K** — two semantic-search corpora and two
recsys corpora beyond arXiv, all with real filters. Dropped: Amazon
Reviews 2023 (out in general), Yambda-full and Cohere Wikipedia (scale
without meaningful filters; the existing Yambda unfiltered runs stay as
they are), PubMed + MedCPT (kept only as the fallback if E2's snapshot
pass proves too heavy). Survey and ingestion plans:
[dataset-candidates.md](dataset-candidates.md) §3–§4.

- [ ] **E1 — YFCC-10M.** D §3.4 / §4.5 (download-bound, ≈ 3 GB; A100
  minutes). Retry the download with the exact URLs in D §3.4 (served on
  2026-09-05). Ingest: uint8 CLIP → fp16/int8 codes; tag bags → the
  narrow clause tensor (cap K per D §3.4 gotcha a); the shipped 100k
  queries with their tag predicates and filtered GT become the query
  set, so this is the one dataset whose filtered ground truth is *not*
  ours. Gate: our exact filtered oracle reproduces the shipped GT on the
  100k queries; one `none` + one filter cell run. **ECIR.**
- [ ] **E2 — OpenAlex, ~50 M works.** D §3.9. Snapshot filter pass first
  (`s3://openalex`, ~670 GB works JSONL, ~1 TB scratch, CPU hours):
  English, has abstract, type ∈ {article, preprint, review}, year ≥
  2000, sample 50 M, keep `referenced_works` among the sample. Encode
  with `nomic-embed-text-v1.5` at 256 tokens (same encoder as arXiv,
  D = 256/128/64 without PCA; ≈ 9–14 A100 h, about half on an H100).
  Attributes: topic hierarchy, year, type, oa_status, language, venue,
  country, citation bucket. Queries: held-out works; relevance =
  cited works (citation-based, real) *and* the exact filtered oracle;
  natural filters = year < query year, same field. Gate: layout on
  disk, oracle built, one `none` + one filter cell. **ECIR if the D1
  campaign is done by week 4, else SIGIR.**
- [ ] **E3 — KuaiRand-27K, 32 M videos.** D §3.2 / §4.3. ETL to the
  harness layout; train gSASRec D=128 with a *shared* item table (two
  32 M-row tables are ~100 GB fp32 + Adam) or train on the 5-core
  subset while indexing all 32 M; attributes: video_type, upload_type,
  category hierarchy, tags, duration / upload-date buckets. Two filter
  protocols: target-derived (optimistic) and business-rule (exclude
  ads, duration bucket; pessimistic, LiNR-style pass-rate tiers). Gate:
  checkpoint on HF, layout on disk, one `none` + one filter cell.
  **SIGIR unless E1/E2 finish early.**
- [ ] **E4 — campaign cells on E1–E3** on the D1 harness state; extend
  the report. Needs D1.

### Phase F — the paper (weeks 4–6; F1 and F3 can start any day)

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
- [ ] **F5 — write, per P §C.4 page budget.** Abstract 12 Oct, paper
  19 Oct 2026.

### Phase G — after ECIR (SIGIR version, ≈ Feb 2027)

- [ ] **G-a — Triton transposed bloom (TF-1) + retune (TF-3/4).** O §8,
  WP-8 (2 d + A100). Gate: parity bit-exact, bloom kernel-only within
  1.3× of official. Then rerun B3.
- [ ] **G-b — P gaps G10–G16** (scale ladder, pass-rate sweep, co-design
  ablation depth, cuVS, Filtered-DiskANN / ACORN, V3 bit width, extended
  batch grid) and whatever of E2/E3 did not make ECIR.
- [ ] **G-c — ECIR 2027 Resource track for the library itself**
  (deadline 2 Nov 2026 — only if F5 lands early; otherwise skip).
- [ ] **G-d — deferred kernel optimizations** (`oporp_1bit_match_topk`
  hardware popcount; allocator hygiene in the LiNR kernels). No timeline.
- [ ] **G-e — parked feature plans**:
  [torch-export-refactor.md](torch-export-refactor.md),
  [live-update-api.md](live-update-api.md). Not started; both carry
  stale-anchor banners; re-scope before executing. Research menu:
  [future-work-and-research.md](future-work-and-research.md).

## 2. Dependencies at a glance

```
A1 ─┬─> A4 (merge)
    └─> C1 ─> C2 ─> C3 ─> C4 ─┬─> D1 ─> D4 ─┐
A2 ─> A3 ─> B1 ─> B2 ─┬─> B4   │   D2, D3 ──┼─> F2, F5
                      └─> B3 ──┘            │
B5 (any time before D1)                     │
E1, E2, E3 ingest (any time) ─> E4 (after D1) ┘
F1, F3 (any time); F4 (after D1)
```

The A100 critical path is A1 → A2 → A3 → B2 → B3 → C4 → D1 → D2/D3 →
E4. Mac work (B1, B4, B5, C1–C3, D4, F1, F3) fills the gaps; E1–E3
ingestion runs on the box whenever it is otherwise idle.

## 3. Superseded and parked

- **Refactor validation runbook** — library steps passed on the A100
  2026-09-02 (see *Done*); harness steps 4–7 are replaced by A1. The two
  implemented plans ([kernels-layers-design.md](kernels-layers-design.md),
  [evaluation-refactor.md](evaluation-refactor.md)) archive with C3.
- **Thesis results expansion (old item 4)** — its 4a schema items are H
  §2/§3, its 4b reruns are D1; the section is gone from this file.
- **Goodreads oracle rerun (old item 2)** — D1.
- **Deferred kernel optimizations (old item 3)** — G-d.

---

## Done

### CuTe DSL SilverTorch backend — implemented and benchmarked 2026-09-02

`SilverTorch(backend="cute")`: a one-to-one port of the CUDA C++ backend
into NVIDIA's CuTe DSL (`nvidia-cutlass-dsl`, optional `cute` extra),
authored and validated on the A100 in one session on
`feat/cute-dsl-scorer`. Same three kernels plus the generic fallback,
same op signatures, same transposed-bloom buffers (imported from the
cuda module, not copied), so a cute checkpoint is byte-identical to a
cuda one; bit-exact against both cuda and Triton on every regime.
`Backend` is now four-valued. The question it answers — kernel size and
speed of the DSL against C++ — is settled in the plan's §5: kernel-only
the two are equal, the port's real cost was the DSL launch (63–72 µs of
host time vs 4–11 µs), trimmed to wall-clock parity at B=16 and gone
under CUDA-graph replay, which is the harness's deployed path.

Plan, decisions, spike findings and the validation record:
[cute-dsl-scorer.md](cute-dsl-scorer.md); raw scripts and outputs:
[cute-dsl-scorer-artifacts/](cute-dsl-scorer-artifacts/README.md);
mechanism: [../system/kernels.md](../system/kernels.md#codesigned_probe_score_cute--the-cute-dsl-backend).

### CUDA SilverTorch backend — implemented 2026-07-06, **validated + tuned on A100 2026-09-02**

A second implementation of SilverTorch's Algorithm 1 phases 2+3 in CUDA
C++, selected by `SilverTorch(backend="cuda")`. Where the Triton kernel
fuses a row-wise bloom read into scoring, this follows the paper: a
transposed, cluster-major bloom index evaluated into 1-bit-per-item
masks, then masked `__dp4a` scoring. Scores are bit-identical to the
Triton backend by construction; ids match up to permutation within tied
scores.

**Phase 2 (2026-09-01)** closed the three gaps that
first pass left, per
[cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md). `filter_mode=
"exact"` now runs on cuda too — as a *second phase-2 mask kernel*
(`cps_clause_mask_kernel`, one warp per output word, two ballots) rather
than a third scoring kernel, since the scorer is filter-agnostic and any
filter is a 1-bit-per-item mask. The scorer gained a config-gated
`UNROLL ∈ {1, 2, 4}` knob (items in flight per segment; `1` is the
original loop, and the arithmetic — hence bit-exactness — is identical at
every value). And two static reviews, one reading the kernels and
launchers like a compiler, one fact-checking the toolchain and API
assumptions against upstream docs, were triaged and applied: an explicit
`nvcc`-major check before the JIT build, `ToolchainMissing` split from a
real build failure so a compile error can no longer masquerade as a test
skip, `if constexpr` where a dead ternary arm was doing out-of-range
pointer arithmetic, and the tie-tolerant id gate above.

**A100 validation (2026-09-02, commit `0f7792c`).** The runbook ran
§4–§6: 47/47 parity tests bit-exact, `IDP.4A` in SASS, no `LDSM` /
`BAR.SYNC` in the scorer. As shipped the CUDA path was *slower* than
Triton at every `B=16` large-`P` regime (one 128 B row per warp in
flight); the fix — `SEG = D/16` lanes per item with one `int4` per lane,
ids prefetched one iteration ahead, a thread-per-slot clause-mask kernel,
`DEFAULT_CONFIG = (128, 8, 1)` — took kernel-only bloom scoring from
147.8 µs to 58.6 µs against Triton's 124.6 µs (B=16, P=58k, D=128) and
left no-filter and exact within ±3 % of Triton. Gates (1)–(3b) pass; (4)
end-to-end was not run (no dataset on the box); ncu was blocked in the
container. Full record:
[cuda-silvertorch-handoff.md §13](cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).

Design and constraints:
[../system/kernels.md](../system/kernels.md#codesigned_probe_score_cuda--the-cuda-c-backend).
Phase-2 plan: [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md).
Validation runbook:
[cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md).

This also made `Backend` three-valued (`"torch" | "triton" | "cuda"`),
which matters beyond SilverTorch: every other layer dispatches
`triton`-or-torch, so `"cuda"` silently resolves to the torch path
there. See
[../system/architecture.md](../system/architecture.md#backend-dispatch).

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
- [evaluation-refactor.md](evaluation-refactor.md) — harness leanness:
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

**All ten Triton kernel ops are `@triton_op`.** The CUDA backend later
added three ops on `@torch.library.custom_op` — deliberately, since a C++
extension has no `@triton.jit` body for inductor to see. Current totals:
13 registered ops across 8 kernel files. (An earlier version of this
roadmap claimed nothing remains on `@custom_op`; that stopped being true
when the CUDA backend landed.)

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
native-CUDA experiment — that experiment is no longer shelved, it
shipped as `backend="cuda"`).
