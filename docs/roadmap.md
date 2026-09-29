---
title: roadmap
created: 2026-09-26
updated: 2026-09-29
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, docs/validation.md]
---

# Roadmap: the open queue

The single ordered work queue. It lists only what is still to do; what
already holds is in [validation](validation.md), and the standing
decisions are in [decisions](decisions.md). An agent starting a session
reads [AGENTS.md](../AGENTS.md), then this page, then only the wiki pages
its step names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the step is **removed** from this page
by the orchestrator. Step names (D1, E2, ...) are stable identifiers used
in records, commits and code comments; they are not renumbered.

## Needs the user

- **Citability of a narrowed campaign.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. The user decides
  once `bench report` runs on D1's real output, not in the abstract.
- **E0**: the Semantic Scholar API key is an identity-bound form (E3 runs
  the OpenAlex fallback meanwhile).
- **A4**: merging `staging` into `main` is on hold until the user decides.
- **`.git` history size.** H1 removed the tracked JSON/JSONL results from
  `HEAD`, but their bytes stay in history; shrinking `.git` needs a
  history rewrite (`git filter-repo`), which forces every clone and
  worktree to re-sync ([decisions](decisions.md#harness)).

## Phase D: campaign and baselines (GPU)

- [ ] **D1: run the full campaign on the harness.** Scope (user,
  2026-09-26; the `campaign-d1` chain notes win if this and reality
  disagree): `filter` on **goodreads, arxiv, yfcc10m, pubmed**, in scale
  order, then `deep` and `codesign` (S9) on **arxiv, then goodreads**.
  openalex's `filter` leg is E5's; kuairand is out of D1 and E5
  (`dev/hstu`, a separate session, owns it). Deliverable after the
  `filter` legs, before `deep`/`codesign`: a cross-scale filter comparison
  across algorithms (`recall_oracle`, latency, QPS, memory per algo per
  dataset at the headline operating point). Seeds {0, 1, 2} on the
  headline sweeps; `n_probe` in {24, 32}. `codesign`'s early smoke cell is
  not a finding; the real sweep decides
  ([validation](validation.md#campaign-roadmap-d1-in-progress-not-yet-validated)).
  Gate per stage: `bench report` with no missing cells; `median_ms(bs=16)
  < 16 × median_ms(bs=1)`; ids identical across modes; a rerun
  byte-identical in quality. Closes paper gaps G3 (P99 / QPS), G4 (seeds),
  G7, G8 (cross-dataset deep sweeps).
  **Currently paused, VM stopped** (2026-09-29): paused for a fix pass
  the investigation workers' findings required — the pubmed/triton
  register spill, LiNR V3's OPORP OOM, the bloom-path divergence
  (verdict: not a bug) — merged as `code_version c0e42d1`
  ([validation](validation.md#campaign-roadmap-d1-in-progress-not-yet-validated)
  has the targeted-rerun policy), then the box itself had to stop.
  Everything is saved: every leg's records are verified byte-identical
  on the Hub, nothing exists only locally (`d1/goodreads`, `d1/arxiv`,
  `d1/yfcc10m`, `d1/pubmed`, `d1/arxiv-deep-partial`, plus the campaign's
  own scripts/logs as `artifacts/d1-campaign`, all in
  [hub-index.md](artifacts/hub-index.md)). The exact resume plan (steps
  A-E, precise `bench run` commands, **narrow `--algo`/`--backend`
  scoping — not `bench campaign --resume`, which would rerun everything
  since the resume key includes `code_version`**) is in
  `.chains/campaign-d1/2026-09-29-093000000-d1-saved-before-vm-stop.md`
  (a chain note, not committed — read it on the box that resumes this,
  or restage from the Hub artifact if the box is new). Per the user's
  standing rule, this is a **targeted** rerun of only pubmed's
  `silvertorch/triton` and `linr_v3` filter cells plus arxiv deep's
  remaining `linr_v3` jobs — not a full D1 rerun; every other record is
  proven numerically unaffected and no record is re-stamped.
- [ ] **D2: add Faiss, HNSW, cuBLAS and cuVS baselines as harness
  algorithms.** Faiss-GPU and Faiss-CPU IVF-Flat, HNSW, a cuBLAS
  brute-force floor at matched recall; then cuVS IVF-Flat / IVF-PQ / CAGRA
  with a bitset prefilter (G13) and Filtered-DiskANN or ACORN (G14).
  Required for any submission (G5). Needs D1's records to compare against.
- [ ] **D3: measure bloom false-positive rate and memory against filter
  width**, for both blooms on real attributes (G6; paper claim S8). The
  head-to-head saw zero false positives at `m_bits 1024`, so the sweep
  must go below it.

## Phase E: datasets

- [ ] **E0: request the Semantic Scholar API key** (needs the user). Not
  blocking: E3 runs the OpenAlex fallback. Open in case the user wants
  the proper Semantic Scholar source later.
- [ ] **E5: run openalex's `filter` leg, extend the report with the
  unfiltered cells retired from the `quality` suite.** Needs D1 (E2, E3
  are done). Pubmed's `filter` leg and `deep`/`codesign` belong to D1;
  kuairand is out of E5 (see D1).

## Phase F: the paper

- [ ] **F2: finish the official-vs-reimplementation section** with D1's
  numbers: confidence intervals, paired tests and a second seed, p95 and
  p99. The section exists ([paper](paper/official-vs-reimplementation.md))
  and marks each line that waits on D1. Needs D1.
- [ ] **F4: package the artifacts**: a tagged `torchretrieve` release, a
  Zenodo DOI including the pinned official sdist, Hub datasets, oracles
  and results, a one-command reproduction, an anonymised mirror for
  review. Needs D1 and the user's accounts.
- [ ] **F5: write the paper**, every table produced by `bench report`.
  Needs everything above.

## Phase G: after the paper

- [ ] **TF-3/TF-4 retune**: TF-3 (retune) and TF-4 (`evict_first`,
  0-5 %), second-order after TF-9 (probe layout) and TF-1 (transposed
  bloom index), which landed. Small, low priority. Needs a fresh
  `bench report` head-to-head against the post-kernel-opt code: the
  kernel-opt gate checked only the bloom kernel-only ratio, and the
  end-to-end comparison in
  [validation](validation.md#official-against-our-triton-reimplementation-citable-contested)
  still reflects the pre-kernel-opt code.
- [ ] **G-b: extended experiments**: a synthetic scale ladder to 240M and
  1B items (L4, L5), a controlled pass-rate sweep (the LiNR V1/V2
  crossover), co-design ablation depth, V3 bit width, an extended batch
  grid (G10-G12, G15, G16).
- [ ] **G-c: a resource paper about the library.** After F5.
- [ ] **G-e: the two re-scoped parked plans** (implementation deferred
  until after F5 by the user): `torch.export` of the composites is small
  (the kernel side is export-clean; only three `Tensor | None` forward
  params in `modules/linr.py` remain). The live upsert/delete API
  (`LiveIndexMixin` on `retrieve.modules`) is medium-large, comparable in
  scope to the kernel-opt pass: a new subsystem across five module
  classes and both filters.
- [ ] **TF-10: official capturability.** File the upstream issue: Meta's
  scorer syncs because `fused_kmean_ann_cuda.cu` never passes the explicit
  output size `faster_repeat_interleave` accepts. A patched build may be
  measured only if a reviewer asks, labelled "not the official release".

## Sequential encoder follow-ups (dev/hstu, not scheduled)

- [ ] **KuaiRand d128 memory fix**: a single gather for the shared table's two
  lookups (one dense gradient instead of two) or a sparse / row-wise
  optimizer; d128 runs out of memory without it
  ([probe](artifacts/seqrec-encoder/k128-probe-oom/README.md)).
- [ ] **Atomic `_resume.pt` write** (R10 F1): it is written in place, so a
  crash during the ~45 s write destroys the only resume state. Writing to a
  temporary file and `os.replace` doubles its peak on disk (~98 GB at
  KuaiRand d64).
- [ ] **Window val-day clicks** for `train_on_val` users with more than 200
  of them, as `train.parquet` is windowed: 25 % of KuaiRand's val-day
  transitions are not trained
  ([validation](validation.md#gates)).
- [ ] **Launch overhead / CUDA graphs** in the training step.
- [ ] **goodreads d64 R@100**: −0.0039 against the bar, the one miss of the
  success rule ([final models](validation.md#final-models-the-e1c-recipe)).
- [ ] **Merge dev/hstu into staging**: the user's call.

## Known defects, unscheduled

- Large `.log`/`.txt` dumps elsewhere in `docs/artifacts/` (the biggest:
  two `cute-dsl-scorer` `kernel_only-*.txt` at ~182 KB each, an
  `e3-openalex` convert log at 176 KB, a `cute-dsl-scorer` diagnostic at
  135 KB) weren't touched by H1's cleanup — candidates for the same
  Hub-or-drop treatment if the user wants them gone too.
- `bench/report.py` appends a false provenance sentence ("These records
  predate the D1 campaign...") to every non-citable report.
- `partial` is stamped per process (`bench/run.py`, the `reasons0` list):
  an eager-only pass marks every record `partial`, including `official`,
  whose graph entry would be `not_capturable` anyway.
- `bench/oracle.py`'s `item_embs.t().contiguous()` holds a second full
  fp32 copy of the item table on top of the item table itself, so the
  harness's real per-dataset limit at native width is about half the
  device memory divided by `4·D` bytes, not the full device memory —
  found staging OpenAlex at 768-d (15 M items fit the item table alone
  but not both copies; scoped to 10 M instead, see
  [validation](validation.md#datasets)). A view instead of a contiguous
  copy would remove the second copy; `bench/` is gated, so this needs
  its own check against the existing oracle results and golden cells.
- **`bench campaign`'s default `--timeout` (6 h per group) is far too
  short at scale**: pubmed `filter` cells take 20-46 min each at
  D=768 × 10 M, and an arxiv `deep` `silvertorch` group took ~10 h.
  pubmed's `filter` groups hit the default and lost 4 cells (recoverable
  by `--resume`, at the cost of a manual follow-up pass); D1 restarted
  `deep`/`codesign` with `--timeout 48` before they did. Worth a larger
  default or a dataset/suite-scaled timeout before the next campaign this
  size.
- The shared Inductor cache (`/tmp/torchinductor_root`) does not
  invalidate on a `code_version` change, so a graph-mode harness run
  after a library edit can silently replay stale kernel code (found
  during the kernel-opt pass: four compile tests passed against a stale
  cache and failed correctly against a fresh one). The harness should key
  its cache directory by `code_version`
  ([storage](system/storage.md#environment)).
- `linr_v2` and `linr_v3` diverge from their golden files (recall@100
  4.5e-4 / 1.7e-5), for a cause still unidentified that predates the
  kernel-opt pass ([validation](validation.md#harness-gates)).
- The quality subset is a 10k prefix of the query file, not a seeded
  sample. It is safe on the current datasets (files are shuffled) by
  accident.

## Unmeasured, unscheduled

- LiNR V2 after the backend-parity fix: the arxiv and bloom cells, and
  V3 stage 2 (the same kernel).
- What the old harness's quality pass did to `linr_v4`'s batch; needs the
  frozen golden worktree (`tmp/golden-rederive`).
- `bloom_compact`'s `block_n` has not been retuned for the two-phase
  compaction shape the kernel-opt pass introduced.
- A GPU-kernel-technique survey (2026-09-26, web research) found
  background reading, not scheduled work: a warp-ballot (`__ballot_sync`)
  candidate-selection pattern used across recent GPU-IVF kNN kernels,
  worth a one-time check against whether the probe kernel already does
  something equivalent; a tunable-vectorization GPU bloom filter design
  (arXiv 2512.15595) and a cuckoo-filter alternative (arXiv 2603.15486)
  as citable comparisons for the transposed bloom-index kernel; a
  bucket-based coalesced-access layout for filtered graph search
  (GRAB-ANNS, arXiv 2604.16402) as a citable alternative mechanism to
  the compact CSR-like probe layout; recall-bucketed / Pareto-frontier
  reporting (NVIDIA cuVS Bench methodology) as a possible improvement to
  `bench report`'s recall/latency tables, instead of point comparisons.
  Two citations for the paper: Meta's own public SilverTorch numbers
  (`github.com/meta-recsys/silvertorch`, an Engineering-at-Meta blog
  post) as target figures for
  [official-vs-reimplementation](paper/official-vs-reimplementation.md);
  two ANN-benchmark trustworthiness critiques (arXiv 2507.00379, a
  YDB.tech write-up) for
  [provenance-and-disclosure](paper/provenance-and-disclosure.md). No
  public LiNR reproduction exists anywhere to compare against.

## Dependencies

```
D1 ─┬─> D2, D3 ─┐
    ├─> F2      ├─> F5 ─> G-c
    ├─> F4      │
    └─> E5 (E2, E3 done; openalex's `filter` leg + report extension)
TF-3/TF-4 retune ─> rerun the head-to-head
```

GPU steps still open: D1, D2, D3, E5, G-b. Everything else runs
on CPUs beside them.
