---
title: roadmap
created: 2026-09-26
updated: 2026-09-26
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

- **Run the campaign?** Running more of D1 needs the user's go-ahead. The
  grid is 450 jobs, about 44 GPU hours at the measured ~537 s per record.
  The choice is the full grid or its headline subset, set against the
  rental horizon.
- **Citability of a narrowed campaign.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. Whether a
  deliberate, recorded narrowing (eager everywhere, graph on triton) should
  read differently changes what the paper may claim.
- **YFCC and fp16 scoring.** `PostfilterKNN` scores in fp16, so the exact
  algorithms fail the `recall_oracle@1000 ≥ 0.99` gate on YFCC (0.964),
  where the top-1000 spans about fifteen fp16 quanta. Options: fp32
  scoring for exact algorithms (a library change, so after D1), or a
  per-dataset gate.
- **The compaction −1 tail (Q4).** Q4 moves the −1 fill into the kernel,
  which saves a launch but not the write traffic. Dropping the contract
  (readers bound by `counts`, parity compares `[:counts]`) would save the
  traffic but changes a gated contract.
- **E0**: the Semantic Scholar API key is an identity-bound form.
- **A4**: merging `staging` into `main` is on hold until the user decides.

## Phase D: campaign and baselines (GPU)

- [ ] **D1: run the full campaign on the harness.** The `filter` and
  `deep` suites of `evaluation/config/suites.yaml` over goodreads, arxiv
  and yfcc10m; seeds {0, 1, 2} on the headline sweeps; `n_probe` in
  {24, 32}; the S9 co-design ablation (`OfficialConfig(bloom_path="full")`).
  The goodreads `filter` leg at seed 0 is done; the arxiv leg, seeds 1-2,
  `deep` and the ablation are not. Gate per stage: `bench report` with no
  missing cells; `median_ms(bs=16) < 16 × median_ms(bs=1)`; ids identical
  across modes; a rerun byte-identical in quality. Closes paper gaps G3
  (P99 / QPS), G4 (seeds), G7, G8 (cross-dataset deep sweeps). Blocks:
  nothing under `retrieve/src/retrieve/` changes until D1 ends
  ([decisions](decisions.md#library)).
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

- [ ] **E0: request the Semantic Scholar API key** (needs the user). The
  long pole for E3; if refused, E3 runs on OpenAlex.
- [ ] **E2: stage PubMed + MedCPT as a 10M slice.** ETL is written and
  dry-run on one shard; native 768-d, no PCA. Budget at 10M: 217.9 GB of
  raw download streamed shard by shard, peak disk 27.2 GB, 15.4 GB of fp16
  items. Gate: layout on disk (`bench check`), oracle built, one filter
  cell.
- [ ] **E3: stage a 50M-paper Semantic Scholar SPECTER2 slice** (OpenAlex
  fallback: a ~670 GB snapshot pass plus 9-14 A100 hours of encoding).
  Abstracts, English, year ≥ 2000; held-out papers as queries; relevance
  from citations and the exact filtered oracle; filters year < query year,
  same field. Needs E0. Gate as E2.
- [ ] **E4: stage KuaiRand-27K and train gSASRec over its 32M videos.**
  A shared item table (two 32M-row tables with AdamW moments are ~100 GB),
  or train on the 5-core subset while indexing all 32M. Attributes:
  video type, upload type, category hierarchy, tags, duration and upload
  date buckets. Two filter protocols: target-derived and business-rule.
  Gate: checkpoint on the Hub, layout on disk, one filter cell.
- [ ] **E5: run the campaign on the new datasets and extend the report**,
  including the unfiltered cells retired from the `quality` suite. Needs
  D1 and E1-E4.

## Phase Q: gate hardening and hygiene

Q1 and Q2 are CPU work and run now, beside the GPU steps, on disjoint
trees. Q3 needs the GPU but leaves `code_version` alone (it touches no
file under `retrieve/src/retrieve/`); it serializes with D1. Q4 changes
the library and waits for D1.

- [ ] **Q1: write the missing rules and fix the doc drift.** Rules in the
  [coding guidelines](contracts/coding-guidelines.md): no dated or
  history comments in code (a measurement goes to
  [validation](validation.md) and its artifacts, the comment cites the
  page or says nothing); what a gate test must assert (a mutated function
  turns it red; no smoke-only or vacuous assertions; rejections checked
  with `pytest.raises(match=)`; degenerate inputs give exact sentinels;
  each tolerance stated at its call site with its reason). In
  [agent orchestration](contracts/agent-orchestration.md) §5: a
  fixed-heading dispatch brief, and what counts as proof per kind of
  change (refactor: the gate is bit-exact before and after; behaviour: a
  test that pins it; performance: before and after with `sm_mhz`, the
  `unstable` flag and the shape, the prediction written before the
  measurement, a slower result recorded as a result; a cross-backend
  ratio sampled interleaved in one process). One home per fact: a renamed
  fact is grepped across `AGENTS.md`, `docs/` and `retrieve/docs/` and
  every copy fixed. In `docs/system`: an "adding an op" checklist in
  `kernels.md`; a code-area to page map in `architecture.md`; the
  `testing.md` tuner count and its rule on private oracles against the
  shared `ops/reference` twin; the contiguity claim in `kernels.md`. The
  README's guide list and install command. Gate: link checker at zero;
  every changed claim checked against the code.
- [ ] **Q2: harden the harness gates and lint.** One reader per
  environment variable, enforced by an AST test (fixes the two
  `RETRIEVE_DATA_ROOT` defaults in `eval_datasets`). Every tree-sweeping
  test asserts it scanned a non-trivial tree and that each allow-list
  entry is still needed. Every `evaluation/config/*.yaml` parses through
  the real loader against every suite. A wider ruff rule set on
  `evaluation/` (bugbear with `zip(strict=)`, comprehensions, simplify,
  unused-`noqa`, blind-except, no inline imports), each per-file ignore
  with its reason, one pinned ruff version, the pre-commit hook covering
  `evaluation/`, plus merge-conflict and large-file (~10 MB) hooks. The
  doc checker also verifies backticked repo paths and `bench` /
  `eval-data` subcommands named in docs, with a found-at-least-N guard.
  `bench env` prints the provenance and clock block as JSON, including
  the installed official commit. Gate: harness suite green on CPU; ruff
  clean; link checker at zero.
- [ ] **Q3: harden the library's correctness gates (GPU).** Top-k parity
  compares the non-finite pattern exactly and finite values separately,
  and prints the mismatches; integer-exact paths use `torch.equal`;
  tolerance arguments have no default. CUDA-graph tests replay with new
  inputs against eager on the same backend, `torch.equal`. Inputs are
  unchanged after every op. Poisoned-output tests for every kernel that
  writes into `torch.empty`, proving the poison was used. Kernel runs on
  both sides of each cutoff read from the code's constants, asserting the
  regime was hit. Degenerate rows give exact sentinels. Bit-exact
  identities between kernel variants and between batched and single-row
  calls. Every registered op has a same-signature reference twin; the
  official op schemas are pinned; `opcheck` on the fake impls. Measured
  HBM copy bandwidth and launch floor on the box, which replace the
  datasheet figure in every efficiency claim. Gate: suite green, or each
  red test reported as a finding (rule 3: nothing loosened). Needs Q1.
- [ ] **Q4: widen kernel offsets to int64 and fix the library findings
  (GPU). Risky.** Offsets overflow int32 at B·N ≥ 2³¹ on outputs and at
  N·row-stride ≥ 2³¹ on item tables (N ≈ 107-134M), inside the paper's
  scale ladder. Widen once on the base pointer, keep lane offsets int32.
  Also: pin float scalar kernel arguments to fp32; let the compaction
  kernel write its own −1 tail instead of a `[B, N]` prefill; reject
  unsupported shapes and non-contiguous item tables at the boundary; the
  tuner keeps the current config within a noise band, caps worst-regime
  regressions and records clocks and commit; the dependency floors equal
  the validated versions; the same ruff widening as Q2 on `retrieve/`.
  Risk: it changes `code_version`, so D1's records no longer measure the
  shipped code. Gate: every parity gate bit-exact; compile gates; a
  large-size test per overflow class (gated on free memory); the golden
  gate; kernel timings before and after, interleaved, within noise, else
  stop and report. Needs D1 and Q3; blocks F4.

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

- [ ] **G-a: the scorer's two measured deficits.** Both wait for D1
  because they change `code_version`; rerun the head-to-head after them.
  - TF-9, probe layout: replace the padded probe table with a CSR or a
    capped-pad layout. The pad tax is why Meta's scorer is 10.9-17.8×
    ours on goodreads (97 % padding) and 1.15-3.2× on arXiv.
  - TF-1, transposed bloom index in Triton: Meta's transposed search beats
    our row-wise one 2.0× at 0.8M items and 6.1× at 3.0M. Estimate: bloom
    kernel-only at batch 16 from 117 µs to about 55 µs. Gate: parity
    bit-exact; bloom kernel-only within 1.3× of official.
  - TF-3 (retune) and TF-4 (`evict_first`, 0-5 %) are second order after
    these two.
- [ ] **G-b: extended experiments**: a synthetic scale ladder to 240M and
  1B items (L4, L5), a controlled pass-rate sweep (the LiNR V1/V2
  crossover), co-design ablation depth, V3 bit width, an extended batch
  grid (G10-G12, G15, G16).
- [ ] **G-c: a resource paper about the library.** After F5.
- [ ] **G-d: deferred kernel work**: a hardware popcount in
  `oporp_1bit_match_topk`; allocator hygiene in the LiNR kernels.
- [ ] **G-e: re-scope the parked plans**: `torch.export` of the composites
  (their single forward signature is what an export needs) and a live
  upsert/delete API (`LiveIndexMixin` on `retrieve.modules`).
- [ ] **TF-10: official capturability.** File the upstream issue: Meta's
  scorer syncs because `fused_kmean_ann_cuda.cu` never passes the explicit
  output size `faster_repeat_interleave` accepts. A patched build may be
  measured only if a reviewer asks, labelled "not the official release".

## Known defects, unscheduled

- `bench campaign --suite all` raises `KeyError: 'quality'`: `SUITES` in
  `evaluation/bench/cli.py` still lists the retired suite
  ([evaluation](system/evaluation.md#cli)).
- `bench/report.py` appends a false provenance sentence ("These records
  predate the D1 campaign...") to every non-citable report.
- `partial` is stamped per process (`bench/run.py`, the `reasons0` list):
  an eager-only pass marks every record `partial`, including `official`,
  whose graph entry would be `not_capturable` anyway.
- `bench upload`'s LFS path is unexercised above 37 MB; the D1 sidecars
  (~450 MB) will hit it. The 73 MB samples sidecar in git belongs on the
  Hub.
- `bench report` has not been rerun over all 126 goodreads cells.
- The quality subset is a 10k prefix of the query file, not a seeded
  sample. It is safe on the current datasets (files are shuffled) by
  accident.

## Unmeasured, unscheduled

- LiNR V2 after the backend-parity fix: the arxiv and bloom cells, and
  V3 stage 2 (the same kernel).
- What the old harness's quality pass did to `linr_v4`'s batch; needs the
  frozen golden worktree (`tmp/golden-rederive`).
- `bloom_compact`'s `block_n` has not been retuned for the two-phase
  compaction shape (with G-a).

## Dependencies

```
D1 ─┬─> D2, D3 ─┐
    ├─> F2      ├─> F5 ─> G-c
    ├─> F4      │
    └─> E5 <── E2, E3 (after E0), E4
D1 ─> G-a (TF-9, TF-1) ─> rerun the head-to-head
Q1 ─> Q3 ─┐
D1 ───────┴─> Q4 ─> F4        Q2: no dependency
```

GPU steps: D1, D2, D3, E5, the encode and oracle builds of E2-E4, Q3, Q4,
G-a validation, G-b. Everything else runs on CPUs beside them.
