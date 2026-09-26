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
- **The compaction −1 tail contract.** Done at the kernel-opt pass: the
  −1 fill now happens inside the scatter kernel (saves the launch and
  most of the write traffic; `bloom_compact` −4.9 %). Still open: whether
  to drop the −1-tail contract entirely (readers bound by `counts`,
  parity compares `[:counts]`) for the remaining traffic — a further
  contract change, not done here.
- **E0**: the Semantic Scholar API key is an identity-bound form; E3
  is running the OpenAlex fallback instead, not waiting on this.
- **A4**: merging `staging` into `main` is on hold until the user decides.
- **`.git` history size.** H1 (2026-09-26) removed 221 tracked
  JSON/JSONL files from `HEAD` going forward, but a plain `git rm` keeps
  their bytes in history — the `.git` directory itself doesn't shrink.
  Actually shrinking it needs a history rewrite (`git filter-repo` or
  equivalent), which is disruptive on an already-pushed shared branch
  (every existing clone/worktree needs to re-sync). Not done; a separate
  decision from H1 itself.

## Phase D: campaign and baselines (GPU)

- [ ] **D1: run the full campaign on the harness.** The `filter` and
  `deep` suites of `evaluation/config/suites.yaml` over goodreads, arxiv
  and yfcc10m; seeds {0, 1, 2} on the headline sweeps; `n_probe` in
  {24, 32}; the S9 co-design ablation (`OfficialConfig(bloom_path="full")`,
  not yet encoded in `suites.yaml` — needs a config addition first). Q1-Q4,
  G-a and G-d landed first (orchestrator re-sequencing, 2026-09-26: the
  only reason to run D1 before a library change was to avoid invalidating
  a campaign in flight, not a data dependency, so doing the code changes
  once and D1 once afterward avoids ever rerunning it). Consequence: the
  126 previously-committed goodreads `filter`-leg records (seed 0) carry
  the pre-kernel-opt `code_version` and will be re-run by `--resume`
  (~19 GPU hours; quality is expected identical, `official`/`silvertorch`
  timings faster per the kernel-opt artifacts). Gate per stage:
  `bench report` with no missing cells; `median_ms(bs=16) < 16 ×
  median_ms(bs=1)`; ids identical across modes; a rerun byte-identical in
  quality. Closes paper gaps G3 (P99 / QPS), G4 (seeds), G7, G8
  (cross-dataset deep sweeps).
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
  blocking E3 any more: it is running the OpenAlex fallback instead.
  Still open if the user wants the proper Semantic Scholar source later.
- [ ] **E4: stage KuaiRand-27K and train gSASRec over its 32M videos.**
  ETL, config and layout done (staged, `bench check` ok, two filter
  protocols across 7 clause slots). Remaining: the gSASRec checkpoint
  (`reuse_item_embeddings`, ~65.6 GB estimated peak), the Hub publish and
  one filter cell — GPU work, queued behind D1.
  See [datasets](system/datasets.md#kuairand).
- [ ] **E5: run the campaign on the new datasets and extend the report**,
  including the unfiltered cells retired from the `quality` suite. Needs
  D1 and E1-E4.

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

- [ ] **TF-3/TF-4 retune**: TF-9 (probe layout) and TF-1 (transposed bloom
  index) landed in the kernel-opt pass (2026-09-26); TF-3 (retune) and
  TF-4 (`evict_first`, 0-5 %) were second-order after those two and are
  still open — small, low priority. Needs a fresh `bench report` head-to-
  head against the post-kernel-opt code (the kernel-opt gate only checked
  the bloom kernel-only ratio, not the full official-vs-reimplementation
  comparison in [validation](validation.md#official-against-our-triton-reimplementation-citable-contested),
  which still reflects the pre-kernel-opt numbers).
- [ ] **G-b: extended experiments**: a synthetic scale ladder to 240M and
  1B items (L4, L5), a controlled pass-rate sweep (the LiNR V1/V2
  crossover), co-design ablation depth, V3 bit width, an extended batch
  grid (G10-G12, G15, G16).
- [ ] **G-c: a resource paper about the library.** After F5.
- [ ] **G-e: the two re-scoped parked plans** (re-scoping done
  2026-09-26, implementation not started, deferred until after F5 by the
  user): `torch.export` of the composites is now small (the kernel side
  was already export-clean from other work; only three `Tensor | None`
  forward params in `modules/linr.py` remain). The live upsert/delete API
  (`LiveIndexMixin` on `retrieve.modules`) is still medium-large,
  comparable in scope to the kernel-opt pass — a new subsystem across
  five module classes and both filters.
- [ ] **TF-10: official capturability.** File the upstream issue: Meta's
  scorer syncs because `fused_kmean_ann_cuda.cu` never passes the explicit
  output size `faster_repeat_interleave` accepts. A patched build may be
  measured only if a reviewer asks, labelled "not the official release".

## Known defects, unscheduled

- Large `.log`/`.txt` dumps elsewhere in `docs/artifacts/` (the biggest:
  two `cute-dsl-scorer` `kernel_only-*.txt` at ~182 KB each, an
  `e3-openalex` convert log at 176 KB, a `cute-dsl-scorer` diagnostic at
  135 KB) weren't touched by H1's cleanup — candidates for the same
  Hub-or-drop treatment if the user wants them gone too.
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
- The Triton kernels need a power-of-two `D`/`W` (`tl.arange`;
  `ValueError` at the op boundary since the kernel-opt pass, was a
  compiler error before). Found staging PubMed and OpenAlex (both
  D = 768): `codesigned_probe_score*`, `fused_masked_knn_topk` and
  OPORP all need it, so both datasets run SilverTorch on `official`
  only and LiNR V2/V3 not at all
  ([validation](validation.md#library-gates)). Fix is masked padding to
  the next power of two inside the kernel, with its own parity and
  timing gates — worth doing before E5.
- `bench/oracle.py`'s `item_embs.t().contiguous()` holds a second full
  fp32 copy of the item table on top of the item table itself, so the
  harness's real per-dataset limit at native width is about half the
  device memory divided by `4·D` bytes, not the full device memory —
  found staging OpenAlex at 768-d (15 M items fit the item table alone
  but not both copies; scoped to 10 M instead, see
  [validation](validation.md#datasets)). A view instead of a contiguous
  copy would remove the second copy; `bench/` is gated, so this needs
  its own check against the existing oracle results and golden cells.
- The shared Inductor cache (`/tmp/torchinductor_root`) does not
  invalidate on a `code_version` change, so a graph-mode harness run
  after a library edit can silently replay stale kernel code (found
  during the kernel-opt pass: four compile tests passed against a stale
  cache and failed correctly against a fresh one). The harness should key
  its cache directory by `code_version`
  ([storage](system/storage.md#environment)).
- The golden-baseline row in [validation](validation.md#harness-gates)
  was stale: `linr_v2` and `linr_v3` already diverged from the golden
  files before the kernel-opt pass, for a cause still unidentified.
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
    └─> E5 <── E4 (E2, E3 done)
TF-3/TF-4 retune ─> rerun the head-to-head
```

GPU steps still open: D1, D2, D3, E5, the encode/training of E3-E4, G-b.
Everything else runs on CPUs beside them.
