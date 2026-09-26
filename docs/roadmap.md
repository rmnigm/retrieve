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

- **Citability of a narrowed campaign.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. User's call,
  2026-09-26: decide after D1's actual report is in front of us, not in
  the abstract — revisit this the moment `bench report` runs on real D1
  output.
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

- [ ] **L4: pad non-power-of-two `D`/`W` to the next power of two in the
  Triton kernels.** Every kernel with a `tl.arange` over the embedding or
  word width needs a power of two (`check_pow2`, `retrieve/src/retrieve/ops/triton/_host.py`),
  raising `ValueError` otherwise: `codesigned_probe_score(_exact)` and
  `fused_masked_knn_topk` on `D`; `oporp_1bit_match_topk` on `W` (from
  `k_bits`, default `= D`); `bloom_match`/`bloom_compact` on `W` (from
  `m_bits`, currently always chosen a power of two by convention, so
  lower real-world severity). Raised in priority 2026-09-26 (user, live
  in the `d1` session): D=192 (yfcc10m) and D=768 (pubmed, now pulled
  into D1 — see D1 below) both fail this way, so `silvertorch`/triton,
  `linr_v2` and `linr_v3` cannot run on either dataset today — only
  `linr_v1_filter_mask` and `silvertorch`/`official` survive. Fix: masked
  padding to the next power of two inside each kernel (pad with values
  that are provably inert to that kernel's reduction — zero for the fp32
  dot products, whatever OPORP's popcount/hamming path needs to exclude
  padding bits from the count), each with its own bit-exact parity gate
  against the unpadded reference and a timing gate (padding must not
  regress the already-power-of-two cases). **Blocks D1's yfcc10m and
  pubmed `filter` legs** — land before those run, same `code_version`
  economics as L1/L2/L3 (fix now, while the campaign is early, rather
  than after — a later fix invalidates every record collected up to
  that point regardless of dataset).
- [ ] **D1: run the full campaign on the harness.** Re-sequenced live by
  the user, 2026-09-26, directly in the `d1` campaign session (not
  through the orchestrator — reconciled here after the fact): `filter`
  runs first on every dataset, in order — goodreads, arxiv, yfcc10m,
  pubmed, kuairand (openalex's `filter` leg stays with E5, not pulled
  in) — then `deep` and `codesign` (S9) run on arxiv only, not on
  goodreads too. This absorbs pubmed's and kuairand's `filter` legs from
  E5 (E5 narrows to openalex's `filter` leg only). Seeds {0, 1, 2} on the
  headline sweeps; `n_probe` in {24, 32}. **Blocked on L4** (below): D=192
  (yfcc10m) and D=768 (pubmed) are not powers of two, so `silvertorch`
  triton, `linr_v2` and `linr_v3` currently fail at the op boundary on
  both — confirm L4 lands before those two legs run, or they will record
  `status: failed` for most of their algo matrix. Q1-Q4, G-a, G-d, L1/L2, the S9
  path for L1's fp32-output `mm`/`bmm`, which had no CPU kernel and broke
  the harness's CPU-only test suite — found by the S9 worker, fixed
  2026-09-26, harness suite back to 255 passed / 1 skipped) landed first
  (orchestrator re-sequencing, 2026-09-26:
  the only reason to run D1 before a library change was to avoid
  invalidating a campaign in flight, not a data dependency, so doing the
  code changes once and D1 once afterward avoids ever rerunning it).
  Consequence: the 126 previously-committed goodreads `filter`-leg records
  (seed 0) carry a pre-L1/L2 `code_version` and will be re-run by
  `--resume` (~19 GPU hours; quality is expected identical except
  `linr_v1_filter_mask`, which moves by design per L1 — see
  [validation](validation.md#datasets); `official`/`silvertorch` timings
  faster per the kernel-opt artifacts). YFCC's exact-algorithm gate is
  expected to pass now (L1 fixed it in the library suite,
  `recall_oracle@1000` 0.994 on one manual cell) — confirm on the actual
  D1 YFCC cell rather than assuming. `codesign`'s one smoke cell (S9
  worker, 2026-09-26) found `full` *faster* than `partial` on goodreads —
  the opposite order to the paper's §4.4 — on one seed, one sweep,
  unlocked clocks: not a finding, confirm or overturn on D1's real sweep.
  Gate per stage: `bench report` with no missing cells; `median_ms(bs=16)
  < 16 × median_ms(bs=1)`; ids identical across modes; a rerun
  byte-identical in quality. Closes paper gaps G3 (P99 / QPS), G4 (seeds),
  G7, G8 (cross-dataset deep sweeps).
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
- [ ] **E5: run the campaign on openalex, extend the report**, including
  the unfiltered cells retired from the `quality` suite. Needs D1, E2,
  E3. Pubmed's `filter` leg and KuaiRand's `filter` leg were pulled
  forward into D1 directly by the user, 2026-09-26 (live re-sequencing in
  the `d1` campaign session, not through the orchestrator) — **the
  earlier "KuaiRand excluded for now" decision is reversed**: E4's
  gSASRec checkpoint is still weak (val NDCG@10 0.0361, test 0.0088 — a
  4× drop only partly explained by item cold start,
  [validation](validation.md#datasets)), and that caveat was not
  restated to the user at the point they added it — flagged here so it
  isn't lost, not to relitigate the choice. `deep` and `codesign` still
  run on one dataset only (arxiv, user's choice) — E5's scope narrows to
  openalex's `filter` leg only.

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
- `bench/report.py` appends a false provenance sentence ("These records
  predate the D1 campaign...") to every non-citable report.
- `partial` is stamped per process (`bench/run.py`, the `reasons0` list):
  an eager-only pass marks every record `partial`, including `official`,
  whose graph entry would be `not_capturable` anyway.
- `bench upload`'s LFS path is unexercised above 37 MB; the D1 sidecars
  (~450 MB) will hit it. The 73 MB samples sidecar in git belongs on the
  Hub.
- `bench report` has not been rerun over all 126 goodreads cells.
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
L4 ─> D1 ─┬─> D2, D3 ─┐
          ├─> F2      ├─> F5 ─> G-c
          ├─> F4      │
          └─> E5 (E2, E3 done; E4 done but excluded from E5's scope)
TF-3/TF-4 retune ─> rerun the head-to-head
```

GPU steps still open: L4, D1, D2, D3, E5, G-b. Everything else runs
on CPUs beside them.
