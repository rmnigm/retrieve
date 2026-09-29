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
- **Kuairand's width in the `filter` suite (E4).** The suite's `dims`
  are intersected with each dataset's (`bench/config.py` `load_matrix`),
  and kuairand is now d64 only; adding 64 to the suite would also pull
  goodreads d64 into it, breaking "each dataset contributes one width".
  Options: a per-dataset width in `suites.yaml`, or kuairand run with an
  explicit `bench run --dim 64` outside the matrix. Blocks E4's leg.
- **Official vs Triton on goodreads (F2).** The head-to-head's goodreads
  numbers are on the gSASRec embeddings; rerun them on the new encoder, or
  keep them labelled as a gSASRec-embedding measurement
  ([validation](validation.md#encoder-switch-evals-to-redo)).
- **HF: squash `pinkmeme/eval-goodreads-work-id`** to reclaim 2.3 GB
  (`HfApi().super_squash_history(repo_id=..., repo_type="dataset")`); the
  seqrec line's cleanup left it to the user.
- **`.git` history size.** H1 removed the tracked JSON/JSONL results from
  `HEAD`, but their bytes stay in history; shrinking `.git` needs a
  history rewrite (`git filter-repo`), which forces every clone and
  worktree to re-sync ([decisions](decisions.md#harness)).

## Phase H: harness prerequisites for the encoder switch

The harness encodes goodreads, yambda-500m and kuairand with the E1c
checkpoints since the `dev/hstu` merge; what that invalidates is in
[validation](validation.md#encoder-switch-evals-to-redo). These two steps
come before any sequential-dataset cell runs again.

- [ ] **H2: the record key carries the input identity; the golden gate
  pins its checkpoint.** `bench/records.py` `KEY_FIELDS` has no checkpoint
  or input fingerprint, so `--resume` would skip a goodreads cell on the
  new encoder as `ok` and `aggregate` would let it overwrite (or be
  overwritten by) the gSASRec record. Add an `inputs` identity to the key
  block (the checkpoint's ckpt-id for sequential datasets, the
  `content_dir` for text ones; a digest of the item/query tensors if the
  name is not enough) and a schema bump; old records read as their
  gSASRec / content identity. The golden goodreads cells must keep running
  on `gsasrec-d128-drop0.5-id`: give the golden gate a way to pin that
  checkpoint (a checkpoint override on `bench run`, or a golden dataset
  file) instead of following `config/goodreads.yaml`. Gate: the harness
  suite green; a test that two records differing only in checkpoint get
  different resume keys; the golden gate rerun on the pinned checkpoint
  gives today's numbers (validation row *Golden baseline*). CPU plus one
  short GPU golden run. Harness code: `fable` per the orchestration
  contract if it grows past a key-field change, else `opus`.
- [ ] **R1: stage the E1c checkpoints on the box and prove the harness
  reads them.** `hf download` goodreads and yambda-500m
  `checkpoints/sasrec-ssm-logq-d{64,128,256}` into
  `$RETRIEVE_DATA_ROOT/<dataset>/checkpoints/`; `bench check --dataset
  goodreads` at d128; one `--skip-perf` goodreads cell
  (`linr_v1_filter_mask`/triton `c0_genre`) to confirm the encode path
  (`encode_split` with a `normalize=true` config, fresh encode cache,
  fresh oracle blob) and that the exact gate (`recall_oracle@k_max ≥
  0.99`) holds on normalized embeddings. Record the query norms (1.0 under
  `normalize`) and the cell in validation. Needs H2.

## Phase D: campaign and baselines (GPU)

- [ ] **D1: run the full campaign on the harness.** Scope (user,
  2026-09-26; the `campaign-d1` chain notes win if this and reality
  disagree): `filter` on **goodreads, arxiv, yfcc10m, pubmed**, in scale
  order, then `deep` and `codesign` (S9) on **arxiv, then goodreads**.
  openalex's `filter` leg is E5's; kuairand is E4's.
  **Goodreads runs on the new encoder** (`sasrec-ssm-logq-d128`, since
  the 2026-09-29 `dev/hstu` merge): its `filter` leg is rerun in full
  (105 cells; `d1/goodreads` on the Hub stays as the gSASRec run, not
  deleted), and its `deep` / `codesign` legs run on the new encoder only.
  The goodreads part needs H2 and R1; the arxiv/pubmed resume below does
  not. Deliverable after the
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

- [ ] **E4: KuaiRand on the E1c encoder.** Needs R1 and the user's
  decision on kuairand's width in the `filter` suite (above). (a) Rebuild
  the eval inputs with `eval-data kuairand all` (~20 min; they are not on
  the Hub) and `hf download pinkmeme/eval-kuairand --include 'trainer/*'`
  for the trainer inputs, plus `test.parquet` / `item_id_map.json` from
  the rebuild. (b) Retrain `sasrec-ssm-logq-d64-trainval` with the flags
  in [its command.sh](artifacts/seqrec-encoder/k64-refit-sasrec-ssm-logq/command.sh)
  (`train_on_val=true`, 4 epochs; 62.7 GB peak on the H100, so it fits an
  80 GB A100; ~20 min there, expect longer on an A100) into
  `data/kuairand/checkpoints/sasrec-ssm-logq-d64-trainval/`. Gate: test
  ndcg@10 / R@100 within ±0.002 of 0.0276 / 0.0063 (the refit is seeded;
  a miss is reported, not tuned). Keep it off the Hub (user). (c) `bench
  check --dataset kuairand`, then the `filter` leg at d64. Replaces the
  stale gSASRec `t_cat1` cell. GPU.
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
  p99. Its goodreads numbers are on the gSASRec embeddings: rerun or
  relabel per the user's decision (Needs the user). The section exists ([paper](paper/official-vs-reimplementation.md))
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

## Sequential encoder follow-ups (not scheduled)

- [ ] **KuaiRand d128 memory fix**: a single gather for the shared table's two
  lookups (one dense gradient instead of two) or a sparse / row-wise
  optimizer; d128 runs out of memory without it
  ([probe](artifacts/seqrec-encoder/k128-probe-oom/README.md)). With it,
  kuairand could join the `filter` suite at d128 like the others.
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
- [ ] **Goodreads trainer-input tiebreak**: `cmd_prep` sorts by
  `(user_id, ts)` with no tiebreak, so the order inside timestamp ties
  (and which items fill a 200-window's oldest end) changes run to run
  ([validation](validation.md#trainer-inputs-with-timestamps-data-gates-not-citable)).
  A tiebreak moves the output off the Hub copy.
- [ ] **yambda val/test row order**: the `uid` joins in `cmd_prep` give a
  fresh row order each run; nothing may align to it by position.
- [ ] **`--resume` of a `train_on_val` run on the GPU** has never run.

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
H2 ─> R1 ─┬─> D1 (goodreads part; arxiv/pubmed resume needs neither)
          └─> E4 (also needs the user's width decision)
D1 ─┬─> D2, D3 ─┐
    ├─> F2      ├─> F5 ─> G-c
    ├─> F4      │
    └─> E5 (E2, E3 done; openalex's `filter` leg + report extension)
TF-3/TF-4 retune ─> rerun the head-to-head
```

GPU steps still open: R1, D1, D2, D3, E4, E5, G-b (H2 needs one short golden run). Everything else runs
on CPUs beside them.
