---
title: roadmap
created: 2026-09-26
updated: 2026-10-07
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, evaluation/bench/, infra/runpod/, docs/validation.md, docs/artifacts/hub-index.md]
---

# Roadmap: the open queue

The single ordered work queue: what is left to run or implement, in
dependency order. What already holds is in [validation](validation.md),
the standing decisions in [decisions](decisions.md), work that is not
queued in the [backlog](backlog.md). An agent starting a session reads
[AGENTS.md](../AGENTS.md), then this page, then only the wiki pages its
step names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator **removes** the step
from this page. Step names (D1, H2, ...) are stable identifiers used in
records, commits and code comments; they are not renumbered.

**Estimates** are A100-SXM4-80GB GPU-hours (the reference GPU for every
citable number), taken from the per-cell times D1 measured; a *stream* is
a slice of a step that can run on its own GPU at the same time as the
others (see [Multi-GPU execution](#multi-gpu-execution)).

## Needs the user

- **Official vs Triton on goodreads (F2): rerun or relabel.** The
  head-to-head's goodreads numbers are on the gSASRec embeddings; rerun
  them on the E1c encoder (F2-R below), or keep them labelled as a
  gSASRec-embedding measurement
  ([validation](validation.md#encoder-switch-evals-to-redo)).
- **Citability of a narrowed campaign.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. The user decides
  once `bench report` runs on D1's real output (D1-G), not in the abstract.
- **E0**: the Semantic Scholar API key is an identity-bound form (E5 runs
  the OpenAlex fallback meanwhile).
- **A4**: merging `staging` into `main` is on hold until the user decides.
- **HF: squash `pinkmeme/eval-goodreads-work-id`** to reclaim 2.3 GB
  (`HfApi().super_squash_history(repo_id=..., repo_type="dataset")`).
- **`.git` history size.** The JSON/JSONL results left `HEAD`, but their
  bytes stay in history; shrinking `.git` needs a history rewrite (`git
  filter-repo`), which forces every clone and worktree to re-sync
  ([decisions](decisions.md#harness)).

## Running GPU work on a pod

GPU work runs on RunPod pods launched with
[`infra/runpod/pod.sh`](../infra/runpod/pod.sh) (`pod.sh up --gpu a100
--gpus N`); disks and environment in [storage](system/storage.md).

### Restore on a fresh pod

1. Checkout `staging` (the image clones it into `/workspace/retrieve`);
   the venv is `/venvs/retrieve`. A worker that needs a different
   environment gets its own venv (`UV_PROJECT_ENVIRONMENT=/venvs/<name> uv
   sync --all-packages --all-groups --extra official`): a worker that
   re-syncs the shared venv to its worktree breaks every running job.
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads: `arxiv-papers` (the license-fixed
   version: `item_attrs_narrow.pt` min = -1, `eval_split.parquet`
   `query_attrs_narrow` min = 0), `goodreads-work-id` with its checkpoints
   (R1 lists them). PubMed is not on the Hub: rebuild the 10 M slice with
   `eval-data pubmed` ([datasets](system/datasets.md#pubmed); ~1 h
   download-bound `convert`, 17 GB). OpenAlex is not on the Hub either (E5
   restages it).
3. Results: the remaining runs do not need the old records locally (the
   narrow scoping below never resumes against them), but the final uploads
   and the report do: `bench fetch --path-in-repo d1/<subtree> --results
   <dir>` per [hub-index](artifacts/hub-index.md) row, merged into
   `evaluation/results/<suite>/`.
4. Every GPU job: a fresh `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`
   (the cache is not keyed by `code_version`, H4), `HF_HOME=/scratch/hf`.
   Run from `evaluation/` as `/venvs/retrieve/bin/python -m bench.cli …`,
   **not** `uv run`, which masks bench's exit code. Pass `--timeout 48` to
   every `bench campaign` (the 6 h default kills `deep` groups, H3). A long
   job runs from one sequential driver script per GPU (one process group,
   no chained waiters, and no log line containing the text a watcher
   greps for).

### Multi-GPU execution

On a pod with several GPUs, each job owns one GPU
(`CUDA_VISIBLE_DEVICES=<i>`), its own inductor cache dir and its own
`--out` tree or `--output` file; `aggregate` reads every
`<suite>/*.jsonl`, so per-stream files in one tree merge without a step.
Pin each job's CPU threads to its GPU's NUMA-local cores (`taskset -c`,
the affinity column of `nvidia-smi topo -m`): batch-1 latency is
launch-bound, so a neighbour's host load moves it. Record in the step's
validation row that neighbour GPUs were busy. Split streams by `--algo`,
then by `--filter-kind` / `--sweep` / `--seed`; keep `silvertorch`'s
`triton` and `official` backends in **one** stream, in that order: they
share a `_parity/<dataset, dim, algo>` spill and the second backend's
`jaccard_vs_first` compares against the first's. `bench campaign` has no
`--algo`, so a split stream is a `bench run`. M1 decides whether
multi-GPU numbers are comparable at all.

### Rerun policy

**No record is re-stamped.** `code_version` (the tree hash of
`retrieve/src/retrieve`) is in the resume key, so a library fix makes
`bench campaign --resume` rerun everything. Instead, a fix's own gates
decide which arms it changes, and only those arms are rerun with narrow
`bench run` scoping (`--algo` / `--backend` / `--filter-kind` / `--sweep`
/ `--seed`); every other record keeps its `code_version`
([policy](validation.md#campaign-roadmap-d1-in-progress-not-yet-validated)).
Today's `code_version` is `c0e42d1`; D1's existing records are at
`72e5a90`, so **never run `bench campaign --resume` on a leg that already
has records** (arxiv `filter`/`deep`, pubmed `filter`). Campaign is safe
only on a leg with no records yet.

## Phase H: harness prerequisites

- [ ] **H2: the record key carries the input identity; the golden gate
  pins its checkpoint.** *Problem*: `bench/records.py` `KEY_FIELDS`
  (dataset, dim, suite, filter_kind, sweep, algo, backend, params, seed;
  + `code_version` in the resume key) names no checkpoint or embedding
  source, so a goodreads cell on the E1c encoder has the same resume key
  as its gSASRec record in `d1/goodreads`: `--resume` skips it as `ok`, and
  `records.latest` / `aggregate` keep one of the two. *Build*: (1) an
  `inputs` identity in the key block: the ckpt-id for sequential datasets
  (the checkpoint's directory name, e.g. `sasrec-ssm-logq-d128`), the
  resolved `content_dir` for text ones; whether a name is enough or a
  digest is needed is written in [evaluation](system/evaluation.md#resume).
  (2) **Existing records must keep their resume key**: a record without
  the field derives it (sequential: the gSASRec ckpt-id its config named,
  goodreads `gsasrec-d{dim}-drop0.5-id`, kuairand `gsasrec-d128-shared`;
  text: the `content_dir` today's config resolves), and a new arxiv/pubmed
  job computes the same key, or D1's remaining arxiv/pubmed scoping reruns
  hundreds of hours. Bump `SCHEMA_VERSION` only if the record layout
  changes, and say why either way. (3) `results.parquet` and `bench report`
  carry the identity as a column; a report groups or refuses mixed
  identities within one dataset × dim. (4) The golden goodreads cells run
  on `gsasrec-d128-drop0.5-id`: the smallest honest pin (a `--checkpoint`
  override on `bench run` recorded in the record, or a golden dataset
  file). *Gates*: the harness suite green on CPU; a test that two jobs
  differing only in checkpoint get different resume keys, and that an old
  arxiv record (a fixture from a real D1 key block) keeps today's key; the
  real Hub records (`d1/arxiv-deep-partial`, `d1/pubmed` fetched to
  scratch) give the same set of done cells before and after (script under the
  H2 artifacts directory); the golden gate rerun on the pinned checkpoint
  gives today's numbers (validation *Golden baseline*: 8/9 identical, V1
  at 0.0; fetch `checkpoints/gsasrec-d128-drop0.5-id/*` from
  `pinkmeme/eval-goodreads-work-id`). **0.5 GPU-h** (the golden run), one
  stream; the rest is CPU. `opus` if it stays a key-field + override
  change, `fable` if it forces a redesign of `records` / `report`.
- [ ] **R1: stage the E1c checkpoints and prove the harness reads them.**
  Needs H2. `hf download pinkmeme/eval-goodreads-work-id --repo-type
  dataset --include "checkpoints/sasrec-ssm-logq-d128/*" --local-dir
  /data/goodreads-work-id`; the directory has `best_model.pt`, a
  `config.json` with `"loss": "sampled_softmax"` and `"normalize": true`,
  and an `item_id_map.json` sha256-equal to the data root's
  (`dd6b8005…107265`). `bench check --dataset goodreads` at d128; then one
  cell on a fresh inductor dir: `bench.cli run --dataset goodreads --dim
  128 --suite filter --algo linr_v1_filter_mask --backend triton
  --filter-kind clause --sweep c0_genre --seed 0 --skip-perf`. *Gates*: a
  fresh encode (a new cache file in the checkpoint dir) and a fresh oracle
  blob; the exact gate `recall_oracle@k_max ≥ 0.99` holds on normalized
  embeddings (a miss is a finding, reported, `EXACT_MIN_RECALL` not
  loosened); query L2 norms 1.0 ± 1e-3; `n_kept` recorded against the
  gSASRec cell's 9,859. Record the cell in validation (not yet validated)
  and on the Hub as `artifacts/r1`. **0.5 GPU-h**, one stream.
- [ ] **H3: `bench campaign`'s per-group timeout fits the work.** The
  default `--timeout 6.0` killed pubmed `filter` groups (cells take 20-46
  min at D = 768 × 10 M) and is far below an arxiv `deep` `silvertorch`
  group (~10 h). A default sized for the largest group, or one scaled by
  dataset and suite. *Gate*: the harness suite green, a test on the
  default; until it lands, every campaign passes `--timeout 48`. CPU.
- [ ] **H4: the harness keys its inductor cache by `code_version`.** The
  on-disk FX-graph / AOT-autograd caches do not invalidate when a
  `@triton_op` body changes, so a `graph`-mode run after a library edit can
  replay stale kernel code ([testing](system/testing.md#running)). `bench
  run` sets a `TORCHINDUCTOR_CACHE_DIR` that includes `code_version` unless
  one is given. *Gate*: a test that two code_versions resolve to two cache
  dirs; the harness suite green. CPU. Until it lands, the restore steps'
  fresh cache per job is the workaround.
- [ ] **H5: `bench report` states true provenance.** `bench/report.py`
  appends "These records predate the D1 campaign…" to every non-citable
  report, marks every non-citable caption `[PRE-CAMPAIGN RECORDS — NOT
  CITABLE]` (`_caption`) and watermarks figures "PRE-CAMPAIGN" (`_figure`),
  all false for D1's own records. Each names the actual reason a report is
  not citable (the record's `status`, `dirty`, `partial_reasons`, or a
  gate not green). *Gate*: `test_report.py` pins the text for each reason.
  CPU. Before D1-G and F5.
- [ ] **H6: `partial` is stamped per job, not per process.** `bench/run.py`
  builds `reasons0` once per process, so an eager-only pass marks every
  record `partial`, including `official`, whose `graph` entry is
  `not_capturable` anyway. *Gate*: a test that an `official` record from an
  eager-only run is not `partial` for `modes`, and a `triton` one is. CPU.
  Before D1-G.
- [ ] **H7: a held-out metric with no target is null, not 0.0.** Pubmed's
  `c3_journal_reverse` records carry held-out recall 0.0 where no held-out
  target passes the filter, which a report averages in as a miss. The
  harness writes null there, and `bench report` skips null held-out
  metrics. *Gate*: a test on a cell with no in-filter held-out target; the
  existing `d1/pubmed` records read as null through the same rule (no
  rerun). CPU. Before D1-G.
- [ ] **L6: widen the official T1 parity gate past powers of two.**
  `retrieve/tests/parity/test_official.py` `test_t1_int32_path_bitexact`
  still compares official against Triton only when `d` is a power of two
  (`d = 96` is reference-only), a guard from before L4's padding. Drop the
  guard and add a D1 width (192, 768) to the parametrization. *Gate*: the
  int32 path `torch.equal` official vs Triton at every width (bit-exact;
  a mismatch is reported, not tolerated). **≈ 0.2 GPU-h**, library suite.
- [ ] **M1: multi-GPU interference control.** Before any number from a
  multi-GPU pod is compared with a single-GPU one: one D1 cell (an arxiv
  `filter` cell at bs = 1, `silvertorch`/triton and `linr_v2`) timed alone
  on the pod, then with every other GPU loaded by a D1 job, interleaved,
  cores pinned. *Gate*: the bs = 1 and bs = 16 medians with neighbours
  loaded within each arm's own repeat noise of the alone runs; otherwise
  the timed steps run one GPU at a time and only `--skip-perf` work runs
  in parallel. **0.5 GPU-h** on a ≥ 2-GPU pod.

## Phase D: campaign and baselines (GPU)

- [ ] **D1: the full campaign.** Scope (user): `filter` on goodreads,
  arxiv, yfcc10m and pubmed, in scale order, then `deep` and `codesign`
  (S9) on arxiv, then goodreads. openalex's `filter` leg is E5's. Seeds {0, 1, 2} on the headline sweeps; `n_probe` ∈ {24, 32}. Done:
  the `filter` legs of arxiv (126/126), yfcc10m (7/7) and pubmed (44/56),
  and 803 of arxiv `deep`'s 870 cells
  ([validation](validation.md#campaign-roadmap-d1-in-progress-not-yet-validated)).
  Goodreads runs on the E1c encoder `sasrec-ssm-logq-d128`; `d1/goodreads`
  stays on the Hub as the gSASRec run. Everything below runs at
  `c0e42d1` (the rerun policy above). After each sub-step, `bench upload
  --verify` and a [hub-index](artifacts/hub-index.md) row. Closes paper
  gaps G3 (P99 / QPS), G4 (seeds), G7, G8 (cross-dataset deep sweeps).
  `R=$REPO_DIR/evaluation/results`; every command is `python -m bench.cli`
  from `evaluation/`, with `--out $R --config-dir config`.
  - [ ] **D1-A: arxiv `deep`, the remaining `linr_v3`** (14 of its 30
    jobs, 70 cells; the 16 jobs done at `72e5a90` are not rerun; bloom
    `c0_maincat` seed 1 has 3 of 5 pools at `72e5a90`, and the newer
    record supersedes them):
    ```
    run --dataset arxiv --dim 128 --suite deep --algo linr_v3 --backend triton --filter-kind bloom --sweep c0_maincat --seed 1 --seed 2 --resume
    run --dataset arxiv --dim 128 --suite deep --algo linr_v3 --backend triton --filter-kind bloom --sweep c2_year --sweep c3_nversions --sweep c0c2 --sweep all4 --resume
    ```
    Upload the complete leg as `d1/arxiv-deep` (803 records at `72e5a90` +
    70 at `c0e42d1`, the mixed version noted in the row), replacing
    `d1/arxiv-deep-partial`. If H2 merged first, confirm its
    old-records-keep-their-key gate before trusting `--resume`. **≈ 6
    GPU-h** (~5 min a cell); 2 streams (the two commands).
  - [ ] **D1-B: arxiv `codesign`** (10 jobs, no records yet): `campaign
    --suite codesign --dataset arxiv --resume --timeout 48`. Upload
    `d1/arxiv-codesign`. **≈ 3-5 GPU-h**, one stream (both `bloom_path`
    arms are `silvertorch`/official).
  - [ ] **D1-E: pubmed `filter`, the targeted rerun** (the three-fix pass
    changed only these arms):
    ```
    run --dataset pubmed --dim 768 --suite filter --algo silvertorch --backend triton --resume   # all 16 cells, retimed on the per-width tile
    run --dataset pubmed --dim 768 --suite filter --algo linr_v3 --backend triton --resume       # 8 cells, OOM at build before
    run --dataset pubmed --dim 768 --suite filter --algo linr_v2 --backend triton --filter-kind bloom --sweep c0c2 --seed 0 --resume   # the cell the 6 h timeout cut
    ```
    Re-upload `d1/pubmed`. `records.latest` keys on `code_version`, so
    the 13 old `silvertorch`/triton records and the 8 failed `linr_v3`
    ones stay beside the new ones: the D1 report reads only the `c0e42d1`
    records for those two arms (say so in validation). **≈ 12 GPU-h**
    (`silvertorch`/triton 16 × ~22 min, `linr_v3` 8 cells, `linr_v2` 1);
    3 streams (the three commands; ~6 h wall).
  - [ ] **D1-F: goodreads `filter` on the E1c encoder, the full leg** (105
    cells). Needs H2 and R1. `campaign --suite filter --dataset goodreads
    --resume --timeout 48`. Upload as `d1/goodreads-e1c`; label
    `d1/goodreads` as the gSASRec run in hub-index. **≈ 4 GPU-h**; up to 4
    streams by `--algo` (`bench run`, `silvertorch` holding both backends).
  - [ ] **D1-C: goodreads `deep`** (135 jobs, no records yet). Needs H2
    and R1. `campaign --suite deep --dataset goodreads --resume --timeout
    48`. Upload `d1/goodreads-deep`. **≈ 25-45 GPU-h** (arxiv `deep`: ~2
    min a cell over 870 cells, ~30 h; goodreads has more jobs on a 0.8 M
    catalog, 4× smaller); 2 streams by `--algo` (`silvertorch` both
    backends, `linr_v3`), each splittable by `--filter-kind`.
  - [ ] **D1-D: goodreads `codesign`** (6 jobs). Needs H2 and R1. `campaign
    --suite codesign --dataset goodreads --resume --timeout 48`. Upload
    `d1/goodreads-codesign`. **≈ 2 GPU-h**, one stream.
  - [ ] **D1-G: D1's report and gate.** Needs D1-A..F, H5, H6 and H7. `bench
    report` over every leg (goodreads rows from the E1c records only;
    pubmed's `silvertorch`/triton and `linr_v3` from `c0e42d1` only); the
    cross-scale filter comparison across algorithms (`recall_oracle`,
    latency, QPS, memory per algo per dataset at the headline operating
    point), goodreads redone on E1c. *Gate per leg*: `bench report` with no
    missing cells; `median_ms(bs=16) < 16 × median_ms(bs=1)`; ids identical
    across `eager` and `graph` (never run yet); a rerun byte-identical in
    quality (a subset of each leg's cells, rerun with `--force` into a
    scratch tree). `codesign`'s early smoke cell is not a finding; the
    real sweep decides. **≈ 5 GPU-h** for the reruns; streams by leg.
- [ ] **D3: bloom false-positive rate and memory against filter width**,
  for both blooms on real attributes (G6; paper claim S8). The
  head-to-head saw zero false positives at `m_bits 1024`, so the sweep
  goes below it. Needs D1. Code first (a width sweep in the harness or an
  artifact script), then **≈ 3 GPU-h**; streams by dataset.
- [ ] **D5: the generic-torch postfilter baseline.** The study's baseline
  is what a practitioner writes without a retrieval library
  ([decisions](decisions.md#harness)): a dense matmul over the whole item
  table on the GPU, `torch.topk`, then drop the ids that fail the filter,
  so a row returns fewer than K results whenever filtered-out items took
  top-K slots. No harness algorithm does this today
  (`linr_v1_filter_mask` masks *before* top-k and is exact).
  - [ ] **D5-code: the `postfilter` algo.** In `bench` (torch backend
    only; clause and bloom filter kinds): fetch top-αK, filter, keep the
    first K survivors, pad short rows with the `-1` / `-inf` sentinel so
    lost candidates count as a recall loss. α is a query param swept over
    {1, 2, 4, 8} (`params.postfilter` in the `filter` suite), so the
    report shows what a naive system pays in latency to recover recall;
    α = 1 is the headline baseline. *Gate*: on a small fixture its ids and
    scores equal a hand-computed torch reference (matmul → topk(αK) →
    filter → first K) at every α; `recall_oracle` reported against the
    exact oracle; the config gate expands it on every `filter` dataset;
    the harness suite green. Needs H2 (its goodreads cells need the input
    identity). CPU plus a smoke cell, `opus`.
  - [ ] **D5-run: the baseline on every `filter`-suite dataset** (goodreads
    on E1c, arxiv, yfcc10m, pubmed, openalex) with narrow `bench run
    --algo postfilter` scoping, so no existing record reruns; upload as
    `d5/<dataset>`. Needs D5-code and R1 (goodreads); openalex after E5's
    restage. **≈ 5-8 GPU-h** (4 α values; a dense matmul per batch);
    streams by dataset.

## Phase E: datasets

- [ ] **E5: openalex's `filter` leg; the report gains the unfiltered
  cells retired from the `quality` suite, for the text datasets only.**
  Ordered after D1; technically
  independent of it and of H2 (a text dataset). The 10 M OpenAlex catalog
  is not on the Hub: restage it on the pod (`openalex convert` streams 297
  GB over S3, then `prep --keep-items 10000000`, `encode_text`,
  `encode_queries`, `attrs`; [datasets](system/datasets.md#openalex)), or
  `reshard` from a larger staged copy. `bench check --dataset openalex`,
  then `campaign --suite filter --dataset openalex --resume --timeout 48`;
  upload `e5/openalex`. **≈ 25-35 GPU-h** (10 M × 768 like pubmed, 20-46
  min a cell) **+ ≈ 3 GPU-h** for a fresh encode; 4 streams by `--algo`.

## Phase F: the paper

- [ ] **F2: finish the official-vs-reimplementation section** with D1's
  numbers: confidence intervals, paired tests and a second seed, p95 and
  p99. The section ([paper](paper/official-vs-reimplementation.md))
  marks each line that waits on D1. Needs D1 and the user's goodreads
  decision. **F2-R** (if the user chooses rerun): the goodreads
  head-to-head on the E1c embeddings, b3 methodology
  ([h2h.py](artifacts/kernel-opt/h2h.py)), **≈ 2 GPU-h**, one stream (the
  arms interleave in one process).
- [ ] **F4: package the artifacts**: a tagged `torchretrieve` release, a
  Zenodo DOI including the pinned official sdist, Hub datasets, oracles
  and results, a one-command reproduction, an anonymised mirror for
  review. Needs D1 and the user's accounts.
- [ ] **F5: write the paper**, every table produced by `bench report`.
  Needs everything above.

## Estimates

| step | A100 GPU-h | parallel streams |
|---|---|---|
| H2 | 0.5 | 1 |
| R1 | 0.5 | 1 |
| M1 | 0.5 | 1 (on a ≥ 2-GPU pod) |
| L6 | ≈ 0.2 | 1 |
| D1-A arxiv `deep` `linr_v3` | ≈ 6 | 2 |
| D1-B arxiv `codesign` | ≈ 3-5 | 1 |
| D1-E pubmed targeted rerun | ≈ 12 | 3 |
| D1-F goodreads `filter` (E1c) | ≈ 4 | up to 4 |
| D1-C goodreads `deep` | ≈ 25-45 | 2, each splittable by filter kind |
| D1-D goodreads `codesign` | ≈ 2 | 1 |
| D1-G gate reruns | ≈ 5 | by leg |
| E5 openalex `filter` | ≈ 25-35 (+ 3 encode) | 4 |
| D5-run postfilter baseline | ≈ 5-8 | by dataset |
| D3 (after its code) | ≈ 3 | by dataset |
| F2-R (if chosen) | ≈ 2 | 1 |

H3-H7 are CPU; L6 is a short library-suite run. Runnable now, before H2: D1-A, D1-B, D1-E (and E5's
restage). The total is ≈ 95-130 GPU-h, of which D1 is ≈ 57-79.

## Dependencies

```
D1-A, D1-B, D1-E  (runnable now)
H2 ─> R1 ─┬─> D1-F, D1-C, D1-D ─┐
          └─> D5-code ─> D5-run (openalex cells also after E5)
H5, H6, H7 ─────────────────────┼─> D1-G ─┬─> D3 ──────────┐
D1-A, D1-B, D1-E ───────────────┘         ├─> F2 (+ F2-R)  ├─> F5
                                          ├─> F4           │
                                          └─> E5 ──────────┘
D5-run ───────────────────────────────────────────────────┘
H3, H4: before the next campaign (workarounds meanwhile: --timeout 48, a fresh cache per job)
M1: before any timed step runs on a multi-GPU pod
```
