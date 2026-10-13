---
title: roadmap
created: 2026-09-26
updated: 2026-10-13
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, evaluation/bench/, infra/runpod/, research/validation.md, research/artifacts/hub-index.md]
---

# Roadmap: the open queue

The single ordered work queue: what is left to run or implement, in
dependency order. What already holds is in [validation](validation.md),
the standing decisions in [decisions](decisions.md), work that is not
queued in the [backlog](backlog.md). An agent starting a session reads
[AGENTS.md](../AGENTS.md), then this page, then only the wiki pages its
step names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

**The study** ([decisions](decisions.md#goal-and-scheduling)): Meta's
official SilverTorch code, adapted by us, against LiNR V1-V3. The queue
below finishes the adapted SilverTorch backend, measures it for the paper
(LiNR's numbers until then come from the final pass, `campaign-final`,
library `e16512f5`), lets the user write the paper, removes our Triton
SilverTorch reimplementation from the code, and reruns everything at one
frozen tag (FULL-RERUN, before RENAME and the Triton deletion: controller
C1, 2026-10-13, for the user's review).

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator **removes** the step
from this page. Step names (META-FORK, D1-G, ...) are stable identifiers
used in records, commits and code comments; they are not renumbered.

**Estimates** are A100-SXM4-80GB GPU-hours (the reference GPU for every
citable number); a *stream* is a slice of a step that can run on its own
GPU at the same time as the others (see
[Multi-GPU execution](#multi-gpu-execution)).

## Needs the user

- **Contact the original authors** (re-plan decision 7): the LinkedIn LiNR
  team and Meta's SilverTorch team — filter-set details, the V1/V2 setup,
  the SilverTorch paper's FPR inconsistency (0.067 % vs 0.00173 %) — and
  file the published-code defects (OF-3, OF-4, OF-6, OF-9, OF-10 in
  [deviations](paper/reproduction-deviations.md)) upstream as issues. The
  ECIR call asks what contact happened; replies take weeks.
- **The release decisions** in [release §5](paper/release-and-licenses.md#5-decisions-for-the-user)
  (goodreads checkpoints, Re-LAION vectors, the PubMed slice, the public
  May result repos, `eval-results` when public, GitHub during review).
- **Citability of a narrowed run.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE; the user decides
  whether a deliberate, recorded narrowing reads differently.
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
--gpus N`); disks and environment in [pod storage](environment/storage.md).

### Restore on a fresh pod

1. Checkout `staging` (the image clones it into `/workspace/retrieve`);
   the venv is `/venvs/retrieve`. A worker that needs a different
   environment gets its own venv (`UV_PROJECT_ENVIRONMENT=/venvs/<name> uv
   sync --all-packages --all-groups --extra official`): a worker that
   re-syncs the shared venv to its worktree breaks every running job.
   Meta's code as published runs from an `-O3` build
   (`scripts/build_official_o3.sh`; `env.official_build` records the loaded
   extension).
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads: `arxiv-papers` (the license-fixed
   version: `item_attrs_narrow.pt` min = -1, `eval_split.parquet`
   `query_attrs_narrow` min = 0), `goodreads-work-id` with its checkpoints
   ([checkpoints](../docs/system/checkpoints.md#what-the-harness-reads)). PubMed is not on the Hub: rebuild the 10 M slice with
   `eval-data pubmed` ([datasets](../docs/system/datasets.md#pubmed); ~1 h
   download-bound `convert`, 17 GB), or copy it pod to pod; for the final run use the un-normalised (dot-product) build.
   Yambda-500M: `eval-data fetch` of its checkpoints plus the trainer `test.parquet`.
   LAION 30 M is not on the Hub either (pod-to-pod copies). The synth attrs
   are built on the pod (*eval-data synth-filter*, CPU, deterministic from
   its seed).
3. Results: the final aggregation needs the final pass's records:
   `bench fetch --path-in-repo campaign-final/<subtree> --results <dir>` per
   [hub-index](artifacts/hub-index.md) row.
4. Every GPU job: its own `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`
   (`bench run` keys a default by `code_version` when none is given,
   [evaluation](../docs/system/evaluation.md#inductor-cache)), `HF_HOME=/scratch/hf`.
   Run from `evaluation/` as `/venvs/retrieve/bin/python -m bench.cli …`,
   **not** `uv run`, which masks bench's exit code. `bench campaign`'s
   per-group timeout defaults to 48 h. A long
   job runs from one sequential driver script per GPU (one process group,
   no chained waiters, and no log line containing the text a watcher
   greps for). Every job's log carries the pod's `nvidia-smi -q -d CLOCK`
   and its `sm_mhz` histogram, uploaded with the results.

### Multi-GPU execution

On a pod with several GPUs, each job owns one GPU
(`CUDA_VISIBLE_DEVICES=<i>`), its own inductor cache dir and its own
`--out` tree or `--output` file; `aggregate` reads every
`<suite>/*.jsonl`, so per-stream files in one tree merge without a step.
Pin each job's CPU threads to its GPU's NUMA-local cores (`taskset -c`,
the affinity column of `nvidia-smi topo -m`): batch-1 latency is
launch-bound, so a neighbour's host load moves it. Record in the step's
validation row that neighbour GPUs were busy. Split streams by `--algo`,
then by `--filter-kind` / `--sweep` / `--seed`; keep the `silvertorch`
backends of one cell in **one** stream, in the suite's order: they share
a `_parity/<dataset, dim, algo>` spill and the second backend's
`jaccard_vs_first` compares against the first's. `bench campaign` has no
`--algo`, so a split stream is a `bench run`. M1's result (one timed job
per GPU of a two-GPU pod) is in
[validation](validation.md#campaign-v2-phase-v).

**Timings compare only within one box**
([decisions](decisions.md#harness)): every ratio is read inside one leg on
one pod.

### Stop rules

- **Noise gate.** Every timed comparison in an exhibit has a 95 % CI on its
  ratio that excludes 1.0, or it is reported as "no difference". No more
  than 5 repeats to force significance.
- **Budget gate.** If a step's GPU-h exceed 1.25× its estimate, report it
  and update the remaining estimates from the measured rate; the queue
  continues (user, 2026-10-08: the full grid runs).
- **Surprise gate.** A result that contradicts an earlier one (e.g. the
  adapted backend slower than Meta's published code on a cell its gate
  passed) stops the queue: rerun that one cell interleaved and report
  before anything else runs.

## The queue

The order is the user's (2026-10-11): numbers for the paper first, from
measurement legs on the adapted backend; the paper written on them; the
code cleaned; and **the full rerun of everything, pinned to one tag**. The
controller moved the rerun ahead of RENAME and TRITON-ST-DELETE (C1,
2026-10-13, for the user's review): the code freezes now, and the cleanup
follows the rerun.

### 1b. DOCS-SPLIT: the project pages carry no research residue

Needs 1 merged (`dev/meta-fork` edits the same pages); CPU, docs only,
beside 2; before 5 and 6, which edit the same pages. `docs/system/` is the
released project's documentation ([SCHEMA](SCHEMA.md#domain-and-scope)),
but `kernels.md`, `evaluation.md` and `datasets.md` still carry research
narrative: measured asides and their numbers (`kernels.md` "Deleted
backends", 30 links into `artifacts/`), campaign and study scoping
(`evaluation.md`'s "the campaign (roadmap D1)", which datasets and LiNR
variants the study runs), per-dataset ETL rates and sizes and "Dead
artifacts" (`datasets.md`). Move each fact to its research home
(validation, decisions, an artifact README) and leave a link; what the code
does stays. Headings cited by code (`§`) keep their names.

*Gate*: `python3 scripts/check_doc_links.py` 0 (bar `articles/`), every
moved number present once in `research/`, no `docs/system` page naming a
branch, pod or step's progress.

### 3. QUALITY: tests, review and fixes, no functional change

Running (user, 2026-10-11/12). A test developer (`dev/tests-pass`) and two
review pairs whose findings two fixers apply (`dev/quality-lib`,
`dev/quality-harness`); records must not move, bit-exact gates stay
bit-exact. Merge order: tests, then the fixers rebased on it. Deferred to
5 / 6 (they change the public surface or the records read): the
`PreparedFilter` shape, the two `bloom_path` knobs, the Triton SilverTorch
cut list, `tab_t3` / h2h, pre-schema-4 record readers, the duplicate
`upload-checkpoint` CLI, ETLs for out-of-study datasets.

### 4. PAPER: the user writes it

The NUMBERS legs, their results draft and the META-FORK accept (tag
`campaign-fork`) are done ([validation](validation.md)). The user writes the paper (ECIR 2027 reproducibility format, 12
pages LNCS) on RESULTS' numbers, against the
[ECIR criteria](paper/ecir-criteria.md); agents support with tables,
figures and the `research/paper/` sections on request (F2: the
[Meta as published vs adapted](paper/meta-published-vs-adapted.md) section
from T3 and the META-FORK gates, with CIs and paired tests). The final
tables are regenerated from FULL-RERUN's parquet (step 7).

### 5. RENAME: the adapted backend becomes SilverTorch's default

Needs 2 and FULL-RERUN's run (beside 4, on its own branch). `backend="official-fork"` becomes
the default `SilverTorch` backend under a name decided in this step; the
package, distribution and op-namespace names (`silvertorch_fork`,
`silvertorch-fork`, `stfork`) likewise. Records keep the names they were
written with; the report maps them. Every copy of the old names in
`AGENTS.md`, `README.md`, `docs/`, `research/` and `retrieve/docs/` follows (rule 4).
The user guide's SilverTorch examples (`retrieve/README.md`,
`getting-started.md`, `filtering-and-quantization.md`, `modules.md`), which
today run the default `backend="triton"`, then run the adapted backend.

*Gate*: library and harness suites green, Meta's suite on the renamed
package, the compile gate, outputs `torch.equal` before / after on the
gate grid's int32 cells (a rename changes no behaviour).

### 6. TRITON-ST-DELETE: delete the Triton SilverTorch reimplementation

Needs 5 (rule 5: the adapted backend is accepted and NUMBERS measured it, both done).
Tag `triton-silvertorch-final` on `staging` first, then delete
`SilverTorch`'s `backend="triton"` and `backend="torch"` paths: the
`codesigned_probe_score*` and `bloom_full_mask` kernels and their
`ops/reference` twins, SilverTorch-only host helpers, their parity /
correctness / compile tests (the official parity tests re-pointed at
adapted vs published), the harness arms and suite entries, the
`tune-kernels` entries, and their docs (`kernels.md`'s SilverTorch kernels
section, `architecture.md`, `filtering.md`, `testing.md`, `evaluation.md`,
`retrieve/docs/`). What SilverTorch's other backends share stays: phase 1
(`KMeans`, `quantize_int8*`, `csr_layout`, probe selection), the clause
kernels, the bloom hash (LiNR's `BloomFilter`), `two_level_topk` (LiNR V1).
LiNR's kernels stay.

*Gate*: library and harness suites green, `interfaces.DISPATCH` without
the deleted backends, `ruff`, links 0.

### 7. FULL-RERUN: everything, pinned to one frozen tag

Needs the freeze (tag `final-freeze` on `staging`, library + harness + suites; user's after-legs plan, controller C1
2026-10-13): it runs **before** 5 RENAME and 6 TRITON-ST-DELETE, so its records carry `backend="official-fork"` and the
report maps the name; every fix after the freeze is a new tag and a rerun of what it touches. Rerun everything at the tag:
LiNR, the adapted SilverTorch and the grid of NUMBERS,
into one `results.parquet` with its recipe ([full-rerun](artifacts/full-rerun/README.md): two stages, curves then picks
then timed, legs per pod, the untimed steps, the sizing and the trims) and a manifest that
names, per (dataset, suite, algo, backend), the accepted subtree. Then
D1-G's checks (eager-vs-graph ids, a byte-identical quality subset per
leg, batch scaling, `bench report --manifest`), T1 with C1-C7, T3 as Meta
as published vs adapted (its scorer column from `kernel_scopes`: H-SCOPE's
one GPU test on a real call is left). The paper's tables are regenerated
from this parquet; a number that moves against RESULTS' is reported, not
smoothed.

User rulings it carries (2026-10-12/13, [decisions](decisions.md)): YFCC is out of the study; PubMed is rerun on the
dot-product build (MedCPT as trained) with the query's own document excluded (self-match), and gains a synthetic-filter leg;
SilverTorch runs its bloom filter with no recheck in timed code, fp16 score path (Meta's default); LiNR runs the exact filter only.

Seeds and error bars (user, 2026-10-11) are part of this run: 0-2 where a build, a pool or a synthetic draw is random, one
for exact LiNR; LiNR V2 at bs 256 on PubMed stays out (controller C2). Before it starts: the profile and trims fit to the
credit (the sizing in [full-rerun](artifacts/full-rerun/README.md#sizing) exceeds it).

*Gate*: every planned cell `ok`, D1-G's four checks, 0 manifest gaps.
*Estimate*: [full-rerun § Sizing](artifacts/full-rerun/README.md#sizing): about 194 GPU-h full, 106 lean.

### 8. F4: package the artifacts

Needs 7. The results public, the PubMed slice or its build recipe with
checksums, a tagged `torchretrieve` release (LiNR plus the adapted
SilverTorch; the adapted code keeps Meta's Apache-2.0 licence and
headers), a Zenodo DOI including the adapted source and Meta's pinned
sdist, a one-command reproduction, an anonymised mirror
(anonymous.4open.science) for the double-blind review. Needs the user's
accounts and the release decisions.

## Dependencies

```
3 QUALITY ─> freeze ─> 7 FULL-RERUN ─> 4 PAPER tables (user) ─> 8 F4
                          └─> 5 RENAME ─> 6 TRITON-ST-DELETE (after the rerun)
1b DOCS-SPLIT (CPU, before 5)
Needs the user: release decisions before F4
```
