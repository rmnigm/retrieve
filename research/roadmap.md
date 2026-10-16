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

The full rerun at `final-freeze` is done and its gate is green; SYNTH-MID is in; the export preparation, the docs pass
and RENAME (backend `cuda`, package `silvertorch`, ops `torch.ops.st`; Meta's code as published left the workspace, tag
`meta-published-parity-final`) are merged into `staging` ([validation](validation.md)). Open: 4 PAPER, 9 EXPORT, 8 F4.

### 4. PAPER: drafted by agents, decided by the user

ECIR 2027 replicability track, 12 pp LNCS + references, double-blind, due 2026-10-19
([venue](paper/venue.md)); LaTeX skeleton and bibliography in [`paper/tex/`](paper/tex/README.md),
[submission checklist](paper/submission-checklist.md). The style survey, the writing guide and the first full text
draft (text + figures, no formatting) exist on pod e only, on local branches that are not pushed (user, 2026-10-16).
Open: the user's answers to the draft's open questions, the cut to 12 pages, typesetting into `paper/tex/`.

### 8. F4: package the artifacts

Needs 9. The results public, the PubMed slice or its build recipe with
checksums, a tagged `torchretrieve` release (LiNR plus the adapted
SilverTorch; the adapted code keeps Meta's Apache-2.0 licence and
headers), a Zenodo DOI including the adapted source and Meta's pinned
sdist, a one-command way to run any eval and get its numbers (no paper-table rebuild in the repository; user,
2026-10-16), a clean public Hub repo if the user publishes one (no LiNR bloom, no superseded data), an anonymised mirror
(anonymous.4open.science) for the double-blind review. Needs the user's
accounts and the release decisions.

### 9. EXPORT: the released repository

A branch cut from `staging` holding the code, the human docs (`README.md`, `docs/`,
`retrieve/docs/`) and the agent docs (AGENTS.md / CLAUDE.md, pointing at nothing removed); without `articles/`,
`.chains/` and `research/` (user, 2026-10-16, [decisions](decisions.md)). *Gate*: suites green on the branch, link
checker 0 on it, no file under the dropped trees.

## Dependencies

```
4 PAPER (user decisions, cut, typeset)
9 EXPORT ─> 8 F4
Needs the user: release decisions before F4
```
