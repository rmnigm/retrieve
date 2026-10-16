---
title: roadmap
created: 2026-09-26
updated: 2026-10-17
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, evaluation/bench/, infra/runpod/, research/validation.md, research/artifacts/hub-index.md]
---

# Roadmap: the open queue

The single ordered work queue: what is left to do, in dependency order.
What already holds is in [validation](validation.md), the standing decisions
in [decisions](decisions.md). An agent starting a session reads
[AGENTS.md](../AGENTS.md), then this page, then only the wiki pages its step
names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

The campaign is over: the final run (FULL-RERUN at `final-freeze`) is done
and its gate is green, every result is on the Hub, and no GPU pod is
running. The paper is kept outside this repository and is not a step here.

**Rules.** Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator removes the step from
this page. Step names are stable identifiers used in commits.

## Needs the user

- **Contact the original authors**: the LinkedIn LiNR team and Meta's
  SilverTorch team (filter-set details, the V1 / V2 setup, the SilverTorch
  paper's FPR inconsistency, 0.067 % vs 0.00173 %), and file the defects
  found in Meta's published code upstream as issues (eager only, its ops
  fail CUDA-graph capture; 3-6 host syncs and ~50 launches a forward; the
  README's `pytest silvertorch/` fails at collection; the bloom index
  ignores `m_bits`; `setup.py` builds host code at gcc `-O0`).
- **Release decisions** (before F4):
  1. Goodreads checkpoints: ask the dataset authors whether trained weights
     may be shared; or keep them private and ship the recipe; or publish.
  2. Re-LAION derived vectors: a gated Hub repo, an ungated one, or recipe
     only.
  3. PubMed slice: recipe + checksums, or a vectors-only repo (17 GB, with
     the NLM statement).
  4. The public May result repos (`retrieval-filter-evals-2026-05-*`): keep,
     make private during review, or delete.
  5. `eval-results` when public: the existing repo with superseded subtrees
     deleted, or a fresh repo with only the cited subtrees.
  6. GitHub during review: public (the anonymised mirror suffices) or
     private until acceptance.
- **Merging `staging` into `main`** is on hold until the user decides.
- **HF: squash `pinkmeme/eval-goodreads-work-id`** to reclaim 2.3 GB
  (`HfApi().super_squash_history(repo_id=..., repo_type="dataset")`).
- **`.git` history size.** Results files left `HEAD`, but their bytes stay
  in history; shrinking `.git` needs a history rewrite (`git filter-repo`),
  which forces every clone and worktree to re-sync.
- **Dropped datasets' code.** The ETLs, configs and tests of datasets out
  of the study (KuaiRand, OpenAlex, yambda-5b, YFCC-10M's leftovers) stay in
  the harness until the user decides whether the release keeps them.

## The queue

### EXPORT: the released repository

A branch cut from `staging` holding the code, the human docs (`README.md`,
`docs/`, `retrieve/docs/`) and the agent docs (AGENTS.md / CLAUDE.md,
pointing at nothing removed); without `articles/`, `.chains/` and
`research/` ([decisions](decisions.md#goal-and-scheduling)). Links from
`docs/`, `README.md`, `evaluation/golden/README.md`, `silvertorch/CHANGES.md`
and code comments into `research/` are rewritten or dropped on that branch.
*Gate*: the library suite (GPU) and the harness suite green on the branch,
link checker 0 on it, no file under the dropped trees.

### F4: package the artifacts

Needs EXPORT and the release decisions. The results public, the PubMed
slice or its build recipe with checksums, a tagged `torchretrieve` release
on PyPI (LiNR plus the adapted SilverTorch; the adapted code keeps Meta's
Apache-2.0 licence and headers), a Zenodo DOI including the adapted source
and Meta's pinned sdist, a one-command way to run any eval and get its
numbers, a clean public Hub repo if the user publishes one (no LiNR bloom,
no superseded data), an anonymised mirror (anonymous.4open.science) for
the double-blind review. Needs the user's accounts.

## Dependencies

```
EXPORT ─> F4
release decisions (user) ─> F4
```

## Running GPU work on a pod

GPU work runs on RunPod pods launched with
[`infra/runpod/pod.sh`](../infra/runpod/pod.sh) (`pod.sh up --gpu a100
--gpus N`); disks and environment in [pod storage](environment/storage.md).

### Restore on a fresh pod

1. Checkout `staging` (the image clones it into `/workspace/retrieve`); the
   venv is `/venvs/retrieve`. A worker that needs a different environment
   gets its own venv (`UV_PROJECT_ENVIRONMENT=/venvs/<name> uv sync
   --all-packages --all-groups`): re-syncing the shared venv to a worktree
   breaks every running job.
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads ([checkpoints](../docs/system/checkpoints.md#what-the-harness-reads)).
   PubMed is on the Hub as `pinkmeme/eval-pubmed` (the dot-product build) or
   rebuilt with `eval-data pubmed` ([datasets](../docs/system/datasets.md#pubmed));
   LAION is not on the Hub (rebuild it with its ETL); synthetic attrs are
   built on the pod (`eval-data synth-filter`, deterministic from its seed).
3. Results: `bench fetch --path-in-repo fr/<leg> --results <dir>` per
   [hub-index](artifacts/hub-index.md) row.
4. Every GPU job: its own `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`
   ([evaluation](../docs/system/evaluation.md#inductor-cache)),
   `HF_HOME=/scratch/hf`. Run from `evaluation/` as
   `/venvs/retrieve/bin/python -m bench.cli …`, **not** `uv run`, which
   masks bench's exit code. A long job runs from one sequential driver
   script per GPU, and its log carries the pod's `nvidia-smi -q -d CLOCK`
   and its `sm_mhz` histogram.

### Multi-GPU execution

On a pod with several GPUs, each job owns one GPU
(`CUDA_VISIBLE_DEVICES=<i>`), its own inductor cache dir and its own
`--out` tree; `aggregate` reads every `<suite>/*.jsonl`, so per-stream files
in one tree merge without a step. Pin each job's CPU threads to its GPU's
NUMA-local cores (`taskset -c`, the affinity column of `nvidia-smi topo
-m`): batch-1 latency is launch-bound, so a neighbour's host load moves it.
Split streams by `--algo`, then by `--filter-kind` / `--sweep` / `--seed`;
keep the `silvertorch` cells of one dataset and dim in one stream (they
share a `_parity/<dataset, dim, algo>` spill). One timed job per GPU showed
no interference above about 0.7 %
([validation](validation.md#inputs-to-the-final-run-method-not-results)).
Timings compare only within one box ([decisions](decisions.md#harness)).

### Stop rules

- **Noise gate.** Every timed comparison has a 95 % CI on its ratio that
  excludes 1.0, or it is reported as "no difference". No more than 5
  repeats to force significance.
- **Surprise gate.** A result that contradicts an earlier one stops the
  queue: rerun that one cell interleaved and report before anything else
  runs.
