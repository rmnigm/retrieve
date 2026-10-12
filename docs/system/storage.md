---
title: storage
created: 2026-09-26
updated: 2026-10-11
type: concept
tags: [environment, harness]
sources: [evaluation/eval_datasets/hub.py, evaluation/bench/]
---

# Where the harness keeps data

What a machine running the harness needs on disk, and the variables that
point the harness at it. The GPU pods this project's numbers were measured
on, their disks and budget are in the research wiki
([pod storage](../../research/environment/storage.md)).

## Data root

Staged datasets, encoder checkpoints and raw ETL input live under
`$RETRIEVE_DATA_ROOT`, resolved by `evaluation/eval_datasets/hub.py:data_root()`.
Its layout, the default and the `evaluation/data` symlink are in
[datasets](datasets.md#eval_datasets--what-is-on-disk); the checkpoints'
in [checkpoints](checkpoints.md). Everything there is regenerable: a dataset
is re-fetched with `eval-data fetch` or rebuilt by its ETL, so it can sit on a
large ephemeral disk.

## Results

`bench` writes one JSONL record per cell into the gitignored results tree
under `evaluation/` (or `bench campaign --out <dir>`); resume reads it back
([evaluation](evaluation.md#resume)). A finished tree goes to the Hub with
`bench upload` and may be deleted locally once `bench upload --verify` passed
([results storage](evaluation.md#results-storage)).

## Environment

```bash
export RETRIEVE_DATA_ROOT=/data                       # staged datasets and checkpoints
export TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>   # private per job; the harness appends the code_version subdir
export HF_HOME=/scratch/hf                            # Hub download caches
export UV_PROJECT_ENVIRONMENT=/venvs/retrieve         # optional: one shared venv for every worktree (~7.6 GB each)
```

`TORCHINDUCTOR_CACHE_DIR` must be **private per job** (one job per GPU):
inductor's on-disk FX cache does not invalidate when a `@triton_op` host
wrapper's Python source changes, so a shared `/tmp/torchinductor_root`
silently serves stale kernels
([reproduction-deviations](../../research/paper/reproduction-deviations.md) D-7).
For the same reason it must change with the library tree: a job directory
reused after a library edit replays the old body in every `graph`-mode cell
([validation](../../research/validation.md#library-gates), V1-FUSE), so it is
keyed by `code_version`: `bench` appends the library tree's `code_version` to
the directory it is given, and the library pytest suite uses a fresh
directory per run. A script that compiles outside `bench` and pytest keys its
own directory by `code_version`.
