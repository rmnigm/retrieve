---
title: index
created: 2026-09-26
updated: 2026-09-26
type: summary
tags: [process]
sources: [docs/]
---

# retrieve documentation wiki

> Last updated: 2026-09-26 | Total pages: 13

What is true now about the code and the project. Conventions, page
format and tags: [SCHEMA.md](SCHEMA.md). Structural changes:
[log.md](log.md).

## Start here

- [roadmap](roadmap.md): the open work queue, its gates and dependencies.
- [decisions](decisions.md): standing decisions and constraints, and why.
- [validation](validation.md): which gates pass now, and the measured
  results that stand (citable or not yet).

## System

How the code works today. Code comments cite these pages by section.

- [architecture](system/architecture.md): the `retrieve` package map,
  modules, filter composition, backend dispatch, module layout.
- [kernels](system/kernels.md): every Triton kernel, the reference ops and
  Meta's official ops: launch grids, tiles, autotune, numerics.
- [filtering](system/filtering.md): what each paper specifies for filters,
  the mask and candidate-id paths, bloom hash keys.
- [testing](system/testing.md): the library suite's layout, fixtures,
  baselines and what each file asserts.
- [evaluation](system/evaluation.md): the `bench` harness: measurement
  protocol, config and suites, CLI, cell loop, records, report, upload.
- [datasets](system/datasets.md): `eval_datasets` ETL per dataset, the
  on-disk layout contract, Hub I/O, gSASRec training.
- [checkpoints](system/checkpoints.md): the gSASRec checkpoints: loading,
  embeddings, evaluation, Hub transfer.
- [storage](system/storage.md): the GPU pods' disks, what lives where, the
  budget (contested: the image's data root against the placement rule).

## Contracts

- [agent-orchestration](contracts/agent-orchestration.md): one
  orchestrator, at most three constrained workers, model routing, where
  work lands.
- [coding-guidelines](contracts/coding-guidelines.md): priorities, thin
  code, no compatibility with ourselves, tests as gates, what is not the
  goal.

## Related, not wiki pages

- [paper/](paper/): the reproducibility paper's sections
  ([deviations](paper/reproduction-deviations.md),
  [provenance](paper/provenance-and-disclosure.md),
  [official vs reimplementation](paper/official-vs-reimplementation.md)).
- [artifacts/](artifacts/): raw scripts behind measured numbers, one
  directory per plan; the raw outputs are on the Hub
  ([hub-index.md](artifacts/hub-index.md)).
- [`retrieve/docs/`](../retrieve/docs/): the library user guide shipped in
  the sdist.
- [`articles/`](../articles/): the SilverTorch and LiNR papers, frozen.
  Gitignored (not shipped in git); present locally, not after a fresh clone.
