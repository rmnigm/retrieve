---
title: index
created: 2026-10-11
updated: 2026-10-11
type: summary
tags: [library, harness]
sources: [docs/, retrieve/docs/]
---

# retrieve documentation

> Last updated: 2026-10-11 | Total pages: 9

How the `retrieve` library (package `torchretrieve`) and the `evaluation`
harness work, kept in sync with the code. The research behind it (the
work queue, decisions, measured results, the paper drafts) is a separate
wiki in [research/](../research/index.md).

## Library user guide

Ships in the sdist, written for someone who installed the package:
[getting started](../retrieve/docs/getting-started.md),
[modules](../retrieve/docs/modules.md),
[filtering and quantization](../retrieve/docs/filtering-and-quantization.md),
[indexing and ops](../retrieve/docs/indexing-and-ops.md),
[benchmarking your own dataset](../retrieve/docs/adding-a-dataset.md).

## System

How the code works today. Code comments cite these pages by section, so
their headings are stable.

- [architecture](system/architecture.md): the `retrieve` package map,
  modules, filter composition, backend dispatch, module layout.
- [kernels](system/kernels.md): every Triton kernel, the reference ops and
  Meta's official ops: launch grids, tiles, autotune, numerics, and which
  of them ship.
- [filtering](system/filtering.md): what each paper specifies for filters,
  the mask and candidate-id paths, bloom hash keys.
- [testing](system/testing.md): the library suite's layout, fixtures,
  baselines and what each file asserts.
- [evaluation](system/evaluation.md): the `bench` harness: measurement
  protocol, config and suites, CLI, cell loop, records, report, upload, the merged tables.
- [datasets](system/datasets.md): `eval_datasets` ETL per dataset, the
  on-disk layout contract, Hub I/O, encoder training.
- [checkpoints](system/checkpoints.md): which encoder checkpoint each
  dataset reads, loading, embeddings, evaluation, Hub transfer.
- [storage](system/storage.md): where the harness keeps data, results and
  caches, and the environment variables that point it there.

## Contributing

The coding guidelines ([research/contracts](../research/contracts/coding-guidelines.md))
apply to every change; a change to behaviour updates the matching
`docs/system` page in the same commit, and
`python3 scripts/check_doc_links.py` stays at zero broken links.
