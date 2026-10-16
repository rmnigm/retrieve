---
title: index
created: 2026-09-26
updated: 2026-10-16
type: summary
tags: [process]
sources: [research/]
---

# retrieve research wiki

> Last updated: 2026-10-11 | Total pages: 10

The agentic working wiki: the queue, the decisions, what was measured, the
process, the paper drafts. Not for users of the release; they read
[docs/](../docs/index.md). The study: Meta's official SilverTorch code,
adapted, against LiNR V1-V3
([decisions](decisions.md#goal-and-scheduling)). Conventions, page format
and tags for both trees: [SCHEMA.md](SCHEMA.md). Structural changes:
[log.md](log.md).

## Start here

- [roadmap](roadmap.md): the open work queue, its gates, GPU-hour
  estimates and dependencies, and how GPU work runs on a pod.
- [backlog](backlog.md): known work that is not queued (unscheduled
  defects, post-paper work, background reading).
- [decisions](decisions.md): standing decisions and constraints, and why.
- [validation](validation.md): which gates pass now, and the measured
  results that stand (citable or not yet).

## Environment

- [pod storage](environment/storage.md): the GPU pods' disks, what lives
  where, the budget, the Hub quota.

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
  [Meta as published vs adapted](paper/meta-published-vs-adapted.md),
  [claims](paper/claims.md),
  [method differences](paper/method-differences.md),
  [outline](paper/outline.md),
  [results draft](paper/results.md),
  [abstract (submitted)](paper/abstract.md),
  [ECIR criteria](paper/ecir-criteria.md),
  [release and licenses](paper/release-and-licenses.md),
  [venue](paper/venue.md) (the ECIR 2027 call's rules),
  [submission checklist](paper/submission-checklist.md),
  [LaTeX sources](paper/tex/README.md);
  material: [data, intro, literature](paper/material/data-intro-literature.md),
  [method](paper/material/method.md),
  [results](paper/material/results.md),
  [coverage](paper/material/coverage.md): every citable source and claim to its section).
- [artifacts/](artifacts/): raw scripts behind measured numbers, one
  directory per plan; the raw outputs are on the Hub
  ([hub-index.md](artifacts/hub-index.md)).
- [docs/system/](../docs/index.md): how the code works today (the
  project's documentation; code comments cite it by section).
- [`articles/`](../articles/): the SilverTorch and LiNR papers, frozen.
  Gitignored (not shipped in git); present locally, not after a fresh clone.
