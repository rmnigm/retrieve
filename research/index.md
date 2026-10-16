---
title: index
created: 2026-09-26
updated: 2026-10-17
type: summary
tags: [process]
sources: [research/]
---

# retrieve research wiki

The agentic working wiki: the queue, the decisions, what was measured, the
process. Not for users of the release; they read [docs/](../docs/index.md).
The study: Meta's official SilverTorch code, adapted, against LiNR V1-V3
([decisions](decisions.md#goal-and-scheduling)); the paper is kept outside
this repository. Conventions, page format and tags for both trees:
[SCHEMA.md](SCHEMA.md).

## Start here

- [roadmap](roadmap.md): the open work queue (EXPORT, F4), what waits on
  the user, and how GPU work runs on a pod.
- [decisions](decisions.md): standing decisions and constraints, and why.
- [validation](validation.md): which gates pass now, and the measured
  results that stand (citable or not).

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

- [artifacts/](artifacts/): raw scripts behind measured numbers, one
  directory per plan; the raw outputs are on the Hub
  ([hub-index.md](artifacts/hub-index.md)).
- [docs/system/](../docs/index.md): how the code works today (the
  project's documentation; code comments cite it by section).
- `articles/`: the SilverTorch and LiNR papers, frozen. Gitignored;
  present locally, not after a fresh clone.
