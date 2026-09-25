---
title: agent-orchestration
created: 2026-09-26
updated: 2026-09-26
type: summary
tags: [process]
sources: [CLAUDE.md]
---

# Agent orchestration: one coordinator, constrained workers

A contract set by the user. It says who runs a
step, on which model, how many run at once, and where the output has to
land. It orders nothing: the order is the [roadmap](../roadmap.md). It
relaxes no hard rule in [AGENTS.md](../../AGENTS.md).

## 1. The shape

One orchestrator session, controlled by the user, dispatches constrained
workers. Two levels, no wider:

```
user
 └── orchestrator  (one session, the whole time)
      ├── worker 1        ┐
      ├── worker 2        ├─ at most three running at once (§4)
      └── worker 3        ┘
           └── sonnet nested subagent: web deep research only (§4)
```

The orchestrator holds the queue, decides what is dispatched and in which
order, reviews what comes back, merges it, and keeps the roadmap true. It
is the session the user talks to. Workers do the work: code, tests,
evaluation runs, documentation. A worker is **constrained**: it is handed
one step with its gates, and it does not widen its own scope, start a step
the roadmap does not list, or dispatch peers.

## 2. Decisions

- **D1: one orchestrator for the whole session.** Not a pool, not a
  hand-off chain. A second coordinator would mean two sessions both
  believing they own the roadmap and the merge into `staging`.
- **D2: workers are constrained, not autonomous.** A worker receives the
  step, its gates and its branch. It returns the branch with
  [validation.md](../validation.md) and the `docs/system` page updated. Scope changes go back to the orchestrator, the only party
  that re-reads the roadmap and decides what happens next.
- **D3: route by the shape of the work, not its importance** (§3). The
  split is between work whose difficulty is design-and-write (large kernel
  or architecture rewrites) and work whose difficulty is care and
  follow-through.
- **D4: a `fable` worker gets a big chunk, not a slice.** Routing is worth
  its cost only when one worker holds the whole design. Splitting a library
  rewrite across three sessions brings back the coordination the split was
  meant to avoid. While such a worker owns a subtree, nothing else edits
  it (§4).
- **D5: exactly two levels, with one exception**: a `sonnet` subagent doing
  web deep research (§4). Deeper nesting produces work nobody reviewed and
  records nobody wrote.
- **D6: three workers at once, at most.** A ceiling, not a target; §4 lists
  what lowers it.
- **D7: every worker's result lands in the repository.** What it
  validated goes into [validation.md](../validation.md) as current state,
  and what it changed goes into the `docs/system` page. A run whose output
  is only in a transcript did not happen: a transcript cannot be diffed or
  analysed later.
- **D8: all work ends on `staging` at `origin`.** Whatever branch or
  worktree produced it, work is not finished until it is merged into
  `staging` and pushed (§6).

## 3. Model routing

| work | model | why |
|---|---|---|
| Core **library** rewrites with heavy kernel or coding work: a Triton kernel written or retuned, a module move, a backend adapter | **fable** | one large design-and-write chunk held in one head (D4) |
| Core **harness** rewrites carrying architecture design as well as code: a package split, the cell loop, the record schema | **fable** | the difficulty is the design, and it is not divisible |
| Routine cleanups, refactors that follow a written plan | **opus** | the plan holds the design; the work is care |
| Monitoring a run, babysitting a campaign | **opus** | |
| Debugging a failure, chasing a gate that went red | **opus** | |
| Documentation, `docs/system` sync, link fixes | **opus** | |
| One-time experiments, probes, prototypes | **opus** | throwaway by construction; its record is what survives |
| **Web deep research** | **sonnet**, only as a nested subagent under a worker (§4) | |

When a step does not fall clearly on one side, the orchestrator decides
and says so in the dispatch.

## 4. Concurrency and nesting

At most three workers at once. Three constraints bind below that ceiling,
and each wins against it:

1. **The GPU is serialized** (CLAUDE.md rule 1). At most one worker holds
   GPU work at any time.
2. **Subtree exclusivity.** When a step claims a subtree (a library rewrite
   over `retrieve/`), no second worker touches that subtree until it merges.
3. **Disjoint trees.** Two concurrent workers never edit the same files.
   Parallelism comes from working in different packages, not from
   splitting one.

A worker may spawn exactly one kind of child: a `sonnet` subagent for web
deep research (external practice, a venue's call for papers, a dataset
licence, upstream source that is not in this tree). A worker that wants
more hands asks the orchestrator.

## 5. What a worker is handed, and what it returns

Handed at dispatch:

- the roadmap step and the page or record it builds on, and nothing else
  to read;
- the gates the step must pass, quoted, including which are bit-exact
  (CLAUDE.md rule 3);
- the branch, and the worktree if any;
- the model (§3) and the concurrency constraint it runs under;
- explicitly, what is out of scope.

Returned before the step is called done:

- [validation.md](../validation.md) updated to the new state of every gate
  the step touched: what passes now, on which environment, what was
  skipped and why, what is still unverified;
- raw scripts and outputs under `docs/artifacts/<plan>/`, not in the
  packages (CLAUDE.md rule 6);
- the matching `docs/system` page updated in the same commit if behaviour
  changed (rule 4), and `python3 scripts/check_doc_links.py` at zero;
- a plain statement of what passed, what was skipped and what is
  unverified, never a number presented as citable before its gate is
  green (rule 2).

The orchestrator updates the roadmap after the merge (the finished step
leaves it). A worker does not edit the roadmap.

## 6. Where the work lands

```
dev/<step>  (worker's branch, optionally in a git worktree)
    │  gates green, record written, review fixes in
    ▼
staging  ──push──►  origin/staging
```

All work ends on `staging` at `origin`. A step whose code sits only in a
local branch, a worktree or a detached checkout is not finished: the GPU
box is rented, and the repository is the only durable artifact. The
orchestrator owns the merge and the push; it commits when the user asks
(CLAUDE.md rule 6), and pushing is the same outward-facing action under
the same permission. Merging `staging` into `main` is the user's call.
