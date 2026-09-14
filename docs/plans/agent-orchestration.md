# Agent orchestration — one coordinator, constrained workers

> **Status:** effective 2026-09-15 on `development` at `a2182d8` (user
> decision; no code changed). A contract, not a work list: it says who runs
> what, on which model, how many at once, and where the output has to land.
> It does not order any step — that is the roadmap's job — and it does not
> relax any hard rule in [`../../CLAUDE.md`](../../CLAUDE.md).
>
> Question this document answers: *when a step of the roadmap is dispatched to
> an agent rather than done by hand, what is the session shape, which model
> takes it, how many run in parallel, and what has to exist when it is done?*
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1. This document is
> the authority on *dispatch*, not on order: it never changes which step comes
> next, only who executes it.

## 1. The shape

**One orchestrator session, controlled by the user, dispatching constrained
workers.** Two levels, and the tree is this wide and no wider:

```
user
 └── orchestrator  (one session, the whole time)
      ├── worker 1        ┐
      ├── worker 2        ├─ at most three running at once (§4)
      └── worker 3        ┘
           └── sonnet nested subagent — web deep research only (§4)
```

The orchestrator holds the plan, decides what is dispatched and in which
order, reviews what comes back, merges it, and keeps the roadmap true. It is
the session the user talks to. The workers do the work: code implementation,
testing, evaluation runs, documentation updates. A worker is **constrained** —
it is handed one step with its gates and its plan section, and it does not
widen its own scope, start a step the roadmap does not list, or dispatch
peers.

## 2. Decisions

- **D1 — One orchestrator, for the whole session.** Not a pool, not a
  hand-off chain. A second coordinator would mean two sessions both believing
  they own the roadmap's checkboxes and the merge into `development`.
- **D2 — Workers are constrained, not autonomous.** A worker receives: the
  step, its plan section, its gates, and the branch it works on. It returns a
  validation record and a working tree. Scope changes go back to the
  orchestrator, which is the only party that may re-read the roadmap and
  decide what happens next.
- **D3 — Model by the *shape* of the work, not by its importance** (§3). The
  split is between work whose difficulty is design-and-write — large kernel or
  architecture rewrites — and work whose difficulty is care and follow-through.
- **D4 — A `fable` worker gets a big chunk, not a slice.** The routing in §3
  is worth its cost only when the worker is handed enough of the problem to
  hold the design in one head; splitting a library rewrite across three
  sessions reintroduces exactly the coordination the split was meant to avoid.
  This is also why **L** WP-1 is one worker and not three
  ([library-api-refactor.md](library-api-refactor.md) §11, risk 1: nothing else
  may edit `retrieve/` while it is open).
- **D5 — Exactly two levels, with one exception.** The only permitted nested
  subagent is a `sonnet` agent doing web deep research (§4). Everything else
  is orchestrator → worker. Nesting beyond that produces work nobody reviewed
  and records nobody wrote.
- **D6 — Three workers at once, maximum** — and that is a ceiling, not a
  target; §4 lists the three things that lower it.
- **D7 — Every worker's result is documented in a plan.** A run whose output
  is only in a transcript did not happen: the transcript is not in the
  repository, cannot be diffed, and cannot be analysed later. The validation
  record is the deliverable (§5).
- **D8 — All work ends in `development` on `origin`.** Whatever branch or
  worktree it was done in, work is not finished until it is merged into
  `development` and that branch is pushed to the GitHub remote (§6).

## 3. Model routing

| work | model | why |
|---|---|---|
| Core **library** rewrites with heavy kernel or coding work — a Triton kernel written or retuned, the `modules` / `ops` / `indexing` move, a backend adapter | **fable** | one large design-and-write chunk held in one head (D4) |
| Core **evaluation-harness** rewrites carrying architecture design as well as coding — the package split, the cell loop, the record schema | **fable** | same: the difficulty is the design, and it is not divisible |
| Routine cleanups, refactors that follow an already-written plan | **opus** | the plan holds the design; the work is care |
| Monitoring a run; babysitting a campaign | **opus** | |
| Debugging a failure, chasing a gate that went red | **opus** | |
| Documentation updates, `docs/system` sync, link fixes | **opus** | |
| One-time experiments, probes, prototypes | **opus** | throwaway by construction; its record is what survives |
| **Web deep research** | **sonnet**, and only as a nested subagent under a worker (§4) | |

Roadmap steps, routed: **L1 / L2** (the library layout move and the
composites) and **C5** (the harness package split) are `fable`. **C4**'s gate
rerun, **B3**'s head-to-head, **D1**'s campaign, **D4**'s `report.py`, the
Phase E loaders, and **F1** / **F3** are `opus`. When a step does not obviously
fall on one side, the orchestrator decides and says so in the dispatch.

## 4. Concurrency and nesting

**At most three workers running at once.** Three further constraints bind
below that ceiling, and all of them win against it:

1. **The GPU is one machine and is serialized** (CLAUDE.md rule 1). At most
   one worker may hold GPU work at any time, whatever the worker count is.
2. **Plan-level exclusivity.** Where a plan claims a subtree — **L** WP-1 over
   `retrieve/` — no second worker touches that subtree until it merges.
3. **Disjoint trees.** Two concurrent workers must not edit the same files.
   Parallelism comes from working in different packages, not from splitting a
   package.

**Nesting.** A worker may spawn exactly one kind of child: a **`sonnet`
subagent for web deep research** — surveying external practice, checking a
venue's CfP, confirming a dataset licence, reading upstream source that is not
in this tree. That is the whole exception. No other nested agents: the tree is
orchestrator → worker, and a worker that wants more hands asks the
orchestrator for them.

## 5. What a worker is handed, and what it returns

Handed, by the orchestrator, at dispatch:

- the roadmap step and its plan section (and nothing else to read — CLAUDE.md's
  opening rule: this file, the roadmap, then only the named section);
- the gates the step must pass, quoted, including which are bit-exact
  (CLAUDE.md rule 3);
- the branch to work on and the worktree, if any;
- the model, per §3, and the concurrency constraint it is running under;
- explicitly, what is *out* of scope for it.

Returned, before the step is called done:

- a **validation record appended to the plan the worker executed** — the model
  is [archive/cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360):
  environment, what ran, what passed, what was skipped and why, what is still
  unverified;
- raw scripts and outputs under `docs/plans/<plan>-artifacts/` (CLAUDE.md rule
  6), not in the packages;
- the matching `docs/system` file updated in the same commit, if behaviour
  changed (rule 4), and `python3 scripts/check_doc_links.py` at zero;
- a plain statement of what passed, what was skipped, and what is unverified —
  never a number presented as citable before its gate is green (rule 2).

The orchestrator flips the roadmap checkbox, with date and commit. A worker
does not flip its own.

## 6. Where the work lands

```
dev/<step>  (worker's branch, optionally in a worktree under /workspace/wt/)
    │  gates green, record written, review fixes in
    ▼
development  ──push──►  origin/development
```

**All work ends up on `development` at `origin`**, whatever branch or worktree
produced it. A step whose code sits only in a local branch, a worktree, or a
detached checkout is not finished — the box is rented and the repository is the
only durable artifact. The orchestrator owns the merge and the push; per
CLAUDE.md rule 6 it commits when the user asks, and pushing is the same
outward-facing action under the same permission.

`main` is untouched by this document: A4 (the merge of `development` into
`main`) remains on hold and is the user's call
([00-roadmap.md](00-roadmap.md) §1 Phase A).

## 7. Validation record

*(nothing is recorded here: this document is a contract. Each worker's record
goes in the plan that worker executed, per D7 and §5.)*
