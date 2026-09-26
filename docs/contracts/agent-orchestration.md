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

**In force on the pods (user decision).** The following override this table and §4 for
every pod orchestrator:
- Every worker runs on **opus**, core rewrites included. No `fable`.
- At most **two** herdr agents are alive at once, and a review agent counts as one of them.
- Workers get **no subagents**, not even the `sonnet` research child. Start each one with
  `--disallowedTools Agent Workflow` and say so in the brief.

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

A dispatch brief has these headings, in this order, every time:

- **Step** — the roadmap step id and its one-line description.
- **Read only** — the page or record it builds on, and nothing else;
  a worker does not go read the rest of the wiki on its own initiative.
- **Gates** — quoted verbatim from the roadmap, with which ones are
  bit-exact marked explicitly (CLAUDE.md rule 3).
- **Branch/worktree** — the branch name and worktree path, if any.
- **Model + concurrency** — which model (§3) and which constraint from
  §4 it runs under.
- **Out of scope** — named explicitly, not left implicit.
- **Verify commands** — the exact commands the worker runs to check its
  own gates before handing back.
- **Return** — what comes back and how (§ below).

Two lines stand outside the headings and apply to every dispatch: **the
code wins over any doc, note or memory** — a worker that finds the wiki
and the code disagreeing trusts the code and files the doc as a bug
(AGENTS.md rule 4); and **state the mechanism of a bug before fixing
it** — a worker reports what is actually happening and why before it
changes anything, so the fix is traceable to a cause instead of a guess.

What counts as proof, by the kind of change:

| kind of change | proof |
|---|---|
| Refactor | the named gate is bit-exact before and after — run it, save the result, make the change, run it again. |
| Behaviour change | a test that pins the new behaviour, not just a passing run of the old suite. |
| Performance claim | a before/after pair, each with `sm_mhz`, the `unstable` flag and the shape measured, and the prediction written down *before* the measurement is taken — not fitted to it afterward. A slower result is still a result and is kept, not re-run until it looks better. A ratio between two backends is sampled interleaved (A, B, A, B, ...) in one process, so clock drift over the run hits both arms equally rather than favoring whichever ran first. |

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

### Where briefs, instructions and reports live

In the line of work's chain, `.chains/<chain>/`. The directory is local and git-excluded, and
the user's laptop session syncs it back. Scratch files are not used for this: they are invisible
to the laptop and die with the pod.

Each dispatched agent gets a **chain branch** named after it (`branch: w1-encoder`):
- **The first note** is the brief exactly as sent. Its `parent` is the `main` note that
  decided the dispatch.
- **Every later instruction** from the orchestrator is a new note on that branch.
- **The worker writes its report** as the branch's last note, by absolute path, with `parent`
  set to the previous note on the branch.
- **The outcome** (merged, rejected) goes on `main`.
- Notes are never edited; a correction is a new note.
- Filenames use the laptop's timestamp frame so the chain sorts.

Every brief also carries two lines:
- **Never print env or secrets files** (AGENTS.md rule 9).
- **Set up a worktree's venv with `uv sync`, not `rp-sync`** ([storage](../system/storage.md#environment)).

### The review gate

Before any code merges, a separate review agent (opus) runs the `deslop` skill and then the
`python-review` skill on the diff. It applies behaviour-preserving fixes, reports the rest as
findings, and runs `ruff check`, the harness suite and the link checker. There is no merge
without a CLEAN or CLEAN-AFTER-FIXES verdict.

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

**Exception, the sequential-encoder line:** its work merges into `dev/hstu` and is
pushed there, not to `staging`. Merging `dev/hstu` into `staging` is the user's call.
