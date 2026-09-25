---
title: coding-guidelines
created: 2026-09-26
updated: 2026-09-26
type: summary
tags: [process]
sources: [CLAUDE.md]
---

# Coding guidelines: what the code should look like, and what it is for

A contract set by the user. It orders and gates
nothing (the [roadmap](../roadmap.md) does both). It says what good work
looks like when a step is executed and what is deliberately not worth
doing. Companion: [agent-orchestration.md](agent-orchestration.md), who
executes a step. CLAUDE.md rules 1-8 remain the hard rules and win over
anything here.

## 1. Priorities, in order

1. **Correct code.** Bit-exact where a gate says bit-exact (CLAUDE.md
   rule 3). The only thing never traded away.
2. **Clean architecture.** Thin, direct, one obvious home per concern.
3. **Evals that run**, and profiling that explains them. The numbers are
   the product.

When two of these conflict, the lower number wins; when a guideline below
conflicts with one of them, the priority wins.

## 2. Decisions

- **D1: little defensive programming.** No guard for a condition the
  caller cannot produce, no `try`/`except` that turns a bug into a
  degraded result, no re-validating an invariant the constructor already
  established. Let it fail loudly at the real line. Validate at the
  **boundary** (a public constructor, a CLI argument, a file read), then
  trust the data. The harness already works this way: no exception
  swallowing, no retries ([evaluation](../system/evaluation.md)).
- **D2: no backward compatibility with ourselves.** Our own past shapes are
  not a constraint. Redo a part, delete what it replaced, and keep no
  alias, shim, deprecation path or second code path alive for a feature we
  invented. Compatibility is owed only to things we do not control: Meta's
  op schemas, `torch.ops` signatures, on-disk dataset layouts, published
  checkpoints, and the meaning of a recorded `code_version`. This does not
  weaken CLAUDE.md rule 5: a *kernel* is deleted only after its
  replacement's parity gate is green.
- **D3: thin and concise.** The smallest code that is correct and
  readable. Prefer a function to a class, a dict to a registry, a
  parameter to a subclass. No context bags, no stats dataclasses, no algo
  framework, no config object model, no timing strategy classes.
- **D4: no multi-level abstraction without a demonstrated need.** One level
  of indirection needs a reason; two need a second, written down. A base
  class with one subclass, a protocol with one implementation or a factory
  that builds one thing is a defect.
- **D5: no comment slop; document outside the code.** No multi-line
  commentary narrating the next lines, no restating a signature in prose,
  no banners, no commented-out code. Prefer no comment and a name that
  makes one unnecessary. A short note survives where the code is genuinely
  surprising (a bit order, an overflow bound, a sync-avoidance trick, a
  deviation from a paper) and it says *why*. The documentation is this
  wiki: `docs/system/` for what the code does, [decisions](../decisions.md)
  and the [roadmap](../roadmap.md) for why and what next, kept current as
  the code changes (CLAUDE.md rule 4).
- **D6: tests are gates, not a deliverable.** Write the test a gate names
  and the test that pins a bug you just fixed. No scaffolding, fixtures,
  helpers or matrices beyond that (CLAUDE.md rule 1). Coverage is not a
  target; the parity and correctness suites are.
- **D7: the work is two packages, their evals and their profiles.**
  `retrieve/` and `evaluation/`, run on the A100, profiled with
  `torch.profiler`.

## 3. What is explicitly not the goal

| not the goal | instead |
|---|---|
| Extensive test suites, coverage targets, property tests, CPU emulation of device code | the gate's tests, nothing more (D6; CLAUDE.md rule 1) |
| Fancy formatting, cosmetic refactors, renaming for taste, docstring polish | `ruff check` and `ruff format --check` clean, then stop |
| A large baseline matrix for its own sake | the baselines a roadmap step lists, when that step comes up |
| Speculation about paper framing, venues, phrasing, reviewers | leave it unless a step asks for it |
| Defensive rewrites of working code "to be safe" | fix what a gate or a finding shows is broken |
| Compatibility shims, deprecation cycles, version negotiation for our own API | delete and move on (D2) |

## 4. Open tension with the queue

Baselines. Faiss, HNSW, cuBLAS, cuVS and filtered-graph baselines are not
the goal for their own sake, but any submission needs them: an absent or
untuned baseline invalidates a speedup claim. They
are a scheduled step (roadmap D2), run when it comes up, and no wider than
the paper needs.
