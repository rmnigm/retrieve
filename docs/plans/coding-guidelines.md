# Coding guidelines — what the code should look like, and what it is for

> **Status:** effective 2026-09-15 on `development` at `a2182d8` (user
> decision; no code changed). A contract, not a work list. It orders nothing
> and gates nothing — [00-roadmap.md](00-roadmap.md) keeps both — it says what
> good work looks like when a step is executed, and what is deliberately not
> worth doing. Companion: [agent-orchestration.md](agent-orchestration.md)
> (who executes a step); [../../CLAUDE.md](../../CLAUDE.md) rules 1–8 remain
> the hard rules and win over anything here.
>
> Question this document answers: *given two packages, one GPU and a
> reproducibility paper at the end of it, what should a worker actually write —
> and what should it refuse to write?*

## 1. Priorities, in order

1. **Correct code.** Bit-exact where a plan says bit-exact (CLAUDE.md rule 3).
   This is the only thing that is never traded away.
2. **Clean architecture.** Thin, direct, one obvious home per concern.
3. **Evals that run**, and profiling that explains them. The numbers are the
   product.

Everything else is subordinate. When two of these conflict, the lower number
wins; when a guideline below conflicts with one of these, the priority wins.

## 2. Decisions

- **D1 — Little defensive programming.** No guard for a condition the caller
  cannot produce, no `try`/`except` that turns a bug into a degraded result,
  no re-validating an invariant the constructor already established. Let it
  fail loudly at the real line: a stack trace on this box is a better
  diagnostic than a swallowed error anywhere. Validate at the **boundary**
  (a public constructor, a CLI argument, a file read) and then trust the data.
  The harness's rule is already this — "no exception swallowing, no retries"
  ([evaluation-harness-v2.md](evaluation-harness-v2.md) §4) — and it
  generalises to the library.
- **D2 — No backward compatibility with ourselves.** This project is being
  built from scratch; its own past shapes are not a constraint. Redo a part,
  delete what it replaced, and do not keep an alias, a shim, a deprecation
  path or a second code path alive for a feature we invented. Compatibility is
  owed only to things we do not control: Meta's op schemas, `torch.ops`
  signatures, on-disk dataset layouts, published checkpoints, and a recorded
  `code_version`'s meaning. **This does not weaken CLAUDE.md rule 5**: a
  *kernel* is still deleted only after its replacement's parity gate is green.
  Ditching our own API is free; ditching a validated implementation is not.
- **D3 — Thin and concise.** The smallest code that is correct and readable.
  Prefer a function to a class, a dict to a registry, a parameter to a
  subclass. The harness plan's §4 list ("no context bags, no stats
  dataclasses, no algo framework, no config object model, no timing strategy
  classes") is the worked example; apply the same taste in the library.
- **D4 — No multi-level abstraction without a demonstrated need.** One level of
  indirection needs a reason; two need a second, written down. A base class
  with one subclass, a protocol with one implementation, or a factory that
  builds one thing is a defect, not a design. "We might need it later" is not
  a need — later is when we add it, and by then we will know the shape.
- **D5 — No comment slop; document outside the code.** No multi-line
  commentary narrating what the next lines do, no restating a signature in
  prose, no banner blocks, no commented-out code. Prefer **no comment at all**
  and a name that makes it unnecessary. What survives: a short note where the
  code is genuinely surprising — a bit order, an overflow bound, a
  sync-avoidance trick, a deviation from a paper — and it says *why*, in one
  or two lines. The real documentation is `docs/system/` (what the code does
  today) and `docs/plans/` (why, in what order, and what was measured), kept
  current as the code changes (CLAUDE.md rule 4).
- **D6 — Tests are gates, not a deliverable.** Write the test a plan's gate
  names, and the test that pins a bug you just fixed. Do not build test
  scaffolding, fixtures, helpers or matrices beyond that — CLAUDE.md rule 1
  already forbids it, and this is the same instruction from the other side.
  Coverage is not a target; the parity and correctness suites are.
- **D7 — The work is two packages, their evals, and their profiles.**
  `retrieve/` and `evaluation/`, run on this A100, profiled with
  `torch.profiler`. Work that is not one of those is not the job.

## 3. What is explicitly not the goal

Named here so a worker does not drift into them, and so the orchestrator can
say "out of scope" by citing a line:

| not the goal | what to do instead |
|---|---|
| Extensive test suites, coverage targets, property tests, CPU emulation of device code | the gate's tests, nothing more (D6; CLAUDE.md rule 1) |
| Fancy formatting, cosmetic refactors, renaming for taste, docstring polish passes | `ruff check` + `ruff format --check` clean, and stop |
| A large baseline matrix for its own sake | run the baselines a roadmap step actually lists, when that step comes up (§4) |
| Speculation about paper framing, venues, phrasing, reviewer psychology | [reproducibility-paper.md](reproducibility-paper.md) already holds that research; do not re-derive or extend it unprompted |
| Defensive rewrites of working code "to be safe" | leave it; fix what a gate or a finding shows is broken |
| Compatibility shims, deprecation cycles, version-negotiation for our own API | delete and move on (D2) |

## 4. Where this collides with a live plan

Three live decisions were taken before these guidelines and point the other
way. They are **not** silently overridden here — a plan decision is changed by
amending that plan, and the orchestrator does that with the user. Listed so
the collision is visible when the step comes up:

| collision | plan | the tension |
|---|---|---|
| The one-release compatibility shim (`retrieve.layers` / `retrieve.kernels` re-exporting under `DeprecationWarning`, removed in 0.3) | [library-api-refactor.md](library-api-refactor.md) D10, and WP-1's gate "the golden worktree runs one cell through the shim" | D2 says no compat with ourselves. But the shim is not an API promise to users — it is the tool that lets the **old-harness golden worktree** run against the new library for the golden re-derive. Recommendation: keep it as *temporary tooling* with a deletion step, or drop it and port the golden worktree's imports instead — the user's call |
| `torchretrieve` is published on PyPI, so the package does have external consumers | [library-api-refactor.md](library-api-refactor.md) §7 | D2's "from scratch" holds for internal shape; a released name is a thing we do not fully control. At 0.x, renaming without a deprecation cycle is defensible — decide explicitly rather than by default |
| Faiss / HNSW / cuBLAS / cuVS / filtered-graph baselines | roadmap D2; [reproducibility-paper.md](reproducibility-paper.md) G5, G13, G14 | "a shit ton of baselines" is not the goal, but G5 is P0 for any submission (an untuned or absent baseline invalidates a speedup claim). Reading: baselines are a *scheduled step*, run when D2 comes up — not background work, not expanded beyond what the paper needs |

## 5. Validation record

*(nothing is recorded here: this document is a contract. Code written under it
is recorded in the plan whose step produced it —
[agent-orchestration.md](agent-orchestration.md) §5.)*
