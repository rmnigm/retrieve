---
chain: "agent-orchestration"
branch: "main"
nextStep: "None: a standing contract. The canonical text now lives in docs/system/agent-orchestration.md; amend it there (with the user) and record why here."
created: "2026-09-15T02:00:00Z"
---

# Agent orchestration contract, effective 2026-09-15

Source: `docs/plans/agent-orchestration.md`, effective 2026-09-15 on `development` at `a2182d8` (user decision, no code changed). A contract, not a work list: it orders nothing (the roadmap does) and relaxes no CLAUDE.md hard rule. Moved to the public wiki as `docs/system/agent-orchestration.md` on 2026-09-26; the decisions below are the history of that page.

Question: when a roadmap step is dispatched to an agent, what is the session shape, which model takes it, how many run in parallel, and what has to exist when it is done?

## Decisions
- D1 one orchestrator, for the whole session: not a pool or a hand-off chain (a second coordinator would mean two sessions believing they own the checkboxes and the merge).
- D2 workers are constrained: handed the step, its plan section, its gates, its branch; they return a validation record and a working tree; scope changes go back to the orchestrator.
- D3 model by the shape of the work, not its importance: design-and-write (large kernel or architecture rewrites) vs care and follow-through.
- D4 a `fable` worker gets a big chunk, not a slice (splitting a library rewrite reintroduces the coordination the split avoids; why L WP-1 was one worker: nothing else may edit `retrieve/` while it is open).
- D5 exactly two levels, one exception: a `sonnet` subagent for web deep research. Deeper nesting produces unreviewed work and unwritten records.
- D6 three workers at once maximum; a ceiling, not a target.
- D7 every worker's result is documented in a plan: a run only in a transcript did not happen.
- D8 all work ends in `development` on `origin`.

## Routing table
fable: core library rewrites with heavy kernel or coding work (a Triton kernel written or retuned, the modules / ops / indexing move, a backend adapter); core harness rewrites carrying architecture design (the package split, the cell loop, the record schema). opus: routine cleanups and plan-following refactors, monitoring and babysitting runs, debugging a red gate, documentation and link fixes, one-time experiments and probes. sonnet: web deep research, only nested under a worker. As routed at the time: L1 / L2 and C5 fable; C4's rerun, B3, D1, D4, the Phase E loaders, F1 / F3 opus.

## Concurrency and nesting
Below the ceiling of three: the GPU is one machine and serialized (one worker holds GPU work at a time); plan-level exclusivity (L WP-1 over `retrieve/`); disjoint trees (two concurrent workers never edit the same files). The only child a worker may spawn is a sonnet web-research subagent (external practice, a venue CfP, a dataset licence, upstream source not in this tree).

## Handed / returned
Handed: the step and plan section (nothing else to read), the gates quoted incl. which are bit-exact, the branch and worktree, the model and concurrency constraint, what is out of scope. Returned: a validation record appended to the executed plan (model: the CUDA handoff §13: environment, what ran, what passed, skipped and why, still unverified); raw scripts and outputs under the plan's artifacts dir, not in the packages; the matching system doc updated in the same commit if behaviour changed, links at zero; a plain statement of passed / skipped / unverified, never a number presented as citable before its gate. The orchestrator flips the checkbox; a worker does not.

## Where work lands
`dev/<step>` (optionally a worktree under `/workspace/wt/`), gates green + record + review fixes -> `development` -> push `origin/development`. The orchestrator owns merge and push; it commits when the user asks, and pushing is the same outward action under the same permission. `main` untouched; A4 (merge into `main`) on hold, the user's call.

Later lesson (2026-09-15, orchestrator): verify a worker's gates yourself before merging; a worker reported a suite green that was not (its own docs edit broke a markdown-parsing test).
