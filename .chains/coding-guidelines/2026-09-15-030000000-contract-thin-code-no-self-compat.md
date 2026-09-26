---
chain: "coding-guidelines"
branch: "main"
nextStep: "None: a standing contract. The canonical text now lives in docs/system/coding-guidelines.md."
created: "2026-09-15T03:00:00Z"
---

# Coding guidelines contract, effective 2026-09-15

Source: `docs/plans/coding-guidelines.md`, effective 2026-09-15 on `development` at `a2182d8` (user decision, no code changed). Orders nothing, gates nothing; CLAUDE.md rules 1-8 win over it. Moved to the public wiki as `docs/system/coding-guidelines.md` on 2026-09-26.

Priorities in order: correct code (bit-exact where a plan says so; never traded away), clean architecture, evals that run and profiling that explains them. When two conflict the lower number wins.

Decisions:
- D1 little defensive programming: no guards for conditions the caller cannot produce, no try/except turning a bug into a degraded result, no re-validating constructor invariants; validate at the boundary (public constructor, CLI argument, file read) and trust the data after. The harness rule "no exception swallowing, no retries" (H §4) generalises to the library.
- D2 no backward compatibility with ourselves: redo a part and delete what it replaced; no alias, shim, deprecation path or second code path for a feature we invented. Compatibility is owed only to what we do not control: Meta's op schemas, `torch.ops` signatures, on-disk dataset layouts, published checkpoints, a recorded `code_version`'s meaning. Does not weaken CLAUDE.md rule 5 (a kernel goes only after its replacement's parity gate).
- D3 thin and concise: a function over a class, a dict over a registry, a parameter over a subclass (H §4's list is the worked example).
- D4 no multi-level abstraction without a demonstrated need: one indirection needs a reason, two need a second one written down; a base class with one subclass, a protocol with one implementation, a factory that builds one thing is a defect.
- D5 no comment slop; document outside the code: no narration, no restated signatures, no banners, no commented-out code; a short why-comment only where the code is genuinely surprising (a bit order, an overflow bound, a sync-avoidance trick, a deviation from a paper).
- D6 tests are gates, not a deliverable: the test a gate names and the test that pins a fixed bug; no scaffolding beyond that; coverage is not a target.
- D7 the work is two packages, their evals and their profiles (`torch.profiler` on the A100).

Not the goal: extensive suites, coverage, property tests, CPU emulation; cosmetic refactors and docstring polish (ruff clean and stop); a large baseline matrix for its own sake (run the baselines a step lists, when it comes up); speculation about paper framing or venues; defensive rewrites of working code; shims and deprecation cycles for our own API.

Collisions with live plans at the time (listed, not silently overridden):
| collision | plan | resolution |
|---|---|---|
| the `retrieve.layers` / `retrieve.kernels` shim | L D10 and L WP-1's gate | user 2026-09-15: temporary tooling for the old-harness golden worktree, deleted at a later step (deleted at L5) |
| `torchretrieve` is on PyPI | L §7 | at 0.x, renaming without a deprecation cycle is defensible; decide explicitly |
| Faiss / HNSW / cuBLAS / cuVS / filtered-graph baselines | roadmap D2; P G5, G13, G14 | baselines are a scheduled step (G5 is P0 for any submission), not background work |
