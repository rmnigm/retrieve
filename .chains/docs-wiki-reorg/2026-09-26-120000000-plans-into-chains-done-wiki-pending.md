---
chain: "docs-wiki-reorg"
branch: "main"
nextStep: "Fresh worker in worktree .claude/worktrees/agent-a447d566b3db5db91 on branch docs/wiki-reorg (9d608e8): audit the 50 chain notes for dropped numbers, then build the public wiki from docs/system + the two contracts, fix known stale docs, move roadmap/artifacts, relink docs/paper and CLAUDE.md, delete docs/plans."
created: "2026-09-26T09:00:00Z"
---

# Plans are chains; wiki not started

## 1. Request
User: reorganise project docs with the `wiki` skill so they're up to date; plans move into chains; use the new skillset. Constraints: public `CLAUDE.md` / `README.md` never mention `.chains/`, chains/wiki skills or agent tooling; `docs/thesis/`, `docs/presentation/` purged from history (gitignored); don't edit `articles/`, `retrieve/docs/`, code.

## 3. Work completed
- First worker (context ~800k, stopped by session end) produced 22 chains / 50 notes under `.chains/` (~41k words from ~145k in `docs/plans` + `archive`). Committed by orchestrator as `9d608e8` on `docs/wiki-reorg`, rebased on origin/development `335a390`. Link checker: 0 broken. Every note has `nextStep`; no thesis references.
- Chains: agent-orchestration(1), architecture-review-2026-09-06(2), coding-guidelines(1), cuda-silvertorch(3), cute-dsl-scorer(2), dataset-candidates(3), deterministic-compaction(1), evaluation-harness-v2(8), evaluation-package-layout(2), evaluation-refactor(1), future-work-and-research(1), kernels-layers-design(1), library-api-refactor(2), library-harness-boundary(1), linr-v2-backend-parity(2), live-update-api(1), refactor-validation-handoff(3), reproducibility-paper(2), roadmap(3), silvertorch-official-integration(6), silvertorch-reverse-clause-wrapper-fix(1), torch-export-refactor(1).

## 6. Unresolved
- Fidelity of condensation unverified beyond one spot check (C4/B3 fp16 finding kept).
- Nothing done yet on: wiki (schema/index/log), `docs/system` stale fixes, the two contracts as public pages, roadmap live page, artifacts location, `docs/paper` relinks, CLAUDE.md repointing, deleting `docs/plans`.
- User decisions pending: `.chains/` public vs private; roadmap placement; artifacts location.
- Old worktrees `agent-a166…`, `agent-a7a9…`, `agent-af08…` still exist (removal denied by classifier); they don't affect this worktree's link checker.
