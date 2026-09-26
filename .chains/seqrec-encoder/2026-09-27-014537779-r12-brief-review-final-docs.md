---
chain: "seqrec-encoder"
branch: "r12-review"
parent: "2026-09-27-014448404-w11-report.md"
nextStep: "R12 (opus, agent r12-review): review dev/hstu...dev/hstu-final; report as an r12-review chain note."
created: "2026-09-26T22:45:37Z"
---

# R12 brief: review gate for dev/hstu-final (dispatched as sent)

Fork from the W11 report. Scope: the diff only.

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-final; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-final` in worktree `/scratch/wt/final`
(branch `dev/hstu-final`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W11 merged the run artifacts (origin/dev/hstu-runs 4ba35d6) and wrote the final docs: the KuaiRand rows and a drift
subsection in validation.md, decisions.md, a roadmap follow-up item, the datasets.md § kuairand refresh, the contract
wording on `.chains/` snapshots, and a `.chains/` skip in `scripts/check_doc_links.py`. Its report is
`2026-09-27-014448404-w11-report.md`.
Focus:
- **every KuaiRand number in validation.md matches its artifact JSON**; spot-check at least the k64-refit, k64,
  calibration and diagnosis figures;
- no number is presented as citable;
- one home per fact (decisions.md links rather than restates);
- the link-checker change is minimal and still checks everything else (run it, and confirm a deliberately broken link
  in a docs/ page is still caught, then revert);
- the roadmap item is accurate.
Docs-and-script diff; the CPU suite is enough.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/final
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/final && python3 scripts/check_doc_links.py
```

## Return: a chain note
- Commit your fixes on `dev/hstu-final`, ending the message with your harness's Co-Authored-By line. Do not push or merge.
- Write your report as the chain note `/workspace/retrieve/.chains/seqrec-encoder/<$(/scratch/briefs/chain-ts)>-r12-review-report.md`. Never edit an existing note.
- Frontmatter:
  - `chain: "seqrec-encoder"`
  - `branch: "r12-review"`
  - `parent: "2026-09-27-014537779-r12-brief-review-final-docs.md"`
  - `nextStep:` one concrete action for the orchestrator
  - `created:` the UTC ISO time
- Body:
  - the verdict: CLEAN, CLEAN-AFTER-FIXES, or BLOCKING with findings;
  - the fixes applied, one line each;
  - the open findings, each with file:line and its mechanism;
  - the output counts from the verify commands.
- Then stay idle.
