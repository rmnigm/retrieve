---
chain: "seqrec-encoder"
branch: "r8-review"
parent: "2026-09-26-193949491-w7-report.md"
nextStep: "R8 (opus, agent r8-review): review dev/hstu...dev/hstu-configfix; report as an r8-review chain note."
created: "2026-09-26T16:40:12Z"
---

# R8 brief: review gate for dev/hstu-configfix (dispatched as sent)

Fork from the W7 report. Scope: the diff only.

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-configfix; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-configfix` in worktree `/scratch/wt/configfix`
(branch `dev/hstu-configfix`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W7 fixed R6's findings 1-3 in f67c32b:
- `TrainConfig.load(path, **overrides)` re-resolves normalize/logq when the override changes `loss`;
- parametrized pinning tests;
- the validation.md KuaiRand row.
Its report is `2026-09-26-193949491-w7-report.md`; R6's findings are in `2026-09-26-193335047-r6-review-report.md`.
Focus:
- the `load` override semantics (same loss vs changed loss vs explicit value);
- that the tests fail on the mutations they guard;
- the datasets.md wording.
It is a ~60-line diff, so keep the review proportionate.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/configfix
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/configfix && python3 scripts/check_doc_links.py
```

## Return: a chain note
- Commit your fixes on `dev/hstu-configfix`, ending the message with your harness's Co-Authored-By line. Do not push or merge.
- Write your report as the chain note `/workspace/retrieve/.chains/seqrec-encoder/<$(/scratch/briefs/chain-ts)>-r8-review-report.md`. Never edit an existing note.
- Frontmatter:
  - `chain: "seqrec-encoder"`
  - `branch: "r8-review"`
  - `parent: "2026-09-26-194012821-r8-brief-review-configfix.md"`
  - `nextStep:` one concrete action for the orchestrator
  - `created:` the UTC ISO time
- Body:
  - the verdict: CLEAN, CLEAN-AFTER-FIXES, or BLOCKING with findings;
  - the fixes applied, one line each;
  - the open findings, each with file:line and its mechanism;
  - the output counts from the verify commands.
- Then stay idle.
