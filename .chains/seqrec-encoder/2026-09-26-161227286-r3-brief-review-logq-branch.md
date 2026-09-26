---
chain: "seqrec-encoder"
branch: "r3-review"
parent: "2026-09-26-161153746-w3-report.md"
nextStep: "R3 (opus, agent r3-review, worktree /scratch/wt/logq): deslop + python-review on dev/hstu...dev/hstu-logq; report as an r3-review chain note."
created: "2026-09-26T13:12:27Z"
---

# R3 brief: review gate for dev/hstu-logq (dispatched as sent)

Fork from the W3 report: the review agent's branch. Scope: the diff only.

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-logq; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-logq` in worktree `/scratch/wt/logq`
(branch `dev/hstu-logq`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W3 (opus) added `TrainConfig.logq` and the logQ correction to `sampled_softmax_loss`: candidates get
`- log q_j` with q from the mixed in-batch plus uniform proposal, and the positive is never
corrected. It also added `target_frequencies` in train.py, the tests, and the docs. Its report is the
chain note `2026-09-26-161153746-w3-report.md` in the same directory; read it.
Focus: the q_j formula and M/K/N in `step_loss`, the `target_frequencies` bincount trick (column 0
removal, padding id 0 excluded), dtype (float64 then fp32), the interaction of the -inf accidental-hit
masking with the correction, and that `log_q=None` leaves plain sampled softmax bit-identical
(W3 moved where the temperature division happens: check it). A small diff; keep the review
proportionate.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/logq
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/logq && python3 scripts/check_doc_links.py
```

## Return: a chain note
- Commit your fixes on `dev/hstu-logq`, ending the message with your harness's Co-Authored-By line. Do not push or merge.
- Write your report as the chain note `/workspace/retrieve/.chains/seqrec-encoder/<$(/scratch/briefs/chain-ts)>-r3-review-report.md`. Never edit an existing note.
- Frontmatter:
  - `chain: "seqrec-encoder"`
  - `branch: "r3-review"`
  - `parent: "2026-09-26-161227286-r3-brief-review-logq-branch.md"`
  - `nextStep:` one concrete action for the orchestrator
  - `created:` the UTC ISO time
- Body:
  - the verdict: CLEAN, CLEAN-AFTER-FIXES, or BLOCKING with findings;
  - the fixes applied, one line each;
  - the open findings, each with file:line and its mechanism;
  - the output counts from the verify commands.
- Then stay idle.
