---
chain: "seqrec-encoder"
branch: "r11-review"
parent: "2026-09-27-010448288-w10-report.md"
nextStep: "R11 (opus, agent r11-review): review dev/hstu...dev/hstu-trainval; report as an r11-review chain note."
created: "2026-09-26T22:05:45Z"
---

# R11 brief: review gate for dev/hstu-trainval (dispatched as sent)

Fork from the W10 report. Scope: the diff only.

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-trainval; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-trainval` in worktree `/scratch/wt/trainval`
(branch `dev/hstu-trainval`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W10 added `TrainConfig.train_on_val`: val rows become train-shaped rows (the tail of `item_ids ++ targets`) with a loss mask to
the val-day positions (`first`). `target_mask` is shared by `step_loss` and `target_frequencies`. There is no val eval and no
early stop, and the last epoch is test-scored. Its report is `2026-09-27-010448288-w10-report.md`; read it.
Focus:
- **with train_on_val=false, nothing changes** (masks, logQ counts, RNG draws), including the masked-select rewrite of
  `target_frequencies`;
- off-by-ones in `first` and in the mask (target position j corresponds to `items[:, j+1]`);
- best_model.pt and eval files written from the last epoch; the resume interplay with `resume_every`;
- the tests' mutation strength and the docs.
The 25% of val-day transitions dropped for >200-click users is a known, accepted caveat, not a finding.
This merges into `dev/hstu`.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/trainval
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/trainval && python3 scripts/check_doc_links.py
```

## Return: a chain note
- Commit your fixes on `dev/hstu-trainval`, ending the message with your harness's Co-Authored-By line. Do not push or merge.
- Write your report as the chain note `/workspace/retrieve/.chains/seqrec-encoder/<$(/scratch/briefs/chain-ts)>-r11-review-report.md`. Never edit an existing note.
- Frontmatter:
  - `chain: "seqrec-encoder"`
  - `branch: "r11-review"`
  - `parent: "2026-09-27-010545607-r11-brief-review-trainval.md"`
  - `nextStep:` one concrete action for the orchestrator
  - `created:` the UTC ISO time
- Body:
  - the verdict: CLEAN, CLEAN-AFTER-FIXES, or BLOCKING with findings;
  - the fixes applied, one line each;
  - the open findings, each with file:line and its mechanism;
  - the output counts from the verify commands.
- Then stay idle.
