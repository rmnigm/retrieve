---
chain: "seqrec-encoder"
branch: "r2-review"
parent: "2026-09-26-151400000-w2-report.md"
nextStep: "R2 (opus, agent r2-review, pane w1:p5, worktree /scratch/wt/etl): deslop + python-review on dev/hstu...dev/hstu-etl; report."
created: "2026-09-26T12:19:00Z"
---

# R2 brief: review gate for dev/hstu-etl (dispatched as sent)

Fork from the W2 report: the review agent's branch. Scope: the diff only. What merges back: the verdict and any fix commits on `dev/hstu-etl`.

---

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-etl; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-etl` in worktree `/scratch/wt/etl`
(branch `dev/hstu-etl`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W2 (opus) added a `timestamps` column (list[int64]) to the train/val/test outputs of
`evaluation/eval_datasets/etl/{goodreads,yambda,kuairand}.py`, extended
`tests/eval_datasets/test_kuairand.py`, and updated `docs/system/datasets.md` and
`docs/validation.md`. Its report is at `/scratch/briefs/W2-report.md`. Read it for the gate results
(item_id_map byte-identical; yambda val/test row order and goodreads same-timestamp tie order are
nondeterministic in the existing ETLs, which is reported and out of scope to fix). Do not re-run the
preps. The artifacts under `docs/artifacts/seqrec-encoder/etl-timestamps/` must stay small and should
be scripts plus JSON only.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/etl
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/etl && python3 scripts/check_doc_links.py
```

## Return
- Your fixes as commits on `dev/hstu-etl` (message ends with your harness's Co-Authored-By line).
  Do not push or merge.
- Write `/scratch/briefs/R2-report.md` with: the verdict (CLEAN, or CLEAN-AFTER-FIXES, or
  BLOCKING with findings), the fixes applied (one line each), open findings with file:line and
  mechanism, and the verify command outputs (counts).
- Then stay idle.
