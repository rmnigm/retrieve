---
chain: "seqrec-encoder"
branch: "r6-review"
parent: "2026-09-26-193007031-w6-report.md"
nextStep: "R6 (opus, agent r6-review): deslop + python-review on dev/hstu...dev/hstu-cleanup; report as an r6-review chain note."
created: "2026-09-26T16:30:53Z"
---

# R6 brief: review gate for dev/hstu-cleanup (dispatched as sent)

Fork from the W6 report. Scope: the diff only.

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-cleanup; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-cleanup` in worktree `/scratch/wt/cleanup`
(branch `dev/hstu-cleanup`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W6 (opus) deleted HSTU, RelativeBias, use_time/time plumbing, hidden_dim/in_proj/out_proj and the encoder
switch from `evaluation/training/`. It kept both losses (gbce | sampled_softmax with logQ), made the E1c
recipe the `TrainConfig` defaults and updated the docs. Its report is the chain note
`2026-09-26-193007031-w6-report.md`; read it, including the bit-identical equivalence and Gate A evidence.
Focus:
- leftovers (dead params, unused imports, stale docstrings or docs mentioning hstu/use_time/encoder=);
- the `normalize`/`logq` None-resolution logic and its rejections;
- `TrainConfig.load` legacy handling (no `loss` key means gbce; unknown keys dropped);
- `D128_DROP05_DEFAULTS`;
- that the defaults equal E1c's `config.json` (docs/artifacts/seqrec-encoder/e1c-*/config.json);
- the artifact scripts under `docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/` are small and not in packages.
Do not rerun the GPU gates; the CPU suite is enough.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/cleanup
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/cleanup && python3 scripts/check_doc_links.py
```

## Return: a chain note
- Commit your fixes on `dev/hstu-cleanup`, ending the message with your harness's Co-Authored-By line. Do not push or merge.
- Write your report as the chain note `/workspace/retrieve/.chains/seqrec-encoder/<$(/scratch/briefs/chain-ts)>-r6-review-report.md`. Never edit an existing note.
- Frontmatter:
  - `chain: "seqrec-encoder"`
  - `branch: "r6-review"`
  - `parent: "2026-09-26-193053027-r6-brief-review-cleanup.md"`
  - `nextStep:` one concrete action for the orchestrator
  - `created:` the UTC ISO time
- Body:
  - the verdict: CLEAN, CLEAN-AFTER-FIXES, or BLOCKING with findings;
  - the fixes applied, one line each;
  - the open findings, each with file:line and its mechanism;
  - the output counts from the verify commands.
- Then stay idle.
