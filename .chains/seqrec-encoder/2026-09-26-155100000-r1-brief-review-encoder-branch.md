---
chain: "seqrec-encoder"
branch: "r1-review"
parent: "2026-09-26-154800000-w1-report.md"
nextStep: "R1 (opus, agent r1-review, pane w1:p6, worktree /scratch/wt/encoder): deslop + python-review on dev/hstu...dev/hstu-encoder; write the report as 2026-09-26-155500000-r1-review-report.md on branch r1-review."
created: "2026-09-26T12:31:00Z"
---

# R1 brief: review gate for dev/hstu-encoder (dispatched as sent)

Fork from the W1 report: the review agent's branch. Scope: the diff only. What merges back: the verdict and any fix commits on `dev/hstu-encoder`.

**Changed after dispatch:** the Return section. The report goes to the chain note `2026-09-26-155500000-r1-review-report.md` (branch r1-review), not `/scratch/briefs/R1-report.md`.

---

You are a constrained review agent dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single non-secret
  variables by name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Use
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-encoder; uv sync --all-packages --all-groups --extra official`
  in the worktree.
- No GPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another agent holds the GPU.
- The code wins over docs. State the mechanism of a bug before fixing it.

## Step
Review gate before merge: review the diff `git diff dev/hstu...dev/hstu-encoder` in worktree `/scratch/wt/encoder`
(branch `dev/hstu-encoder`).
1. Invoke the `deslop` skill on that diff and apply its behaviour-preserving fixes.
2. Invoke the `python-review` skill on that diff and apply the fixes that are correctness bugs
   or clear guideline violations (coding-guidelines D1-D6). Anything that changes behaviour or
   that you are unsure about goes into findings, not code.
3. Check that the docs match the code (`docs/system/*`, `docs/validation.md`) and that no number is
   presented as citable.

Stay inside the diff's files: no drive-by refactors elsewhere, no new tests beyond one that pins a
bug you fixed.

## Context
W1 (opus) rewrote `evaluation/training/`: one `Encoder` with `sasrec | hstu` blocks (HSTU with softmax
SDPA plus a relative position/time bias as an additive float mask, and U-gating), `gbce` (per-position
negatives) and `sampled_softmax` (one shared uniform vector plus in-batch, normalized embeddings,
temperature), `TrainConfig`, `train run`, torch.compile on the dense body, GPU-resident batching.
It also touched `eval_datasets/hub.py` and the tests, and updated `docs/system/datasets.md`
§ Training / `checkpoints.md` / `validation.md`. Its full report is `/scratch/briefs/W1-report.md`;
read it, including its plan amendments. Those were accepted by the orchestrator, so do not undo them.

Focus for python-review: the attention mask (left padding, no NaN, padding rows), the sampled-softmax
accidental-hit masking and the positive column, the float64 gBCE calibration kept intact,
`normalize` applied consistently in the loss, evaluate, encode and item_embs, `load_model_for_eval`
still loading the published checkpoints (legacy key rename, `D128_DROP05_DEFAULTS`), resume RNG
state, dead code left from GSASRec, the no-alias rename, and comment slop.
Do not re-run GPU gates; the CPU suite is enough. `ruff format --check` has 7 pre-existing failures
outside this diff (arxiv/goodreads/pubmed/synth_arxiv/yfcc/yfcc_check_gt ETLs and hub.py per an earlier
review). Only files in this diff must be formatted.

## Verify commands (all must be clean before you hand back)
```
cd /scratch/wt/encoder
ruff check retrieve evaluation && ruff format --check evaluation/training evaluation/eval_datasets
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
cd /scratch/wt/encoder && python3 scripts/check_doc_links.py
```

## Return
- Your fixes as commits on `dev/hstu-encoder` (message ends with your harness's Co-Authored-By line).
  Do not push or merge.
- Write `/scratch/briefs/R1-report.md` with: the verdict (CLEAN, or CLEAN-AFTER-FIXES, or
  BLOCKING with findings), the fixes applied (one line each), open findings with file:line and
  mechanism, and the verify command outputs (counts).
- Then stay idle.
