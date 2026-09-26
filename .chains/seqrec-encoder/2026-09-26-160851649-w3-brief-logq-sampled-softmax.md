---
chain: "seqrec-encoder"
branch: "w3-logq"
parent: "2026-09-26-155500001-w1-merged-dispatch-e-runs.md"
nextStep: "W3 (opus, agent w3-logq, worktree /scratch/wt/logq, branch dev/hstu-logq): add logQ correction to sampled_softmax per this brief; CPU only; report as a w3-logq chain note."
created: "2026-09-26T13:08:51Z"
---

# W3 brief: logQ correction for sampled softmax (dispatched as sent)

Fork from `main`: the fix-agent branch. Scope: `evaluation/training/{config,losses,train}.py`,
`tests/training/test_encoder.py`, `docs/system/datasets.md` § Training, `docs/validation.md` (unit
gate row). What merges back: `dev/hstu-logq` after review.

You are a constrained worker dispatched by the dev/hstu orchestrator. Read `CLAUDE.md` (rule 9:
never print env or secrets files) and `docs/contracts/coding-guidelines.md` first.
**No subagents. No GPU** (`CUDA_VISIBLE_DEVICES=""`; the run worker holds the GPU).
The code wins over docs. State the mechanism of any bug before fixing it.

## Step
The user wants sampled softmax **with logQ**. E1 (in-batch 4096 + uniform 8192, no logQ) scored far
below gBCE on yambda, probably because the in-batch negatives suppress popular items. Add the
standard correction:
- `TrainConfig.logq: bool = False`.
- When on, every **candidate** column's logit (after scaling by temperature) gets `- log q_j`, where
  q_j is the probability that the mixed proposal draws item j:
  `q_j = (M * p_train(j) + K / N) / (M + K)`.
  - M = the in-batch candidates actually used, K = the uniform draws, N = the number of items.
  - p_train = item frequency over the train **target** positions, computed once at startup with a
    bincount on the GPU.
  - When M = 0, q is uniform, so the correction is a constant and changes nothing. That is fine.
- The **positive column is not corrected** ("Correcting the LogQ Correction", arXiv 2507.09331).
- Accidental hits stay masked to -inf, as now.
- Reject `logq=true` with `loss=gbce` at the `TrainConfig` boundary. That follows the same pattern
  R1 added for `normalize`.

## Read only
`evaluation/training/{config,losses,train}.py`, `evaluation/tests/training/test_encoder.py`,
`docs/system/datasets.md` § Training.

## Gates
- The existing sampled-softmax test (against a direct `F.cross_entropy`) is extended with a logQ
  case: the oracle is built by hand on a tiny example (explicit candidate list, explicit q). It
  must fail if the sign flips or if the positive gets corrected. Tolerance is stated at the call
  site with a reason.
- The config rejection is pinned with `pytest.raises(..., match=...)`.
- `ruff check retrieve evaluation`, `ruff format --check evaluation/training`, the evaluation
  suite (`cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q`), and
  `python3 scripts/check_doc_links.py` must all be clean.

## Branch/worktree
`/scratch/wt/logq`, branch `dev/hstu-logq` (off dev/hstu 02dccfd). Venv:
`export UV_PROJECT_ENVIRONMENT=/venvs/wt-logq; uv sync --all-packages --all-groups --extra official`.
**Not `rp-sync`**: it resets the venv variable. Commit, but do not push or merge.

## Out of scope
Any other loss change, the "correct positive" variant, running on the GPU, the R1 resume follow-ups,
and formatting of other files.

## Return
Write a chain note `/workspace/retrieve/.chains/seqrec-encoder/$(/scratch/briefs/chain-ts)-w3-report.md`.
- Frontmatter: `chain: "seqrec-encoder"`, `branch: "w3-logq"`, `parent:` this brief's filename,
  `nextStep` for the orchestrator, and `created` in UTC ISO.
- Body: what changed, the verify outputs, and what is unverified.

Then stay idle. Never edit an existing note.
