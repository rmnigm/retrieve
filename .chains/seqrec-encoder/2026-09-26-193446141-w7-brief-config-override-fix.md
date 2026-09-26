---
chain: "seqrec-encoder"
branch: "w7-configfix"
parent: "2026-09-26-193335047-r6-review-report.md"
nextStep: "W7 (opus, agent w7-configfix, worktree /scratch/wt/configfix, branch dev/hstu-configfix, CPU only): fix R6 findings 1-4; report as a w7 chain note."
created: "2026-09-26T16:34:46Z"
---

# W7 brief: loss-override trap and pinning tests (R6 findings 1-4) (dispatched as sent)

Fork from the R6 report. Scope: `evaluation/training/{config,train,model}.py`, `tests/training/`, and
`docs/system/datasets.md` § Training and `docs/validation.md` as far as these touch them. dev/hstu is at
1ee5d4c (cleanup merged).

You are a constrained worker. Read `CLAUDE.md` (rule 9: never print env or secrets files) and
`docs/contracts/coding-guidelines.md`.
- **No subagents. No GPU** (`CUDA_VISIBLE_DEVICES=""`; the run worker is training).
- Venv: `export UV_PROJECT_ENVIRONMENT=/venvs/wt-configfix; uv sync --all-packages --all-groups --extra official`.
  **Not rp-sync.**

## Step
Read `2026-09-26-193335047-r6-review-report.md` (the parent note), findings 1-4. Then:

1. **Finding 1 (bug):** `train run --config X loss=<other>` must resolve `normalize` and `logq` for the
   *resulting* loss, unless they are given explicitly. Pick the thinnest fix. Keeping `None` in the saved or
   loaded config until `__post_init__` is the likely shape. State the mechanism in the commit message.
   Existing saved `config.json` files hold resolved bools; say how they behave after the fix.
2. **Finding 2 (pin):** tests for:
   - `TrainConfig().loss == "sampled_softmax"` with `normalize`/`logq` True;
   - `TrainConfig(loss="gbce")` resolving both to False;
   - a `config.json` without a `loss` key loading as gbce;
   - the finding-1 override in both directions.
   Each must fail on the mutation it guards against (check by hand). Use parametrized cases, no new
   scaffolding.
3. **Finding 3:** `docs/validation.md` KuaiRand `timestamps` row: the staged data now has the column
   (`2026-09-26-184409449-w5-report.md`). Fix the row.
4. **Finding 4:** remove `Encoder.__init__`'s default values, so `TrainConfig` is the one home for defaults.
   Keep this only if every caller passes every argument (check `encode.py`, the tests and the fixtures);
   otherwise report it and leave it.

## Verify
`ruff check retrieve evaluation`, `ruff format --check evaluation/training`,
`cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q`, and `python3 scripts/check_doc_links.py`
must all be clean. Commit on dev/hstu-configfix; do not push or merge.

## Out of scope
R6 findings 5-6, performance work and anything else.

## Return
Write the chain note `/workspace/retrieve/.chains/seqrec-encoder/$(/scratch/briefs/chain-ts)-w7-report.md`:
- frontmatter: branch `w7-configfix`, parent = this brief's filename, `nextStep` for the orchestrator,
  `created` in UTC ISO;
- body: changes, mutation checks, verify outputs.

Then stay idle.
