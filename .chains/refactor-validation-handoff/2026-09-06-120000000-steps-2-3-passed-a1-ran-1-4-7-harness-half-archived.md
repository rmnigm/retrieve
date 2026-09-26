---
chain: "refactor-validation-handoff"
branch: "main"
parent: "2026-07-06-120000000-gpu-validation-runbook-for-the-refactor-track.md"
nextStep: "Steps 5 and 6 remain: scripted in a1_golden_run.sh (stages step5, step6) with a1_step5_compare.py and a1_step6_diff.py; the main-side worktree tmp/main-users-limit-fix also needs the two A1 fixes before step 6 can run."
created: "2026-09-06T12:00:00Z"
---

# Steps 2-3 passed on the A100; A1 ran steps 1, 4, 7; harness half archived

## Work completed
- 2026-09-02: steps 2-3 passed de facto on an A100: the full `retrieve/tests/` suite green at the branch tip (286 tests, `docs/artifacts/cute-dsl-scorer/wp4/pytest-final.txt`); the K2 tracing caveat did not bite.
- 2026-09-06, under roadmap A1 (branch `dev/a1-golden`; record in the evaluation-harness-v2 chain, WP-0): step 1 passes; steps 4 and 7 ran on the A100 for the first time and passed.
- Roadmap C3 deleted the old harness (`cli/evaluate.py`, `cli/run_evaluation.py`, `sweep.py`, `passes.py`, `loaders.py`, `queries_cache.py`, 19 YAMLs), so steps 1, 4, 6, 7, the harness delta list and F4 / F6 moved verbatim to `docs/plans/archive/refactor-validation-handoff-harness.md`.

## Two bugs found, both hidden because steps 1 and 4 to 7 had never run
1. Filter-attrs misalignment: the Hub datasets are the pre-`3b1b5b3` `[N+1, ...]` 1-indexed artifacts; loaders, ETL and docs expected `[N, ...]`. goodreads crashed in the oracle (797085 vs 797084); arxiv did not crash (attrs and embeddings both 1-indexed), only the held-out target shift was wrong: `cos(query, target)` 0.99 to 0.62, no error. Fix: accept the legacy layout. Not a refactor regression (`main` and `development` have byte-identical `load_filter_assets`).
2. `common.clause_pass` under inductor: K3 called helpers as `common.clause_pass(...)`; `torch.compile` rebuilds kernel globals and captures `@triton.jit` callees by name, so every compiled filter algo died with `NameError('common is not defined')`. Fix: import the helpers by name. Exactly what step 4 exists to catch.

## Deferred
Step 5 (per-kernel +-5 %) and the `main` half of step 6 deferred 2026-09-06 (user: heavy evals later). Step 5 estimate: 332 sweep points per side, ~44 min for both at 4 s/point.
