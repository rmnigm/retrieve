---
chain: "evaluation-package-layout"
branch: "main"
nextStep: "WP-V1 + WP-V2 (roadmap C5) after C4 flips and L2 lands: rename retrieval/ to bench/, split the layout contract into eval_datasets, move encode into training, one test tree, three CLIs."
created: "2026-09-15T07:00:00Z"
---

# Plan V: `evaluation/`, three packages and one dependency direction

Source: `docs/plans/evaluation-package-layout.md` §1-§10 (plan "V"), planned 2026-09-15 on `development` at `76f8985` from a file-by-file inventory after harness v2 (C1-C3) and the 2026-09-06 evaluation review. Packaging only; protocol, schema, config matrix and gates are plan H's as built. Absorbs review §1.11 (layout contract), §1.12 (training / metrics ownership), §1.4 (`resume_key` placement). As built: `docs/system/evaluation.md` and `docs/system/datasets.md`.

## 1. Starting point
One distribution `retrieve-evaluation`, 12 console scripts, no pytest config. `retrieval/` 2,958 + 2,412 tests; `eval_datasets/` 6,300 + 849 tests; `training/` 962. Problems: (1) the cycle (`retrieval/encode.py` imported `training.*`, `training/evaluate.py` imported `retrieval.metrics`); (2) the on-disk layout was a contract nobody owned (six writers, one reader re-deriving checks; two E-branches already broke it: PubMed's `cmd_queries` dropped rows, YFCC omitted sidecars); (3) query encoding split three ways; (4) algo wrappers were library code in the harness; (5) record identity spread over three files, no `report.py`; (6) twelve console scripts in three conventions; (7) tests in three places without shared config; (8) dead `__pycache__` dirs, two old-schema YAMLs, pre-v2 results not archived, a stale data-root sentence in checkpoints.md.

## 2. Decisions
- D1 three packages by concern, one direction: `eval_datasets` (on disk) <- `training` (models that make embeddings) <- `bench` (measures); enforced by a test.
- D2 `retrieval/` renamed `bench/` (the field's name read as a sibling of `retrieve`); `code_version` hashes the library, so no cell invalidated.
- D3 `eval_datasets` keeps its name (renamed from `datasets` to stop shadowing HuggingFace) and owns the layout contract (`layout.py`); ETL scripts under `eval_datasets/etl/`.
- D4 query encoding with a trained model belongs to `training` (`training/encode.py`); text encoding at ETL time stays next to the item encoding it must match.
- D5 `training/evaluate.py` gets its own ~30-line recall / ndcg (a checkpoint's reported quality must not move with the harness's metric code), pinned to `bench.metrics` at 1e-9.
- D6 `bench/` = the as-built harness + `records.py` (identity and storage) + `bench.py` -> `measure.py`; `algos.py` becomes the X §3 table after L2; H §4's refusals stand.
- D7 three console scripts: `bench` (run, campaign, report, check, upload), `eval-data` (the ETLs + fetch / publish / publish-checkpoint), `train` (sasrec, upload-checkpoint).
- D8 one test tree `evaluation/tests/` mirroring the packages, root conftest (tiny-dataset writer, `gpu` marker), `[tool.pytest.ini_options]`.
- D9 config and results as built; pre-v2 results to `results/archive/`; the two old-schema YAMLs rewritten.
- D10 the harness imports the library through `retrieve` and `retrieve.functional` only (X §2) (as built: plus `retrieve.interfaces` for `DISPATCH`).
- D11 after the C4 flip and before D1 (D1's campaign must run on the final package: logs, child command lines and upload paths are recorded artifacts).

## 3-8
Target layout and move table as now built (see `docs/system/evaluation.md`, `datasets.md`). Allowed edges: `bench -> {retrieve, retrieve.functional, training.encode, eval_datasets.layout, eval_datasets.hub}`; `training -> {eval_datasets.hub, eval_datasets.layout, eval_datasets.timesplit}`; `eval_datasets -> {}`.

## 9. Work packages
CPU gates: `ruff check evaluation`, `cd evaluation && uv run pytest tests/`, links 0. GPU gate: one `bench run` cell on goodreads d128 `c0_genre` (`silvertorch`, `triton`) with key block and `quality` equal to the pre-rename record at the same `code_version`. WP-V1 the rename, split, tests tree (CPU 1.5 d); WP-V2 CLIs and docs (0.5 d); WP-V3 = D4 `report.py`.

## 10. Risks
The rename touches every import and `test_c4_gate.py`; D5 duplicates two ~15-line functions (the alternative, a package for two functions, is worse); `eval_datasets` grows ~300 lines of readers; runbooks may still name the twelve old scripts (records are not rewritten); the old-harness golden worktree untouched.
