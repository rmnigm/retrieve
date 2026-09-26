---
chain: "review-pass"
branch: "main"
nextStep: "Orchestrator: review commit 77b376e on dev/review-pass (worktree /scratch/wt/review), merge into staging and push. Then decide whether openalex should stop importing eval_datasets.layout at module level (it still loads torch just for --help/plan/download)."
created: "2026-09-26T02:28:53Z"
---

# evaluation/ review pass: shared ETL helpers, flat code (commit 77b376e)

## Primary request
A constrained worker ran a python-review pass over `evaluation/` only (bench, eval_datasets incl. the new kuairand/openalex/pubmed modules, training). The fixes had to be bounded and behaviour-preserving. Out of scope: `retrieve/`, `docs/roadmap.md`, public signatures, CLI shapes, config schemas, on-disk layouts, and the deferred backlog items.

## State
- Branch `dev/review-pass`, local commit `77b376e`, **not pushed or merged** (as instructed).
- Harness suite `cd evaluation && CUDA_VISIBLE_DEVICES="" uv run --no-sync pytest tests/ -q`: 236 passed, 1 skipped, both before and after the change.
- `ruff check evaluation`: clean. `ruff format --check evaluation`: the same 15 pre-existing files as before, and no new ones. For the backlog files I edited (pubmed, yfcc, goodreads, test_pubmed, upload), the formatter diff did not grow.
- `python3 scripts/check_doc_links.py`: 0 problems.
- Venv check: a bare `python -c "import eval_datasets"` resolves to /workspace/retrieve (shared .pth). Under pytest, the Q2 `pythonpath` fix correctly resolves to the worktree (verified by a probe test).
- `docs/validation.md` is untouched: no gate changed state.

## Findings and what was done
Fixed:
1. **Duplication across ETL modules.** Four copies of the `prep_log.json` merge (kuairand `_merge_prep_log`, openalex/pubmed `_merge_log`, yfcc `_merge_prep_log`) became `common.merge_prep_log`. Two range-spec parsers (pubmed `_parse_shards`, openalex `_parse_files`) became `common.parse_ranges(spec, n)`. Four file-hash helpers (kuairand/pubmed `_md5`, yfcc `_sha256`, goodreads `sha256_of`) became `common.file_hexdigest(path, algo)`; `hashlib.file_digest` is not usable because requires-python is >=3.10.
2. **openalex imported `pmid_hash`/`select_pmids` from pubmed**, a coupling between two ETL modules. Both functions moved to `common.py` with their names unchanged. `test_pubmed` and the datasets.md references were updated.
3. **`common.py` imported torch at module level.** That silently defeated the per-file PLC0415 rationale (keep torch out of --help/download/plan) for kuairand, pubmed, arxiv and goodreads. `synthesize_qa_narrow` now uses `narrow_t.new_full(...).long()`, and torch is imported only under TYPE_CHECKING. Verified: those four no longer load torch at import. pubmed's `layout.atomic_write` import stays lazy for the same reason.
4. **Dead code.** `common.dense_remap_ids` had no callers and was deleted; its mention in datasets.md was fixed.
5. **Nested defs.** pubmed `_fetch_one._log` → `_log_line`, `cmd_medline._one` → `_medline_one` (via partial), `cmd_convert._submit` → `_submit_fetches` (returns futures). kuairand `main`'s argparse helpers moved to module level. bench/cli `say` → `_say`, bench/upload `at` → `_at`, bench/run `_rotate` closures → partials over `_call_next` / `_call_next_filtered`. timesplit `drop` → `_drop_non_train`.
6. **`assert` on caller input → ValueError or error return.** bench/algos (2 sites), timesplit (3 asserts), kuairand cmd_attrs (now logs ERROR and returns 1).
7. **`while True` loops** in pubmed became walrus reads. pubmed `_medline_worker` now takes typed args instead of a tuple.
8. **Comment slop.** Section banners in the three new modules were removed. openalex's measured-number comment (898/755 docs/s, cosine) now just cites datasets.md § openalex, where the numbers already live. A stray `#` in timesplit was removed.
9. **Type hints.** `cmd_*(args: argparse.Namespace)` in kuairand, openalex and pubmed, plus openalex's `_load_encoder`/`_encode`/`_encode_meta`/`_add_*_args` and kuairand's `csv_to_parquet(src: Path | IO[bytes])`.

Deliberately skipped:
- **eval_datasets/cli.py `_forward`** (a click command defined in a loop, with a `module=module` default-arg hack). I rewrote it with `click.Command` + partial, but `scripts/check_doc_links.py` parses exactly the `@main.command` decorator-in-loop shape via AST and crashed. `scripts/` is outside my scope, so I reverted the change.
- **bench/upload.py `sha256`** duplicates `file_hexdigest`, but bench → eval_datasets.common would be a new edge that tests/test_dependency_direction.py does not allow.
- **Resumable HTTP GET implemented four times** (kuairand `_fetch`, pubmed `_fetch_one`, yfcc `_download_one`, goodreads `download_one`). They differ in HEAD use, retries, sleeps and logging, so unifying them would change behaviour and touch backlog files. Candidate for the evaluation-cleanup backlog.
- **openalex still loads torch at import**, via its top-level `from eval_datasets.layout import atomic_write`. Fixing it needs either inline imports in 4 functions or a torch-free home for `atomic_write`. Left to the orchestrator.
- **Long functions:** openalex `cmd_prep` (~150 lines), pubmed `cmd_attrs` (~220) and `cmd_convert` (~180). Splitting them would be a rewrite. Consider only.
- **`# fmt: skip` hand-formatting** in openalex/pubmed: cosmetic, left alone.
- **bench/run.py asserts** (`inp`/`assets`/`filter_mod is not None`): internal type narrowing, not input validation. bench/oracle.py and run.py `except Exception`: marked recording boundaries (H §7, torn-cache rebuild). layout `except BaseException`: re-raises. All acceptable.
- **training/**: nothing substantive; its comments explain why. train.py is on the format backlog.
- Deferred and untouched as instructed: the SUITES bug in bench/cli.py, report.py provenance wording, goodreads' broad excepts, the 15 format failures.

## Small behaviour notes (all intended)
- pubmed `verify`'s md5 message is now "md5 mismatch against <file>.md5" instead of printing both digests.
- `parse_ranges` also tolerates whitespace and empty parts for openalex `--files`, as pubmed's parser already did.
- timesplit and algos now raise ValueError where they used to assert. Nothing changes except under `python -O`, where they previously did not check at all.
