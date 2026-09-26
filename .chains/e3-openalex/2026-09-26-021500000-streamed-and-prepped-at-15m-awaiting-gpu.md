---
chain: "e3-openalex"
branch: "main"
parent: "2026-09-26-011534836-openalex-streaming-etl-written-and-dry-run.md"
nextStep: "When the orchestrator says the GPU is free: from /scratch/wt/e3/evaluation with RETRIEVE_DATA_ROOT=/data PYTHONPATH=/scratch/wt/e3/evaluation run `python -m eval_datasets.etl.openalex encode_text` (~4.6 h, resumable per 1M shard), then `encode_queries`, then `attrs`, then `bench check --dataset openalex` (needs evaluation/data -> /data symlink in the worktree); record results in datasets.md + chain, commit on dev/e3-openalex."
created: "2026-09-26T02:15:00Z"
---

# E3 OpenAlex: streamed and prepped at 15M; encode waits for the GPU

## 1. Request

The orchestrator merged 9f64057 into staging (629ab58, abda553, which carries the hub.py
registration and the evaluation.md config count). Decisions it passed on:

- Run at ~15M, not 50M.
- Do all the non-GPU work: `convert` + `prep`.
- **Do not start encode_text / encode_queries.** `kopt` holds the GPU.
- Commit on `dev/e3-openalex` and report.

## 2. Done (all on /data; branch dev/e3-openalex fast-forwarded to abda553, then commit 8692906, not pushed)

- **Flag spelling:** `--keep-items` is a `prep` flag. `convert` takes `--sample-rate`.
- **Sample rate 0.35:** 15M × (1 + margin) sits comfortably under 0.35 × eligible. The 15M
  smallest hashes are the same set at any rate above the cut.
- **`convert --sample-rate 0.35 --workers 64`:**
  - All 2,040 files, 297.1 GB read in **572 s** (526 MB/s average), **0 failures**.
  - Staging is 16 GB at `/data/_raw/openalex/staging`, with `params.json` pinned.
  - Filter counts summed over the full scan:
    - 476,196,327 rows in
    - year: 373.1M
    - English: 261.7M
    - type: 154.2M
    - flags: 130.9M
    - hash sample: 45.8M
    - abstract present: **29,944,334 staged**
    - 11 dropped as `abstract_truncated` (predicted ~10).
  - **Eligible is ~85.6M.** `plan`'s row-group sample said 54.6M; type clustering biases the
    sample. Use `plan` only for the byte count, which is exact.
- **`prep --keep-items 15000000`:** 410 s, 64 GB peak RSS.
  - 15,000,000 items; the pool is 14,944,334.
  - 6,481,495 pool papers cite the catalog (26.1M edges); 4,336,859 are eligible queries.
  - **10,000 held out.**
  - Per query: 4.77 relevant, 2.85 same field + earlier era (median 2).
  - Query eras 1/2/3/4 = 1,727 / 2,051 / 2,267 / 3,955.
  - 916 queries have no source, so C3 is inactive for them. 22,170 items have no field.
  - Outputs in `/data/openalex`: `papers.parquet` 9.3 GB, `item_id_map.json` 318 MB, plus
    `heldout.parquet`, `queries.parquet`, `qrels.parquet` (47,654 rows), `prep_log.json`.
- **`verify_prep.py /data/openalex`: 0 of 10,000 queries mismatched.**
- Docs:
  - `datasets.md` § openalex: status and budget now show the measured 15M run, and the stale
    50M table is replaced.
  - `storage.md` E3 bullet: peak disk ~49 GB with staging kept, ~33 GB without.
- Raw logs are in `docs/artifacts/e3-openalex/run15m-*`. `*.log` is gitignored, so they are
  force-added like the other artifact dirs. This also recovered `plan-2026-09-23.log`, which
  the first commit had silently missed.
- `docs/validation.md` and `docs/roadmap.md` are untouched, per the worker rules.

## 3. Not done / waiting
- `encode_text` (15M ÷ 898 docs/s ≈ **4.6 A100-h**, 15 × 1M shards, resumable), then
  `encode_queries`, then `attrs`.
- `bench check`. For the worktree: `ln -s /data evaluation/data`, then
  `PYTHONPATH=.:../retrieve/src python -m bench.cli check --dataset openalex`.
- Then the oracle and cells: E5.
- Disk now: overlay 97 GB used, 104 GB free. The encode adds 23 GB. Staging (16 GB) can be
  deleted once the encode is done, but keep it until attrs and bench check pass.

## 4. For the orchestrator
- Merge 8692906 (docs and artifacts only; no code change since 9f64057).
- Update `validation.md`'s datasets row: "E3 OpenAlex: streamed + prepped at 15M,
  verify_prep clean; not encoded; not citable".
