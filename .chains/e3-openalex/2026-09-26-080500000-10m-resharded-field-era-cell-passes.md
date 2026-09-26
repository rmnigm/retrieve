---
chain: "e3-openalex"
branch: "main"
parent: "2026-09-26-075500000-encoded-bench-check-ok-cell-ooms-in-oracle.md"
nextStep: "Orchestrator: merge dev/e3-openalex (aa93860) into staging; fix docs/system/evaluation.md:255 (openalex.yaml is now in the filter suite); update validation.md's datasets row with the 10M cell; decide whether to delete /data/openalex-15m (34 GB) and /data/_raw/openalex/staging (16 GB); remove the stray /run10m-encode-queries-attrs.log."
created: "2026-09-26T08:05:00Z"
---

# E3 OpenAlex: 10M catalog resharded from the 15M encode; the field_era cell passes

## Done (dev/e3-openalex, commit aa93860; not pushed)

Option A, as the user chose: cut the catalog to 10M and reuse the vectors.

- `/data/openalex` was renamed to `/data/openalex-15m`, then `prep --keep-items 10000000`
  ran: 374 s, 10M items, a pool of 19.9M, 4.89M eligible queries, 10,000 drawn, 35,673 qrels.
  - `verify_prep.py`: 0 of 10,000 mismatched.
- New subcommand **`openalex reshard --from-dir <larger>`**. It gathers the fp16 vectors by
  work id, and refuses to run if the new catalog is not a subset of the source.
  - Its `encode_params.json` matches what `encode_text` would compute for this directory, so
    a later `encode_text` does not re-encode.
  - It shares `_write_index` with `encode_text`.
  - Took 34 s. CPU test added (`test_reshard_gathers_a_smaller_catalogs_vectors_by_work_id`).
  - `check_reshard.py` (join on work_id, independent of reshard's searchsorted): all 10M
    rows are `torch.equal`, 0 missing.
- `encode_queries` + `attrs` ran with the worktree code. The target passes field and era
  100 %, subfield 75 %.
- `openalex` is in `suites.yaml`'s filter suite for good.
  - `config/openalex.yaml` header updated.
  - `bench check`: `openalex d768: ok`.
- **Cell** (`linr_v1_filter_mask`/triton, clause `field_era`, eager, `--skip-perf`, so
  `partial`):
  - `recall_oracle@1000` **0.9959**; pass rate 0.0526.
  - Held-out (a cited paper): `recall@100` 0.505, `recall@1000` 0.741; n = 10,000.
  - Reserved memory 64 GB; the oracle build took 27 s.
  - Record: `docs/artifacts/e3-openalex/filter-openalex-d768-cell.jsonl`.
- The harness suite: 238 passed, 1 skipped. `ruff check` and the link checker are clean.
  The 9 files `ruff format` flags under `evaluation/` all predate this step.
- Docs: `datasets.md` § openalex (status, 10M prep, budget, "what fits"),
  `storage.md` E3 bullet, `run.sh`.

## Incident
- A mis-chained shell command (`&` backgrounded the `A=` assignment) ran `encode_queries` +
  `attrs` once without `PYTHONPATH` and wrote a stray log at `/run10m-encode-queries-attrs.log`.
  - Both were re-run correctly afterwards; the outputs are identical.
  - The user denied my `rm` of the stray file, so it is **still at `/`** for the user to
    remove.

## Left for the orchestrator
- `docs/system/evaluation.md:255` lists `openalex.yaml` as "in no suite yet". It is now in
  `filter`. Not edited: the file was off-limits in my brief.
- `validation.md` datasets row: E3 OpenAlex staged locally at 10M (not on the Hub),
  `bench check` passes, `field_era` oracle built (pass rate 0.0526). One cell,
  `linr_v1_filter_mask`/triton: `recall_oracle@1000` 0.9959, held-out `recall@100` 0.505,
  n = 10,000.
- Disk is at 73 % of the overlay (56 GB free). Deletable, but only on the user's call:
  - `/data/openalex-15m` (34 GB): redundant now that the 10M vectors are verified, but it
    cost 4.35 GPU-h to produce.
  - `/data/_raw/openalex/staging` (16 GB): only needed to re-prep another N.
- SilverTorch official and the other algorithms have not run on openalex. Triton
  SilverTorch and LiNR V2/V3 cannot run at D=768 (the known power-of-two limit).
