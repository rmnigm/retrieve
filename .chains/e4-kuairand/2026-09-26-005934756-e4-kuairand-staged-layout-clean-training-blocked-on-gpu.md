---
chain: "e4-kuairand"
branch: "main"
nextStep: "Orchestrator: merge dev/e4-kuairand (commit da381be) into staging, apply the hub.py EVAL_REPOS line below plus the two docs/system/evaluation.md edits, then — once the GPU is free — run the gSASRec training command below, publish the dataset + checkpoint, add kuairand to the filter suite and run one filter cell."
created: "2026-09-26T00:59:34Z"
---

# E4 KuaiRand-27K: staged, layout clean, training blocked on the GPU

## 1. Request

Worker brief for roadmap E4: stage KuaiRand-27K and train gSASRec over its 32 M
videos, with a shared item table (`reuse_item_embeddings`, the orchestrator's
call), `[N, 7, 4]` attrs, two filter protocols (target-derived and business
rule) and `config/kuairand.yaml`. Do not touch hub.py, retrieve/, bench/,
docs/system/evaluation.md or testing.md, docs/validation.md or the roadmap.
Commit locally and do not push.

## 3. Work completed (branch `dev/e4-kuairand`, worktree `/scratch/wt/e4`, commit `da381be`)

- `evaluation/eval_datasets/etl/kuairand.py`: `download` / `convert` / `prep`
  / `attrs` / `all`, registered in `eval_datasets/cli.py` `ETL`.
- `evaluation/config/kuairand.yaml`: dims `[128]`; checkpoint
  `data/kuairand/checkpoints/gsasrec-d{dim}-shared/best_model.pt`; 10 clause
  sweeps (`t_*` over C0–C3, `b_*` over C4–C6) and 4 forward-only bloom
  sweeps. Through a scratch copy of suites.yaml it expands to 70 jobs / 98
  cells in `filter`.
- `evaluation/tests/eval_datasets/test_kuairand.py`: 9 CPU tests on a tiny
  tarball in the upstream member layout. Every value is hand-derived, and
  `validate_layout == []` is asserted. Mutation-checked: dropping the
  `is_rand` filter, dropping the reverse flip, and the bucket off-by-one
  each turn it red.
- `evaluation/pyproject.toml`: `pythonpath = [".", "../retrieve/src"]`,
  byte-identical to Q2's hunk.
- Docs: a `docs/system/datasets.md` § kuairand section, plus the shapes
  table, module table and EVAL_REPOS listing; a `docs/system/storage.md`
  E4 bullet with measured disk; `docs/artifacts/e4-kuairand/` (the profiling
  script and its output, `prep_log.json`, `convert_log.json`,
  `attr_vocab.json`, the convert/prep/attrs run logs).
- **Real data staged** at `$RETRIEVE_DATA_ROOT=/data`:
  - raw `/data/_raw/kuairand/raw/`, 13.6 GB, both md5-verified by `download`;
  - processed `/data/_raw/kuairand/processed/`, 4.5 GB;
  - bench layout `/data/kuairand/`, 8.3 GB.
- `bench check --dataset kuairand` gives `kuairand d128: ok`.

## hub.py — apply at merge (verbatim)

In `evaluation/eval_datasets/hub.py`, `EVAL_REPOS`, after the `"pubmed"` line:

```python
    "kuairand":          "pinkmeme/eval-kuairand",
```

No `RAW_REPOS` entry is needed: `raw_dir("kuairand")` works without one, like
yfcc and pubmed, and the source is Zenodo, not the Hub.
`docs/system/datasets.md` § HuggingFace I/O already lists this entry as
registered and not published, so the doc and the code agree once the line
lands.

## docs/system/evaluation.md — two stale copies (not my file)

- § Config: "Seven files under `evaluation/config/`" should read eight, with
  `kuairand.yaml` in the list (in no suite yet).
- § Tests table (the line with `test_yfcc.py`, `test_pubmed.py`): add
  `test_kuairand.py`; "the two ETL loaders" becomes three.

## 4. Decisions (with evidence)

- **Positives are `is_click==1 & is_rand==0`.** In the data, `is_rand` is 0 on
  all 322,278,385 standard-log rows and 1 on all 1,186,059 random-log rows,
  so the filter drops exactly the random log. 122,052,542 clicks remain.
- **`final_video_id` is the 27K `video_id`.** The supplement is dense
  `0..32,038,724`, and its level-1 category equals the basic features' first
  `tag` on 64.5 % of videos.
- **The catalog is all 32,038,725 videos** (identity map, `item_id = video_id
  + 1`), with no n-core; 12.06 M are clicked in train. The survey's "5-core
  subset" alternative was not taken, because the shared table covers
  everything.
- **Split.** Shanghai local dates, a fixed UTC+8 (the container has no tzdata;
  China has had no DST since 1991): test 05-07..05-08, val 05-06, 30 min
  gaps. Test is one row per user, 26,221 rows.
- **Train rows are 200-transition windows** (561,486 rows, 109.6 M
  transitions), because `SequenceDataset` only reads a row's last 201 items.
  This is a prep-side choice; no trainer change. The survey's
  "multi-cut-point eval" was not needed, since 26,221 test rows exceed
  `users_limit: 10000`.
- **Clauses.** C0 cat1, C1 fine category (levels 4→3→2, deepest first), C2
  tag and C3 upload_type are target-derived. C4 video_type is REVERSE and
  business ("no ads"). C5 (duration ≤ {15, 30, 60, 180} s) and C6 (uploaded
  ≤ {3, 7, 14, 30} d) are cumulative-threshold clauses: range predicates
  written as equality. The protocols need disjoint slots because the harness
  reads one `query_attrs_narrow`.
- **Rejected attributes.** Upload month: 90 % of uploads fall within 42 days.
  Author as reverse: 8.8 M authors, median 1 video, so it passes about 100 %.
  `music_type`: 65 % one value.
- **Exact mean pass rates over all 26,221 queries:** C0 3.6 %, C1 1.0 %,
  C2 4.5 %, C3 14.2 %, C4 99.9 %, C5 52.9 %, C6 25.8 %.

## 6. Blocked / unverified

- **GPU-blocked, not run:**
  - gSASRec training;
  - `eval-data publish kuairand` and `publish-checkpoint`;
  - the filter cell.
- **Training memory is arithmetic, not a measurement.** The table is
  32,038,726 × 128 fp32 = 16.4 GB. With its dense gradient and two AdamW
  moments that is 65.6 GB before activations. The survey's ~49 GB left out
  the gradient. If it OOMs, lower `--negs-per-pos` or `--batch-size`.
- **The C4 "no ads" clause is nearly a no-op** (99.9 % pass): real, but it
  only exercises the reverse path.
- **Pre-existing, not mine:** `tests/eval_datasets/test_yfcc.py::TestRealSlice`
  fails on this box because `/data/_raw/yfcc10m/*.spmat` is absent while
  `/data/yfcc10m` exists.

## 7. Pending (GPU, in order)

```bash
uv run --directory evaluation train sasrec --data-dir data/kuairand \
    --checkpoint-dir data/kuairand/checkpoints/gsasrec-d128-shared \
    --embedding-dim 128 --dropout 0.5 --reuse-item-embeddings --eval-max-users 4096
uv run --directory evaluation eval-data publish kuairand            # after the hub.py line
uv run --directory evaluation train upload-checkpoint ...           # ckpt id gsasrec-d128-shared
# add kuairand to suites.yaml filter.datasets, then one cell:
uv run --directory evaluation bench run --dataset kuairand --suite filter \
    --algo linr_v1_filter_mask --filter-kind clause --sweep t_cat1 --k 100 --bs 1
```

In a worktree, the console scripts resolve to `/workspace/retrieve` through
the venv's `.pth`. Run from the merged checkout, or run
`python -m`/`python -c` from `evaluation/` with the cwd first on `sys.path`.
