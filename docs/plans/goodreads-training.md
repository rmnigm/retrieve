# Goodreads gSASRec training

## Why this exists

We need a filter-bench dataset to round out the unified
[filtering API](./filtering-api.md) story. Goodreads UCSD
adds a sequential-recsys flavour at comparable catalog scale (~1.5M
items dedup-by-`work_id`), with both predicate shapes natively present
(see [goodreads-filter-eval.md](./goodreads-filter-eval.md) for the
filter side). This doc covers only the **training** side: data prep
and three gSASRec checkpoints at embedding dims `64 / 128 / 256`.
gSASRec embeddings are the input to every retrieval algorithm in the
filter bench (`query @ item_embs.T`); training itself is filter-agnostic
— the trained model is a black box for the filter eval, accessed only
via `predict_last(item_seq)` and `get_output_embeddings().weight`.

The `D = {64, 128, 256}` sweep mirrors the yambda 500M setup
([500m-d128.yaml](../../evaluation/conf/500m-d128.yaml),
[checkpoints.md table](../system/checkpoints.md#available-checkpoints)),
where d64 won every quality metric at the 100-epoch budget cap and
d256 over-capacity hurt R@100. We expect the same shape on Goodreads
and want the bench to reproduce it across `D`.

User-locked decisions:
- catalog granularity = **`work_id`** (~1.52M after edition collapse)
- training interactions = **`is_read=true` only**

## Inputs (Stage 1 already done)

Stage 1 produced the input parquets via
[goodreads.py](../../evaluation/data/goodreads.py)
(download → ZSTD parquet → external sort by `(user_id, book_id)`).
Their location is **caller-supplied** through `--processed-dir`
(the existing `goodreads.py` script defaults to
`~/datasets/goodreads-ucsd/processed/`, but the new
`goodreads.py prep` script is path-agnostic and the doc does not
assume that layout). Files expected in `<processed-dir>/`:

- `goodreads_interactions_dedup.parquet` — 228.6M shelf events;
  carries `(user_id, book_id, is_read, rating, date_added,
  date_updated, read_at, started_at, is_reviewed, review_id)`.
  Already sorted by `(user_id, book_id)` — must be re-sorted by
  `(user_id, date_added)` at sequence-build time.
- `goodreads_books.parquet` — book metadata; needed only for the
  `book_id → work_id` map (1 column join).
- `goodreads_book_works.parquet` — authoritative work IDs;
  cross-check the books-side mapping.

## Outputs

```
data/goodreads/work-id/
├── train.parquet                       # (uid: int64, item_ids: list[int64])
├── val.parquet                         # (uid, item_ids, targets: list[int64])
├── test.parquet                        # (uid, item_ids, targets)
├── item_id_map.json                    # work_id (str) → 1-indexed dense int
├── book_to_work.parquet                # (book_id, work_id) — reused by
│                                       #   the filter-eval doc
└── prep_log.json                       # n-core stats, parse failure %,
                                        #   per-split row counts

checkpoints/
├── goodreads-gsasrec-d64-drop0.5/      # best_model.pt + eval_quality.json
├── goodreads-gsasrec-d128-drop0.5/
└── goodreads-gsasrec-d256-drop0.5/
```

`train/val/test.parquet` schema matches yambda's exactly so
[SequenceDataset](../../evaluation/training/dataset.py#L9-L31) and
[EvalDataset](../../evaluation/training/evaluate.py) work unmodified.
Item index `0` is reserved for padding.

## Stage 1.5 — preprocessing (`evaluation/data/goodreads.py`, new)

Mirrors [evaluation/data/yambda.py](../../evaluation/data/yambda.py)'s
shape — single click-CLI script, polars LazyFrame all the way through,
streaming engine.

```
uv run python -m data.goodreads prep \
  --processed-dir <path-to-goodreads-output> \
  --output-dir data/goodreads/work-id \
  --n-core 5 \
  --max-seq-len 200
```

`--processed-dir` is required (no default); it must point at the
directory that `goodreads.py convert` wrote its parquets into.
`--output-dir` is the under-repo location the trainer will read from
(see `data_dir` in the configs below).

Steps (one polars pipeline, no intermediates):

1. **Scan interactions** lazily, projecting `(user_id, book_id,
   is_read, date_added)` only (drops 60 % of bytes).
2. **Filter** `is_read == true`. Drop the (small) rows missing
   `date_added`.
3. **Parse `date_added`** from the legacy `Day Mon DD HH:MM:SS ±zzzz YYYY`
   format → `pl.Int64` unix-seconds. Drop rows with parse failure or
   year < 2007 (the dataset's known sentinel range, e.g. the
   `1988-01-01` `read_at` placeholder). Track failure rate in
   `prep_log.json`; expect < 1 %.
4. **Join `book_id → work_id`** from `goodreads_books.parquet` —
   left join, drop the < 0.1 % unmatched rows. Persist the join
   table as `book_to_work.parquet` (filter-eval reuses it for attr
   extraction).
5. **5-core iterative filter** on `(user_id, work_id)` after the
   collapse. Alternate `count(user_id) ≥ 5` and `count(work_id) ≥ 5`
   until stable; cap at 10 rounds as a safety net. Expected
   post-core: ≈ 600 k users × 800 k works × 90 M interactions
   (extrapolating from per-genre 5-core numbers in the dataset
   description). Record the iteration trace in `prep_log.json`.
6. **Dense `work_id → idx` map**, 1-indexed (idx 0 = padding).
   Persist as `item_id_map.json` — consumed by
   [GSASRecConfig.num_items](../../evaluation/training/config.py#L48-L51).
7. **Group + sequence**: per `user_id`, sort by parsed timestamp
   ascending, slice tail to `max_seq_len + 1 = 201` items. Apply
   [`sequential_split_train_val_test`](../../evaluation/data/timesplit.py#L101-L214)
   directly — same time-split machinery yambda uses. Choose
   `test_timestamp` so that approximately the last 60–90 days are
   test (≈ 2017-09-01 to 2017-11-30, the dataset's tail), and
   `val_size = 30 days`, `gap_size = 7 days`.
   `drop_non_train_items=True` — items absent from train get pruned
   from val/test sequences (matches yambda).
8. **Write parquets**: `train.parquet` (sequence-only),
   `val.parquet` and `test.parquet` (sequence + held-out targets).
   Write `prep_log.json` with N_users / N_items / N_interactions /
   median_seq_len / p99_seq_len / parse-fail % / per-split row count.

The CLI also accepts a `--smoke` flag that subsamples to 100 k users
for fast iteration; the smoke directory is `data/goodreads/work-id-smoke/`
and is the target of the smoke checkpoint described below.

### Stage 1.5 verification

- `(N_users, N_items, N_interactions, median_seq_len, p99_seq_len)`
  printed at the end of `prep`. Floors: `N_items ≥ 700k`,
  `median_seq_len ≥ 8`, `parse_fail_pct ≤ 5`.
- `len(train) + len(val) + len(test) == N_users` (one user per row,
  yambda invariant).
- Random sample of 5 train rows: timestamps strictly increasing inside
  each `item_ids` list (sanity-check sort).
- `data/goodreads/work-id-smoke/` runs end-to-end in ≤ 5 min on the
  user's machine before launching the full training.

## Stage 2 — training

Reuses [train_sasrec.py](../../evaluation/training/train_sasrec.py)
and [GSASRecConfig](../../evaluation/training/config.py) unchanged.
Three JSON configs are added under `evaluation/conf/`, mirroring the
yambda d-sweep ([checkpoints.md table](../system/checkpoints.md#available-checkpoints)):

```
evaluation/conf/
├── goodreads-d64-drop0.5.json
├── goodreads-d128-drop0.5.json
└── goodreads-d256-drop0.5.json
```

Common hyperparameters (all three configs share these — copied from
the d64-drop0.5 yambda recipe that won every quality metric per
[checkpoints.md L17](../system/checkpoints.md#L17)):

| field | value |
|---|---|
| `data_dir` | `data/goodreads/work-id` |
| `max_seq_length` | 200 |
| `num_blocks` | 2 |
| `num_heads` | 2 |
| `dropout` | 0.5 |
| `reuse_item_embeddings` | false |
| `negs_per_pos` | 256 |
| `gbce_t` | 0.75 |
| `batch_size` | 256 |
| `learning_rate` | 1e-3 |
| `weight_decay` | 0 |
| `num_epochs` | 100 |
| `patience` | 15 |
| `eval_batch_size` | 512 |
| `eval_ks` | `[10, 100]` |
| `eval_every` | 1 |
| `mask_history` | false (item-rec, not novelty-only — matches yambda Listen+) |
| `early_stop_metric` | `ndcg@10` |
| `wandb_project` | `goodreads-gsasrec` |

Per-config differences:

| config | `embedding_dim` | `ffn_hidden_dim` | `checkpoint_dir` | `wandb_run_name` |
|---|---|---|---|---|
| d64 | 64 | 256 | `checkpoints/goodreads-gsasrec-d64-drop0.5` | `goodreads-d64-drop0.5` |
| d128 | 128 | 512 | `checkpoints/goodreads-gsasrec-d128-drop0.5` | `goodreads-d128-drop0.5` |
| d256 | 256 | 1024 | `checkpoints/goodreads-gsasrec-d256-drop0.5` | `goodreads-d256-drop0.5` |

Launch (per dim):

```
uv run python -m training.train_sasrec --config conf/goodreads-d128-drop0.5.json
```

Wall clock estimate (single A100 / 4090, bf16 + fused AdamW per the
[yambda d64 recipe](../system/checkpoints.md#L31)): `~6–10 hr / dim`
(90M interactions vs 500M for yambda → roughly 1/5 the per-epoch
cost; 100 epochs × ~5 min/epoch). Run in series to keep VRAM
uncontested.

After training, each `checkpoints/goodreads-gsasrec-d{D}-drop0.5/`
contains:
- `best_model.pt` — state_dict at best val NDCG@10 epoch
- `config.json` — copy of the GSASRecConfig used
- `eval_quality.json` — final test metrics (NDCG@10/100, Recall@10/100,
  coverage, best_epoch)

These are the same artefacts as the yambda 500M runs, so the
[checkpoints.md "loading a checkpoint" snippet](../system/checkpoints.md#loading-a-checkpoint)
generalises with `D` swapped.

### Stage 2 verification

- Smoke training on `data/goodreads/work-id-smoke/` (100 k users,
  10 epochs, d128) reaches `ndcg@10 ≥ 0.05` — proves the pipeline
  end-to-end before committing to a full run.
- For each full-run dim: `eval_quality.json` is produced;
  `ndcg@10 ≥ 0.06` (rough floor for full-catalog Goodreads SASRec;
  per-genre subsets are higher). Below that, the data prep is suspect,
  not the model.
- All three runs converge inside 100 epochs (best_epoch < 95) **or**
  the run is restartable with bumped `num_epochs`. The yambda d64
  was still climbing at epoch 99, so this is plausible here too.
- Checkpoint dir size ≈ 2 × `embedding_dim × num_items × 4 bytes`
  (input + output embeddings dominate); ~700 MB / 1.4 GB / 2.8 GB
  for d64 / d128 / d256 respectively.

## Implementation surface

**Create:**
- [evaluation/data/goodreads.py](../../evaluation/data/goodreads.py)
  — `prep` subcommand only at this stage (`attrs` lands with the
  filter-eval doc).
- `evaluation/conf/goodreads-d64-drop0.5.json` /
  `goodreads-d128-drop0.5.json` /
  `goodreads-d256-drop0.5.json` — three training configs.

**Reuse unchanged:**
- [evaluation/training/train_sasrec.py](../../evaluation/training/train_sasrec.py),
  [dataset.py](../../evaluation/training/dataset.py),
  [model.py](../../evaluation/training/model.py),
  [losses.py](../../evaluation/training/losses.py),
  [evaluate.py](../../evaluation/training/evaluate.py) — no diffs.
- [evaluation/data/timesplit.py](../../evaluation/data/timesplit.py)
  — `sequential_split_train_val_test` is the right fit out of the box.
- [evaluation/data/goodreads.py](../../evaluation/data/goodreads.py)
  — Stage 1 already done by the user.

## Out of scope

- Filter attributes / bench harness — see
  [goodreads-filter-eval.md](./goodreads-filter-eval.md).
- Per-genre subset training. Single full-catalog run per dim; sub-
  corpus splits are a follow-up if needed.
- Per-author / fairness analysis. Stays at the metric-floor sanity check.
- Pushing checkpoints to Hugging Face — same flow as
  [checkpoints.md §"Hugging Face Hub"](../system/checkpoints.md#hugging-face-hub),
  done after the filter-eval numbers are in.

## Implementation order

1. `evaluation/data/goodreads.py prep` script and smoke training first.
2. Full d128 (the medium dim — gives a reference point against the
   yambda 500M-d128 numbers).
3. d64 and d256 in parallel if VRAM permits, otherwise serial.
