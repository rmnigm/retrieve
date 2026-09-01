# Datasets and training

Everything upstream of the benchmark: how the four datasets are fetched
and reshaped into the on-disk layout the harness expects, and how the
SASRec checkpoints that produce query embeddings are trained.

For what happens *after* — the sweep driver, measurement methodology,
output schema — see [evaluation.md](evaluation.md). For the trained
checkpoint inventory and the HF Hub workflow, see
[checkpoints.md](checkpoints.md).

## The three shapes of a dataset

The harness reads two different dataset shapes, and which one a config
gets is decided by whether it sets `checkpoint`:

| shape | datasets | query embeddings come from | filters |
|---|---|---|---|
| **sequential** | yambda-500m, yambda-5b, goodreads | a trained SASRec checkpoint, encoded at eval time | goodreads only |
| **text** | arxiv, arxiv-synth | pre-encoded text embeddings on disk | yes |

A third variant, **synthetic**, is a text dataset grown to arbitrary `N`
by interpolating between real embeddings — used for scale sweeps where a
real catalog that size doesn't exist.

Data lives under `$RETRIEVE_DATA_ROOT` (default `<repo>/data`). Raw
downloads go to `data/_raw/<dataset>/`; bench-side outputs to
`data/<dataset>/<variant>/`.

## `eval_datasets/` — the ETL package

One module per dataset, each an `argparse` CLI exposed as a console
script. The package is named `eval_datasets` rather than `datasets`
because the latter shadows HuggingFace's `datasets` in the shared venv.

| module | console script | source |
|---|---|---|
| [`yambda.py`](../../evaluation/eval_datasets/yambda.py) | `yambda` | HF `yandex/yambda`, Listen+ branch |
| [`goodreads.py`](../../evaluation/eval_datasets/goodreads.py) | `goodreads` | UCSD Book Graph mirror (HTTPS) |
| [`arxiv.py`](../../evaluation/eval_datasets/arxiv.py) | `arxiv` | HF `open-index/open-arxiv` (~2.99M papers) |
| [`synth_arxiv.py`](../../evaluation/eval_datasets/synth_arxiv.py) | — (run as a module) | an already-encoded arxiv directory |
| [`hf_io.py`](../../evaluation/eval_datasets/hf_io.py) | `eval-fetch`, `eval-publish`, `eval-publish-checkpoint` | HF Hub push/pull |
| [`common.py`](../../evaluation/eval_datasets/common.py) | — | shared attribute-synthesis numerics |
| [`timesplit.py`](../../evaluation/eval_datasets/timesplit.py) | — | vendored sequential time-split |

`synth_arxiv` has **no** console script despite its docstring examples —
invoke it as `uv run python -m eval_datasets.synth_arxiv`.

### Shared conventions

- **Item ids are 1-indexed dense ints** in `item_id_map.json`, because
  id `0` is the sequence-padding token on the trainer side. The harness
  drops the padding row when loading (`item_embs[1:]`) and shifts target
  ids by −1, so the `retrieve` library is 0-indexed over real items
  throughout. This is why no filter mask needs an "item 0" fixup.
- **Attribute tensors are 0-indexed dense**: row `i` of
  `item_attrs_narrow.pt` describes `item_id i+1`.
- Every subcommand writes a `prep_log.json` with row counts and
  filtering statistics next to its outputs.
- Subcommands are individually re-runnable; `all` chains them.

### yambda

`uv run yambda prep --variant {500m,5b} --output-dir data/yambda/<v>`

Downloads `<variant>/sequential/listens.parquet`, runs `preprocess()`
(Listen+ branch: `played_ratio ≥ 50%`), and writes the four artifacts the
trainer consumes:

```
<output>/train.parquet      item_ids: list[int64]
<output>/val.parquet        item_ids, targets
<output>/test.parquet       item_ids, targets
<output>/item_id_map.json   {raw_yandex_id: dense_int}
```

Validation history is the train portion (already sliced to the last 200
items in `preprocess`); test history is train ++ val, last 200. Adapted
from the Yambda paper's reference `sasrec/data.py`; only the
listens-Listen+ branch is kept, the rest is replaced by
`evaluation.training`.

### goodreads

`download` → `convert` → `prep` → `attrs`.

`download` mirrors the UCSD top-level files over HTTPS (resumable via
`Range`, sha256 manifest); `convert` streams them into ZSTD parquet.

**`prep`** turns the staged parquets into yambda-shaped trainer inputs:
filters `is_read=true`, parses `date_added`, collapses editions to a
`work_id` catalog, runs an iterative n-core, then time-splits via
`timesplit.sequential_split_train_val_test`. Outputs `train/val/test.parquet`,
`item_id_map.json`, `book_to_work.parquet` (reused by the filter eval),
and `prep_log.json`.

> `prep` deliberately passes `drop_non_train_items=False`, mirroring
> `yambda.preprocess`. Setting it `True` makes polars re-evaluate an
> imploded set per row and blows the job past a 129 GB cgroup limit.

**`attrs`** builds the filter-bench tensors from `prep`'s outputs plus
the catalog parquets:

```
item_attrs_narrow.pt        [N, 5, 4] int64   (genre, lang, format, year, author)
item_attrs_wide.pt          [N, 1, 32] int64  (shelves — see "dead artifacts")
clause_is_reverse_narrow.pt [5] bool = [F, T, F, F, F]
lang_vocab.json / format_vocab.json / author_vocab.json / wide_shelf_vocab.json
wide_shelf_global_freq.pt   [V_wide] int64
eval_split.parquet          aligned 1:1 with test.parquet rows
```

Clause 1 (language) is the **reverse** clause — the one sweep that
exercises negated predicates end-to-end. It is bloom-incompatible, so
goodreads bloom sweeps exclude it.

### arxiv

`download` → `convert` → `prep` → `encode_text` → `encode_queries` →
`attrs`, or `uv run arxiv all --output-dir data/arxiv/papers`.

Arxiv has **no user sequences**, so `prep` does no interactions and no
time-split — it emits `item_id_map.json`, `papers.parquet`, and a
sampled `heldout.parquet`. Held-out items stay in the index.

The two encode steps use the same model (`nomic-embed-text-v1.5`,
Matryoshka-truncated) with **different prefixes** — `"search_document: "`
for items, `"search_query: "` for queries. That asymmetry is
nomic-specific and load-bearing: it is what makes the unfiltered
cross-check sweep meaningful rather than an identity lookup. The loader
asserts the prefix recorded in each `*.meta.json` sidecar, so a
prefix mismatch fails at load instead of silently producing garbage
recall.

Embeddings are written per dimension:

```
<output>/content/       text_emb.pt + query_emb.pt at D=256
<output>/content_d128/  same, Matryoshka-truncated to 128
<output>/content_d64/   same, truncated to 64
```

Configs select one via `content_subdir`.

### synth_arxiv

Grows an encoded arxiv directory to arbitrary `N` (15M / 30M / 50M) by
spherical k-means clustering the source embeddings (`cluster`, cached as
`clusters.pt` and reused across target sizes) and then SLERPing between
same-cluster parents (`synth`). Recall is measured against an exact-kNN
oracle over the *same* synthetic catalog, so synthesis needs no
anti-shadowing guarantees.

Output is a drop-in for the normal arxiv loader, except the embeddings
are sharded (`text_emb_shard_*.pt` + `shard_index.json`) and reassembled
at load. Vocabs, queries, held-out rows, and `eval_split` are copied
verbatim; wide attributes are not synthesized (the arxiv bench is
narrow-only).

There is a hard cap on `N`: above it the attribute tensors exceed
comfortable host RAM during construction.

### On-disk layout the harness expects

Sequential datasets:

```
data/<dataset>/<variant>/
├── item_id_map.json
├── train.parquet / val.parquet / test.parquet
└── (checkpoints live in the checkpoint dir, not here)
```

Text datasets:

```
data/arxiv/papers/
├── item_id_map.json
├── papers.parquet
├── heldout.parquet
└── content{,_d128,_d64}/
    ├── text_emb.pt + text_emb.meta.json
    └── query_emb.pt + query_emb.meta.json
```

Filter sweeps additionally need:

```
├── item_attrs_narrow.pt
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
└── ... per-clause vocab JSONs
```

### Dead artifacts

`item_attrs_wide.pt`, `wide_shelf_vocab.json`,
`wide_shelf_global_freq.pt`, and the `query_attrs_wide_1shelf` /
`_2shelf` columns of `eval_split.parquet` are **built but never read**:
`load_filter_assets` does not load them and `load_query_attrs` skips the
wide columns explicitly. They exist for a wide-bloom sweep that was
never run. Keep or drop them as a unit — they are only meaningful
together.

## HuggingFace I/O

[`hf_io.py`](../../evaluation/eval_datasets/hf_io.py) holds the repo
registry and the fetch/publish helpers, so no path or repo id is
hard-coded in the ETL modules:

```python
EVAL_REPOS = {
    "arxiv-papers":      "pinkmeme/eval-arxiv-papers",
    "yambda-500m":       "pinkmeme/eval-yambda-500m",
    "yambda-5b":         "pinkmeme/eval-yambda-5b",
    "goodreads-work-id": "pinkmeme/eval-goodreads-work-id",
}
```

`eval-fetch` pulls a prepared dataset (optionally a subset of dims),
`eval-publish` pushes one, `eval-publish-checkpoint` pushes a trained
checkpoint under `<repo>/checkpoints/<ckpt-id>/`. Downloading a prepared
dataset is the fast path — the full arxiv encode is hours of GPU time.

## Training — `evaluation/training/`

A compact gSASRec trainer. It exists to produce the item embeddings and
query encoder the sequential benchmarks need; it is not a research
surface of its own.

| file | role |
|---|---|
| [`config.py`](../../evaluation/training/config.py) | `GSASRecConfig` dataclass + `save` / `load` / `num_items` |
| [`model.py`](../../evaluation/training/model.py) | `GSASRec` — `nn.TransformerEncoder`, `norm_first`, separate output embedding, `predict_last` |
| [`dataset.py`](../../evaluation/training/dataset.py) | `SequenceDataset`, negative-sampling collate, train dataloader |
| [`losses.py`](../../evaluation/training/losses.py) | `gbce_loss` |
| [`evaluate.py`](../../evaluation/training/evaluate.py) | chunked full-catalog scoring + metrics (shares `retrieval.metrics`) |
| [`train_sasrec.py`](../../evaluation/training/train_sasrec.py) | `train()` + the click CLI |
| [`upload_checkpoints.py`](../../evaluation/training/upload_checkpoints.py) | push a checkpoint dir to HF |

### Running a training job

```bash
uv run --directory evaluation python -m training.train_sasrec \
    --data-dir data/yambda/500m-listens \
    --checkpoint-dir checkpoints/yambda-500m-d128 \
    --embedding-dim 128 --dropout 0.5
```

Every `GSASRecConfig` field has a matching flag; alternatively pass
`--config <json>` to load a saved config verbatim. `--resume` picks up
from `<ckpt_dir>/_resume.pt`, which is written every epoch and carries
model, optimizer, epoch, best metric, and the Python / numpy / torch /
CUDA RNG states — so a resumed run is reproducible, not merely
restarted.

### The loss

`gbce_loss` implements gBCE: uniform negatives per positive, with the
positive logit passed through a calibration transform parameterized by
`gbce_t` (`0.75` default). The transform runs in **float64** — the
`pow(-beta)` and `1/(x-1)` steps lose the positive class entirely in
fp32. The rest of the step runs under bf16 autocast.

### Loop shape

Per epoch: bf16 autocast forward → `gbce_loss` → grad-clip at 1.0 →
AdamW (fused on CUDA). Every `eval_every` epochs it evaluates on
`val.parquet` by scoring the full catalog in `[B, eval_score_chunk]`
chunks and merging top-k across chunks, optionally masking history.
Checkpoints on improvement in `early_stop_metric` (`ndcg@10`), stops
after `patience` epochs without one.

TF32 is **enabled** here (`_enable_tf32`) — the opposite of the
benchmark harness, which pins it off for measurement determinism.
Training throughput matters more than bit-reproducibility of a matmul.

### What a finished run leaves behind

```
<checkpoint_dir>/
├── best_model.pt          # reloaded, then re-saved from the best epoch
├── config.json            # the GSASRecConfig — the eval loader reads this
├── item_embs.pt           # output embedding matrix, row 0 zeroed
├── item_id_map.json       # copied from data_dir
├── item_attrs.parquet     # copied from data_dir when present
├── eval_quality.json      # test-split metrics
├── train_metrics.json
├── gsasrec-ep{N}-{metric}{v}.pt   # per-improvement snapshots
└── _resume.pt
```

`config.json` is what makes a checkpoint self-describing at eval time.
Checkpoints trained before it was written fall back to
`D128_DROP05_DEFAULTS` in
[`encode.py`](../../evaluation/retrieval/encode.py) — see
[checkpoints.md](checkpoints.md) for which runs those are.

## Adding a dataset

1. Write `eval_datasets/<name>.py` with `download` / `convert` / `prep`
   subcommands emitting the layout above; register a console script in
   [`evaluation/pyproject.toml`](../../evaluation/pyproject.toml).
2. If it has attributes, add an `attrs` subcommand producing
   `item_attrs_narrow.pt`, `clause_is_reverse_narrow.pt`, the vocab
   JSONs, and `eval_split.parquet` aligned 1:1 with `test.parquet`.
3. Add an entry to `EVAL_REPOS` in `hf_io.py`.
4. Add configs under `evaluation/config/<name>/`. Set `checkpoint` for
   the sequential shape, leave it unset for the text shape.
5. Train a checkpoint if sequential; otherwise encode embeddings into
   `content*/` with meta sidecars.

Reuse `common.py` for attribute synthesis (`dense_remap_ids`,
`synthesize_qa_narrow`, `sample_rare_biased_wide`) rather than
re-deriving the sampling — the rare-biased draw in particular is what
keeps filter selectivity in an interesting range.
