---
title: checkpoints
created: 2026-09-26
updated: 2026-10-17
type: entity
tags: [training]
sources: [evaluation/training/, evaluation/eval_datasets/hub.py]
---

# Encoder checkpoints

Checkpoints live under each dataset's data dir:
`data/<dataset>/checkpoints/<ckpt-id>/`, under the data root of
[datasets.md](datasets.md) (`RETRIEVE_DATA_ROOT`, default
`evaluation/data/`). The harness configs in
[`evaluation/config/`](../../evaluation/config/) point at these paths
verbatim.

This document is the **inventory** — which checkpoints exist, which one
the harness reads, and how to move them around. For how
the trainer itself works (the loss, the loop, the config, what a finished
run writes out) see [datasets.md](datasets.md#training--evaluationtraining).

## What the harness reads

The sequential dataset (goodreads) is encoded
with checkpoints of the current trainer (gSASRec body, sampled softmax
with logQ, L2-normalized).

| dataset config | ckpt-id | D | where it is |
|---|---|---|---|
| `goodreads.yaml` | `sasrec-ssm-logq-d{dim}` | 64, 128, 256 | `<owner>/eval-goodreads-work-id` `checkpoints/`; in the [public input archive](datasets.md#the-public-input-archive): d128, the one the final run read (`eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128/`) |

Each dir holds `best_model.pt`, `item_embs.pt`, `config.json`,
`item_id_map.json`, `eval_quality.json`, `train_metrics.json` and a short
`README.md`. The one the final run read comes from the public input archive with

```bash
uv run --directory evaluation hf download filtered-retrieval-repro/filtered-retrieval-inputs --repo-type dataset \
    --include 'eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128/*' --local-dir "$RETRIEVE_DATA_ROOT/_inputs"
```

(it lands under `_inputs/eval-goodreads-work-id/`, the directory
[staging](datasets.md#the-public-input-archive) links as `goodreads-work-id`); from a per-dataset repo of
your own it is `eval-data fetch goodreads-work-id --include-checkpoints` ([below](#hugging-face-hub)).

Arxiv has no SASRec checkpoint — its queries come from a pre-encoded text
embedding tensor (see [evaluation.md](evaluation.md)).

## What's in each checkpoint dir

- `sasrec-ep{N}-…pt` — model `state_dict` saved at the best val epoch (`gsasrec-ep{N}-ndcg10{X}.pt` in the older gSASRec checkpoints).
- `best_model.pt` — same `state_dict`, copied at the end of training (or on
  manual stop). Use this one going forward.
- `eval_quality.json` — the trainer's final test evaluation: `split`, `ks`, `mask_history`, `metrics`.

## Loading a checkpoint

`training.encode.load_model_for_eval` builds the `Encoder` from the
sibling `config.json` (or `D128_DROP05_DEFAULTS` when there is none, as for
the older gSASRec checkpoint `gsasrec-d128-drop0.5`) and renames the older `GSASRec` keys
(`encoder.layers.N.` → `blocks.N.`) on load. The older gSASRec checkpoints
were trained with gBCE (per-position negatives), and
their `config.json` has no `loss` key: `TrainConfig.load` reads that as
`loss=gbce`, so `normalize` is off and scores are raw dot products, and it
drops the keys the trainer no longer has (`negs_per_pos`, …).
`D128_DROP05_DEFAULTS` sets `loss=gbce` for the same reason. Checkpoints
from the current trainer carry `loss` and `normalize` in `config.json`; its
default is sampled softmax with logQ and `normalize` on
([datasets.md § The losses](datasets.md#the-losses)).

```python
import json, torch
from pathlib import Path
from training.encode import load_model_for_eval

DATA_DIR = Path("data/goodreads-work-id")
CKPT_DIR = DATA_DIR / "checkpoints/sasrec-ssm-logq-d128"

# num_items comes from the data dir's id map (written by the dataset's ETL `prep`).
with open(DATA_DIR / "item_id_map.json") as f:
    num_items = len(json.load(f))

model = load_model_for_eval(CKPT_DIR / "best_model.pt", num_items, torch.device("cuda"))
```

The `num_items + 1` row count of every embedding tensor reserves index 0 for
padding — never use id 0 for a real item.

## Extracting item embeddings

Two embedding tables exist on the model: the *input* item embedding (used inside
the transformer) and the *output* embedding (used to score candidates).
For retrieval / ranking, always use `scoring_table()`: the output table,
L2-normalized when the run trained with `normalize`.

```python
item_embs = model.scoring_table().detach().cpu()                 # [num_items+1, D]
item_embs[0, :] = 0.0                                            # zero out padding row
torch.save(item_embs, CKPT_DIR / "item_embs.pt")
```

Use this tensor as the index for ANN search, dot-product retrieval, or
clustering. The id mapping (raw id → dense id) is the
`item_id_map.json` the dataset's ETL `prep` subcommand writes alongside the
parquets ([datasets.md](datasets.md)).

## Encoding a user history into a query

`predict_last(item_seq)` runs the transformer with a causal mask and returns
the hidden state at the last non-padding position — that's the SASRec "next
item" query.

```python
# item_seq is a [B, L] LongTensor of dense item ids, left-padded with 0.
# L should be <= model.max_seq_length (200).
with torch.inference_mode():
    query = model.predict_last(item_seq.cuda())   # [B, D]
    scores = query @ item_embs.cuda().T           # [B, num_items+1]
```

For serving, you typically only need `predict_last`; export the encoder once
and feed it the user's recent interaction sequence.

## Running offline evaluation

Reuses [`training/evaluate.py`](../../evaluation/training/evaluate.py) — full-catalog ranking + NDCG/Recall/Coverage
@10/100 against `val.parquet` / `test.parquet`. Defaults to no history masking,
the setting for re-consumption tasks.

```python
from training.evaluate import evaluate

metrics = evaluate(
    model,
    str(DATA_DIR / "test.parquet"),
    num_items=num_items,
    max_length=200,
    batch_size=256,
    ks=(10, 100),
    device="cuda",
    mask_history=False,   # set True for novelty/recommendation-only tasks
)
print(metrics)  # ndcg@10/100, recall@10/100, coverage@10/100
```

`evaluate()` also takes `num_workers=4` (DataLoader workers),
`use_amp=True` (fp16 autocast on the forward pass), `max_users=None`
(deterministic prefix subset, useful for a quick check on a large
catalog), and `score_chunk=262_144` (chunk size for the per-batch
score matmul — drop it for OOM, raise it for throughput).

For quick re-eval of an existing checkpoint without retraining, the smallest
script is:

```bash
uv run python -c "
import json, torch
from pathlib import Path
from training.encode import load_model_for_eval
from training.evaluate import evaluate

DATA = Path('data/goodreads-work-id')
CKPT = DATA / 'checkpoints/sasrec-ssm-logq-d128'
n = len(json.load(open(DATA / 'item_id_map.json')))
m = load_model_for_eval(CKPT / 'best_model.pt', n, torch.device('cuda'))
print(evaluate(m, str(DATA / 'test.parquet'), num_items=n,
               max_length=200, batch_size=256, ks=(10, 100),
               device='cuda'))
"
```

## Hugging Face Hub

The `.pt` files are **not** stored in git (see `.gitignore`). HF storage is
**one repo per dataset**, with all checkpoints living under that repo's
`checkpoints/<ckpt-id>/` subtree (alongside the dataset's eval inputs).
The registry lives in
[`evaluation/eval_datasets/hub.py`](../../evaluation/eval_datasets/hub.py) as
`EVAL_DATASETS` (`<owner>/eval-<dataset>`, `RETRIEVE_HUB_OWNER`):

| Dataset key | HF repo (dataset type) |
|---|---|
| `goodreads-work-id` | `<owner>/eval-goodreads-work-id` |
| `arxiv-papers` | `<owner>/eval-arxiv-papers` |
| `pubmed` | `<owner>/eval-pubmed` |
| `laion30m` | `<owner>/eval-laion30m` |

A given checkpoint then lands at
`<owner>/eval-<dataset>/tree/main/checkpoints/<ckpt-id>/` rather than its
own model repo. Local layout mirrors that:
`data/<dataset>/checkpoints/<ckpt-id>/`. Override the local root with
`RETRIEVE_DATA_ROOT=/some/path`; otherwise it resolves to
`evaluation/data/` (`hub.data_root()`).

### Auth (one-time setup)

Either log in interactively (token cached under `~/.cache/huggingface/`):
```bash
uv run hf auth login   # paste a write-scoped token
```
or set `HF_TOKEN` in the environment (e.g. in a `.env` or shell profile).
For private repos you need a token with **read** access to download and
**write** access to upload.

### Downloading a checkpoint

`hub.download_checkpoint` pulls a single `checkpoints/<ckpt-id>/` subtree:

```python
from eval_datasets.hub import download_checkpoint

local = download_checkpoint("goodreads-work-id", "sasrec-ssm-logq-d128")
# → data/goodreads-work-id/checkpoints/sasrec-ssm-logq-d128/
```

The matching CLI is `eval-data fetch` (whole dataset, optionally including
checkpoints) — for a single ckpt, call the function above. To grab the dataset's eval inputs (item embeddings, attrs, etc.)
without checkpoints:

```bash
uv run eval-data fetch goodreads-work-id
uv run eval-data fetch goodreads-work-id --dims d64,d128 --include-checkpoints
```

### Uploading a new checkpoint

Push the resulting dir with `train upload-checkpoint`
([`training/checkpoints.py`](../../evaluation/training/checkpoints.py)) — a
thin wrapper over `hub.upload_checkpoint`. It reuses the per-dataset eval
repo (creating it if absent), uploads the checkpoint subtree at
`checkpoints/<ckpt-id>/`, generates a minimal model card from
`eval_quality.json` / `config.json` when `README.md` is absent, and uses
LFS automatically for `.pt` files.

```bash
# Dry-run first to confirm the file list:
uv run train upload-checkpoint \
    --dataset goodreads-work-id \
    --ckpt-id sasrec-ssm-logq-d128 \
    --dry-run

# Real upload:
uv run train upload-checkpoint \
    --dataset goodreads-work-id \
    --ckpt-id sasrec-ssm-logq-d128

# Upload every checkpoint dir under data/<dataset>/checkpoints/ in one go:
uv run train upload-checkpoint --dataset goodreads-work-id --ckpt-id all
```

By default the script skips the epoch-tagged snapshot (`hub.EPOCH_SNAPSHOT_PATTERN`, `*-ep*.pt`: `gsasrec-ep*.pt` and the current trainer's `sasrec-ep*.pt`)
because it has the same bytes as `best_model.pt` (just saved at a different
moment in the training loop). That halves what gets pushed. Pass
`--include-epoch-snapshots` if you want both copies. Other flags:
`--public` (instead of the default `--private`), `--no-write-card` to skip
the auto-generated README. Run output is never pushed
(`hub.CKPT_ALWAYS_IGNORE`: `evaluate.json`, `benchmark.json`, the trainer's
`_resume.pt` and the encode cache `encoded_queries_v2.pt`).

The lower-level `eval-data publish-checkpoint` command exposes the same
upload helper without the `all` selector:

```bash
uv run eval-data publish-checkpoint goodreads-work-id sasrec-ssm-logq-d128 --dry-run
```
