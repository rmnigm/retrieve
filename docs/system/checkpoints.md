# GSASRec checkpoints

> Previously: `retrieve/docs/checkpoints.md` (originally `evaluation/CHECKPOINTS.md`).

Trained on Yambda **500M** Listen+ (50% played-ratio threshold). All runs share
the same architecture and gBCE loss; they differ in `embedding_dim` (and
`ffn_hidden_dim = 4 × embedding_dim`) and dropout. Eval is full catalog ranking
against all 1,866,170 items, no history masking — matches the
[Yambda paper](https://arxiv.org/abs/2505.22238) Table 2 (Listen+) protocol.

Checkpoints now live under each dataset's data dir:
`data/<dataset>/checkpoints/<ckpt-id>/` (e.g.
`data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/`). The eval CLI
configs in [`evaluation/conf/`](../../evaluation/conf/) point at these
paths verbatim.

## Available checkpoints

| Path | Dim | Dropout | Best epoch | Test NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Notes |
|---|---|---|---|---|---|---|---|---|
| `data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/` | 128 | 0.5 | 54 | 0.0751 | 0.0946 | 0.0353 | 0.1362 | beats paper on every metric except NDCG@10 (tie); was still climbing when stopped |
| `data/yambda-500m/checkpoints/gsasrec-d64-drop0.5/` | 64 | 0.5 | 99 | **0.0813** | **0.1029** | **0.0384** | **0.1489** | bf16 + fused AdamW recipe; was still climbing at the 100-epoch budget cap; **best on every quality metric** |
| `data/yambda-500m/checkpoints/gsasrec-d256-drop0.5/` | 256 | 0.5 | 95 | 0.0753 | 0.0910 | 0.0364 | 0.1284 | same recipe as d64; higher coverage (0.126 vs 0.124) but worse R@100 — extra capacity hurts here |

The original `gsasrec-500m-listens-v1/` (dim 128, dropout 0.2, ep 37 — paper
parity) was retired when the dropout=0.5 variants subsumed it.

Other datasets follow the same `data/<dataset>/checkpoints/<ckpt-id>/`
layout: 5B runs at `data/yambda-5b/checkpoints/gsasrec-d{64,128}/`,
goodreads at `data/goodreads-work-id/checkpoints/gsasrec-d{64,128,256}-drop0.5-id/`.
Arxiv has no SASRec checkpoint — its queries come from a pre-encoded text
embedding tensor (see [evaluation.md](evaluation.md)).

Paper Yambda-500M Listen+ SASRec target: NDCG@10 0.0754 · NDCG@100 0.0884 ·
Recall@10 0.0336 · Recall@100 0.1240.

Common hyperparameters:
```
num_blocks=2  num_heads=2  ffn_hidden_dim=4×embedding_dim
max_seq_length=200  batch_size=256  negs_per_pos=256  gbce_t=0.75
lr=1e-3  weight_decay=0  optimizer=AdamW (fused on cuda)
autocast=bfloat16   tf32=on
```

The `-d64-drop0.5` run was the first to use the bf16/fused-AdamW/TF32 stack.
The two earlier `d128` runs predated it and used fp16+GradScaler.

## What's in each checkpoint dir

- `gsasrec-ep{N}-ndcg10{X}.pt` — model `state_dict` saved at the best val epoch.
- `best_model.pt` — same `state_dict`, copied at the end of training (or on
  manual stop). Use this one going forward.
- `eval_quality.json` — final test metrics, `best_epoch`, `paper_target`.

> **Missing artifacts (killed mid-training):** the 500M runs do not include
> `config.json`, `item_embs.pt`, or a copy of `item_id_map.json`. The hyperparams
> are documented above; embeddings can be regenerated in a few lines (see below).

## Loading a checkpoint

```python
import json, torch
from pathlib import Path
from training.model import GSASRec

DATA_DIR = Path("data/yambda-500m")
CKPT_DIR = DATA_DIR / "checkpoints/gsasrec-d128-drop0.5"

# num_items comes from the data dir's id map (written by `data/yambda.py` prep).
with open(DATA_DIR / "item_id_map.json") as f:
    num_items = len(json.load(f))

model = GSASRec(
    num_items=num_items,
    max_seq_length=200,
    embedding_dim=128, num_heads=2, num_blocks=2,
    ffn_hidden_dim=512, dropout=0.5,
    reuse_item_embeddings=False,
).cuda().eval()

model.load_state_dict(
    torch.load(CKPT_DIR / "best_model.pt", map_location="cuda", weights_only=True)
)
```

The `num_items + 1` row count of every embedding tensor reserves index 0 for
padding — never use id 0 for a real item.

## Extracting item embeddings

Two embedding tables exist on the model: the *input* item embedding (used inside
the transformer) and the *output* embedding (used to score candidates).
For retrieval / ranking, always use the output table.

```python
item_embs = model.get_output_embeddings().weight.detach().cpu()  # [num_items+1, D]
item_embs[0, :] = 0.0                                            # zero out padding row
torch.save(item_embs, CKPT_DIR / "item_embs.pt")
```

Use this tensor as the index for ANN search, dot-product retrieval, or
clustering. The id mapping (dense_id → raw_yandex_id) is the
`item_id_map.json` written by
the [`data/yambda.py`](../../evaluation/data/yambda.py) `prep` subcommand
(the library function `preprocess()` returns `Data.item_id_to_idx` in
memory; the CLI persists it to disk alongside the parquets).

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
and feed it the user's recent listen sequence.

## Running offline evaluation

Reuses [`training/evaluate.py`](../../evaluation/training/evaluate.py) — full-catalog ranking + NDCG/Recall/Coverage
@10/100 against `val.parquet` / `test.parquet`. Defaults to no history masking,
which is the correct setting for re-listen tasks like Listen+.

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
print(metrics)
# {'ndcg@10': 0.0751, 'ndcg@100': 0.0946, 'recall@10': 0.0353,
#  'recall@100': 0.1362, 'coverage@10': 0.0399, 'coverage@100': 0.1049}
```

`evaluate()` also takes `num_workers=4` (DataLoader workers),
`use_amp=True` (bf16 autocast on the forward pass), `max_users=None`
(deterministic prefix subset, useful for quick iteration on the 5B
catalog), and `score_chunk=262_144` (chunk size for the per-batch
score matmul — drop it for OOM, raise it for throughput).

For quick re-eval of an existing checkpoint without retraining, the smallest
script is:

```bash
uv run python -c "
import json, torch
from pathlib import Path
from training.model import GSASRec
from training.evaluate import evaluate

DATA = Path('data/yambda-500m')
CKPT = DATA / 'checkpoints/gsasrec-d128-drop0.5'
n = len(json.load(open(DATA / 'item_id_map.json')))

m = GSASRec(num_items=n, max_seq_length=200, embedding_dim=128,
            num_heads=2, num_blocks=2, ffn_hidden_dim=512,
            dropout=0.5, reuse_item_embeddings=False).cuda()
m.load_state_dict(torch.load(CKPT / 'best_model.pt',
                  map_location='cuda', weights_only=True))
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
[`evaluation/data/hf_io.py`](../../evaluation/data/hf_io.py) as
`EVAL_REPOS`:

| Dataset key | HF repo (dataset type) |
|---|---|
| `yambda-500m` | `pinkmeme/eval-yambda-500m` |
| `yambda-5b` | `pinkmeme/eval-yambda-5b` |
| `goodreads-work-id` | `pinkmeme/eval-goodreads-work-id` |
| `arxiv-papers` | `pinkmeme/eval-arxiv-papers` |

A given checkpoint then lands at
`pinkmeme/eval-<dataset>/tree/main/checkpoints/<ckpt-id>/` rather than its
own model repo. Local layout mirrors that:
`data/<dataset>/checkpoints/<ckpt-id>/`. Override the local root with
`RETRIEVE_DATA_ROOT=/some/path`; otherwise it resolves to
`evaluation/data/`.

### Auth (one-time setup)

Either log in interactively (token cached under `~/.cache/huggingface/`):
```bash
uv run hf auth login   # paste a write-scoped token
```
or set `HF_TOKEN` in the environment (e.g. in a `.env` or shell profile).
For private repos you need a token with **read** access to download and
**write** access to upload.

### Downloading a checkpoint

`hf_io.download_checkpoint` pulls a single `checkpoints/<ckpt-id>/` subtree:

```python
from data.hf_io import download_checkpoint

local = download_checkpoint("yambda-500m", "gsasrec-d128-drop0.5")
# → data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/
```

The matching CLI is `eval-fetch` (whole dataset, optionally including
checkpoints) — for a single ckpt it's currently easier to call the function
above. To grab the dataset's eval inputs (item embeddings, attrs, etc.)
without checkpoints:

```bash
uv run eval-fetch yambda-500m
uv run eval-fetch goodreads-work-id --dims d64,d128 --include-checkpoints
```

### Uploading a new checkpoint

Push the resulting dir with the
[`upload-checkpoints`](../../evaluation/training/upload_checkpoints.py)
console script — a thin wrapper over `hf_io.upload_checkpoint`. It reuses
the per-dataset eval repo (creating it if absent), uploads the checkpoint
subtree at `checkpoints/<ckpt-id>/`, generates a minimal model card from
`eval_quality.json` / `config.json` when `README.md` is absent, and uses
LFS automatically for `.pt` files.

```bash
# Dry-run first to confirm the file list:
uv run upload-checkpoints \
    --dataset yambda-500m \
    --ckpt-id gsasrec-d128-drop0.5 \
    --dry-run

# Real upload:
uv run upload-checkpoints \
    --dataset yambda-500m \
    --ckpt-id gsasrec-d128-drop0.5

# Upload every checkpoint dir under data/<dataset>/checkpoints/ in one go:
uv run upload-checkpoints --dataset yambda-500m --ckpt-id all
```

By default the script skips the epoch-tagged `gsasrec-ep*.pt` snapshot
because it has the same bytes as `best_model.pt` (just saved at a different
moment in the training loop). That halves what gets pushed. Pass
`--include-epoch-snapshots` if you want both copies. Other flags:
`--public` (instead of the default `--private`), `--no-write-card` to skip
the auto-generated README.

The lower-level `eval-publish-checkpoint` console script exposes the same
upload helper without the `all` selector:

```bash
uv run eval-publish-checkpoint yambda-500m gsasrec-d128-drop0.5 --dry-run
```

## Hyperparameter notes

- `mask_history=False` is the correct default for re-consumption tasks (music
  re-listens). Setting it to `True` halves NDCG@10 because the items the model
  most wants to recommend (already-heard tracks) get masked out.
- `dropout=0.5` is the SASRec/gSASRec paper default and clearly outperforms
  0.2 on this dataset (see table above).
- The d128-drop0.5 run was stopped manually before convergence; val NDCG@10
  was still rising at epoch 54. Resume / extend training would likely push the
  numbers further past the paper.
