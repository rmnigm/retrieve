# GSASRec checkpoints

> Previously: `retrieve/docs/checkpoints.md` (originally `evaluation/CHECKPOINTS.md`).

Trained on Yambda **500M** Listen+ (50% played-ratio threshold). All runs share
the same architecture and gBCE loss; they differ in `embedding_dim` (and
`ffn_hidden_dim = 4 × embedding_dim`) and dropout. Eval is full catalog ranking
against all 1,866,170 items, no history masking — matches the
[Yambda paper](https://arxiv.org/abs/2505.22238) Table 2 (Listen+) protocol.

## Available checkpoints

| Path | Dim | Dropout | Best epoch | Test NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Notes |
|---|---|---|---|---|---|---|---|---|
| `checkpoints/gsasrec-500m-listens-v1/` | 128 | 0.2 | 37 | 0.0724 | 0.0929 | 0.0339 | 0.1354 | matches paper, slight overfit observed |
| `checkpoints/gsasrec-500m-listens-d128-drop0.5/` | 128 | 0.5 | 54 | 0.0751 | 0.0946 | 0.0353 | 0.1362 | beats paper on every metric except NDCG@10 (tie); was still climbing when stopped |
| `checkpoints/gsasrec-500m-listens-d64-drop0.5/` | 64 | 0.5 | 99 | **0.0813** | **0.1029** | **0.0384** | **0.1489** | bf16 + fused AdamW recipe; was still climbing at the 100-epoch budget cap; **best on every quality metric** |
| `checkpoints/gsasrec-500m-listens-d256-drop0.5/` | 256 | 0.5 | 95 | 0.0753 | 0.0910 | 0.0364 | 0.1284 | same recipe as d64; higher coverage (0.126 vs 0.124) but worse R@100 — extra capacity hurts here |

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

DATA_DIR = Path("data/yambda/500m-listens")
CKPT_DIR = Path("checkpoints/gsasrec-500m-listens-d128-drop0.5")

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

DATA = Path('data/yambda/500m-listens')
CKPT = Path('checkpoints/gsasrec-500m-listens-d128-drop0.5')
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

The `.pt` files are **not** stored in git (see `.gitignore`). Each checkpoint
dir lives as its own model repo on the Hub:

| Local dir | HF repo |
|---|---|
| `checkpoints/gsasrec-500m-listens-v1/` | `<owner>/gsasrec-500m-listens-v1` |
| `checkpoints/gsasrec-500m-listens-d128-drop0.5/` | `<owner>/gsasrec-500m-listens-d128-drop0.5` |

Replace `<owner>` with the HF username/org you uploaded under.

### Auth (one-time setup)

Either log in interactively (token cached under `~/.cache/huggingface/`):
```bash
uv run hf auth login   # paste a write-scoped token
```
or set `HF_TOKEN` in the environment (e.g. in a `.env` or shell profile).
For private repos you need a token with **read** access to download and
**write** access to upload.

### Downloading a checkpoint

```python
from huggingface_hub import snapshot_download

local = snapshot_download(
    repo_id="<owner>/gsasrec-500m-listens-d128-drop0.5",
    local_dir="checkpoints/gsasrec-500m-listens-d128-drop0.5",
    # local_dir_use_symlinks=False,  # uncomment to copy instead of symlink
)
```
After this the loading recipe above works unchanged. To grab a single file
without the whole snapshot (~1.8 GB rather than ~3.6 GB if you only want
`best_model.pt`):
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="<owner>/gsasrec-500m-listens-d128-drop0.5",
                filename="best_model.pt",
                local_dir="checkpoints/gsasrec-500m-listens-d128-drop0.5")
```

### Uploading a new checkpoint

After a training run finishes, push the resulting dir with the helper at
[`training/upload_checkpoints.py`](../../evaluation/training/upload_checkpoints.py). It creates
one HF model repo per checkpoint dir, generates a minimal model card from
`eval_quality.json` / `config.json`, and uses LFS automatically for the
large `.pt` files.

```bash
# Dry-run first to confirm the file list:
uv run upload-checkpoints \
    --owner <hf-user-or-org> \
    --checkpoint gsasrec-500m-listens-d128-drop0.5 \
    --private --dry-run

# Real upload:
uv run upload-checkpoints \
    --owner <hf-user-or-org> \
    --checkpoint gsasrec-500m-listens-d128-drop0.5 \
    --private

# Upload every dir under checkpoints/ in one go:
uv run upload-checkpoints --owner <hf-user-or-org> --checkpoint all --private
```

By default the script skips the epoch-tagged `gsasrec-ep*-ndcg*.pt` snapshot
because it has the same bytes as `best_model.pt` (just saved at a different
moment in the training loop). That halves what gets pushed to Hub. Pass
`--include-epoch-snapshots` if you want both copies.

Other useful flags: `--public` (instead of `--private`), `--repo-name OTHER`
to override the default name (single checkpoint only), `--no-write-card` to
skip the auto-generated README.

## Hyperparameter notes

- `mask_history=False` is the correct default for re-consumption tasks (music
  re-listens). Setting it to `True` halves NDCG@10 because the items the model
  most wants to recommend (already-heard tracks) get masked out.
- `dropout=0.5` is the SASRec/gSASRec paper default and clearly outperforms
  0.2 on this dataset (see table above).
- The d128-drop0.5 run was stopped manually before convergence; val NDCG@10
  was still rising at epoch 54. Resume / extend training would likely push the
  numbers further past the paper.
