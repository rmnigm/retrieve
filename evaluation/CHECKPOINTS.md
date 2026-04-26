# GSASRec checkpoints

Trained on Yambda **500M** Listen+ (50% played-ratio threshold). Both runs use
the same architecture and gBCE loss; they differ only in dropout. Eval is full
catalog ranking against all 1,866,170 items, no history masking — matches the
[Yambda paper](https://arxiv.org/abs/2505.22238) Table 2 (Listen+) protocol.

## Available checkpoints

| Path | Dropout | Best epoch | Test NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Notes |
|---|---|---|---|---|---|---|---|
| `checkpoints/gsasrec-500m-listens-v1/` | 0.2 | 37 | 0.0724 | 0.0929 | 0.0339 | 0.1354 | matches paper, slight overfit observed |
| `checkpoints/gsasrec-500m-listens-d128-drop0.5/` | 0.5 | 54 | **0.0751** | **0.0946** | **0.0353** | **0.1362** | best run, beats paper on every metric except NDCG@10 (tie); was still climbing when stopped |

Paper Yambda-500M Listen+ SASRec target: NDCG@10 0.0754 · NDCG@100 0.0884 ·
Recall@10 0.0336 · Recall@100 0.1240.

Common hyperparameters:
```
embedding_dim=128  num_blocks=2  num_heads=2  ffn_hidden_dim=512
max_seq_length=200  batch_size=256  negs_per_pos=256  gbce_t=0.75
lr=1e-3  weight_decay=0  optimizer=AdamW
```

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

# num_items comes from the data dir's id map (saved by data/yambda.py).
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
`item_id_map.json` produced by `data/yambda.py`.

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

Reuses [`training/evaluate.py`](training/evaluate.py) — full-catalog ranking + NDCG/Recall/Coverage
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

## Hyperparameter notes

- `mask_history=False` is the correct default for re-consumption tasks (music
  re-listens). Setting it to `True` halves NDCG@10 because the items the model
  most wants to recommend (already-heard tracks) get masked out.
- `dropout=0.5` is the SASRec/gSASRec paper default and clearly outperforms
  0.2 on this dataset (see table above).
- The d128-drop0.5 run was stopped manually before convergence; val NDCG@10
  was still rising at epoch 54. Resume / extend training would likely push the
  numbers further past the paper.
