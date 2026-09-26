# Gate B — yambda-500m d64 retrain on the new trainer

Recipe: the published `gsasrec-d64-drop0.5` config (2 blocks, 2 heads, ffn 256, dropout 0.5,
B=256, lr 1e-3, wd 0, 100 epochs, patience 20, eval_every 2, early stop ndcg@10, K=256,
gbce_t 0.75), data `/data/yambda-500m/trainer`, `compile=true`, H100 80GB HBM3.
Command: `command.sh`. Prediction: `prediction.md` (written before each launch).

## Run 1: one shared negative vector per step (the plan's §3) — failed, by design

`/scratch/ckpt/gate-b-shared-k256/`, early stop at epoch 47. Test ndcg@10 0.0160,
recall@100 0.0377, coverage@10 1.6e-5. Val ndcg@10 peaked 0.0162 at epoch 7 and fell after.

Mechanism (`diag_collapse.py` -> `shared-k256-collapse.json`, 2048 test users):

| | published (per-position) | shared K=256 |
|---|---|---|
| mean pairwise query cosine | 0.37 | 0.9988 |
| distinct items across the 2048 top-10s | 12,273 | 21 |
| share of top-10 slots on the 0.1 % most frequent items | 0.28 | 1.00 |
| mean output-row norm, all items | 1.05 | 0.186 (init ~0.16) |
| mean output-row norm, 0.1 % most frequent | 0.59 | 1.15 |

With one `[256]` vector per step, the whole run draws 47 x 358 x 256 = 4.3 M negatives over
1.87 M items (~2 per item). Per-position draws ~47 k x 256 = 12 M per *step*. Almost no
output row ever gets a negative gradient, so the rows of rarely-seen items stay at their init
norm. The positive pull grows the popular rows, and every query in a batch is pushed away
from the same 256 rows, so nothing separates users. The queries collapse onto one direction,
the direction of the popular rows, and every user gets the same ~20 items.

Ruled out as the cause:
- The dataset, targets/mask and eval path: a 2-epoch control through the same
  `load_sequences` / `train_batches` / `step_loss` mask / `evaluate` with only the sampling
  switched to per-position reproduced the published curve (epoch-0 loss 0.0678 vs 0.0688,
  epoch-1 val ndcg@10 0.0199 vs 0.0200).
- Frozen negatives: they are redrawn by `torch.randint` in every `step_loss` call.
- gBCE alpha: `K/(N-1)` per row is the same in both schemes.
- Collisions: a positive equals one of the 256 draws with p ~ 1.4e-4 in both schemes.

Larger shared K (2-epoch probes, same control) does not fix it: K=4096 gives epoch-1 val
ndcg@10 0.0118, K=32768 gives 0.0003. So gBCE samples per position (`losses.gbce_loss`
takes `[P, K]`), and only `sampled_softmax` keeps one shared candidate vector.

## Run 2: per-position negatives (Gate B proper)

`/scratch/ckpt/gate-b-yambda-d64/`: see `config.json`, `train_metrics.json`,
`eval_quality.json`, `log_tail.txt` here.

## Reference numbers on today's data

The brief's targets (test ndcg@10 0.0813, recall@100 0.1489) are the published
`eval_quality.json`, scored on a test split that is not the one on disk. The published
checkpoint re-scored on `/data/yambda-500m/test.parquet` (== trainer/test.parquet up to row
order) gives 0.0846 / 0.1563, and on trainer/val.parquet 0.0920, which matches its own
training-time best val of 0.0920 (see `../gate-a/`).
