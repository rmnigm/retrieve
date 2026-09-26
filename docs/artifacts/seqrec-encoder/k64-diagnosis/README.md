# R-k64 diagnosis (KuaiRand-27K d64)

Bounded diagnosis ordered by instruction 233445061 (20:35-20:40 UTC on the GPU, evals only; H100).
Not yet validated, not citable. `diag.py` -> `diag.json` (protocol, ceilings, most-popular, seen-only
targets); `recency.py` -> `recency.json` (temporal drift). k64 = `best_model.pt` (epoch 3).

## 1. Protocol: val and test targets

| | val | test |
|---|---|---|
| rows | 24,503 | 26,221 |
| targets per row, p10 / p50 / p90 (mean) | 20 / 118 / 346 (158) | 50 / 246 / 676 (320) |
| history length | 200 for every row (capped) | 200 for every row |
| target day | 2022-05-06, the day after train ends | 2022-05-07, two days after; the val day is never trained on |
| recall@10 / recall@100 ceiling (`recall` divides by all targets) | 0.183 / 0.732 | 0.096 / 0.505 |
| targets seen in train | 65.9 % | 44.7 % |
| targets in the row's own history | 0.005 % | 0.004 % |

ndcg@10 normalizes by min(10, n), so more targets per row make it easier to score, not harder. The
recall ceilings differ by 1.45x, which does not explain a 5x ndcg@10 gap. **The protocol alone does
not explain the gap.**

## 2. Calibration (same `evaluate` path, full catalog)

| scorer | val ndcg@10 / R@100 | test ndcg@10 / R@100 |
|---|---|---|
| most-popular over train | 0.0012 / 0.0007 | 0.0027 / 0.0007 |
| **most-popular over the val day's targets** (2022-05-06) | (it is the val day) | **0.0314 / 0.0073** |
| R-k64 | 0.0232 / 0.0104 | 0.0046 / 0.0016 |
| R-k64, train-seen targets only | 0.0233 / 0.0156 | 0.0046 / 0.0032 |

R-k64 beats all-time popularity (19x on val, 1.7x on test). **On test it is beaten ~7x on ndcg@10 by
one global list of yesterday's most-clicked items.** Restricting to train-seen targets leaves ndcg@10
unchanged, so cold items are not the cause. This corrects the R-k64 README, which suggested they were.

## 3. The curve

Val ndcg@10 peaks at epoch 3 (0.0232), then drifts to ~0.020 while train loss keeps falling
(12.7 -> 11.3) and coverage@10 keeps rising (0.0007 -> 0.0018): the model spreads toward the tail,
which is overfitting to the train distribution. It is not a popularity collapse. Test targets are
concentrated on trending items: 47.6 % of them were clicked on the val day, 13.0 % only on the val
day (never in train), 42.3 % in neither. logQ subtracts all-time train popularity, which is the wrong
prior for what trends on a later day.

## 4. Context

`.chains/e4-kuairand/`: the A100 gSASRec d128 run logged val ndcg@10 0.0133 at epoch 2 and has no
test number; the baseline research found no comparable published number.

## 5. Mechanism

KuaiRand-27K next-day clicks are dominated by fresh, trending items. A model trained on
interactions up to 2022-05-05 cannot know what trends on 2022-05-07, and the val day in between is
never trained on. The gap is temporal drift under this split: the model is 1 day stale on val and 2
days stale on test. More capacity (d128) does not address it.
