---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-135000000-research-hstu-argus-losses-efficiency.md"
nextStep: "Create dev/hstu off origin/staging and push it; dispatch W1 (fable, worktree /scratch/wt/encoder, plan steps 1-8 amended below) and W2 (opus, worktree /scratch/wt/etl, ETL timestamps + goodreads download/convert/prep + yambda re-prep with hash check, CPU only). After each returns, dispatch an opus review agent (deslop + python-review) on its diff before merging into dev/hstu."
created: "2026-09-26T10:47:18Z"
---

# Reconciled plan for dev/hstu (orchestrator, pod rp-h100-hstu)

Filename timestamp is in the laptop's local frame (UTC+3) so the chain sorts;
the pod clock is UTC (`created` above).

## 1. Primary request and intent
Replace gSASRec with a better, bigger, generic id(+timestamp)-only history encoder
(HSTU and beyond) with more efficient training; goodreads + yambda-500m first,
KuaiRand only if the winner beats gSASRec at the same D on test NDCG@10 **and**
R@100 on both. All work on `dev/hstu` (off origin/staging ae0290d), pushed to
origin, never merged into staging. I plan, dispatch, review, merge; I do not write code.

## 2. Verification of the architecture plan against origin/staging (ae0290d)
Checked by reading the code on this pod; the plan's facts hold:
- `training/` is 1,162 lines: `GSASRecConfig` (config.py), `GSASRec` wrapping
  `nn.TransformerEncoder` (model.py), per-position `[P,negs,D]` gBCE (losses.py),
  DataLoader with CPU `randint` negatives `[B,L,negs]` (dataset.py), fused AdamW +
  bf16 autocast + TF32 + `clip_grad_norm_` over all params (train.py).
- `eval_datasets/hub.py:84` `EPOCH_SNAPSHOT_PATTERN = "gsasrec-ep*.pt"`, `:87`
  `CKPT_ALWAYS_IGNORE`; `encode.py:27` `D128_DROP05_DEFAULTS`, `:38` `load_model_for_eval`.
- All three ETLs aggregate `timestamp` next to `item_id` (goodreads.py:530,
  yambda.py:59, kuairand.py:273) and drop it at the final select.
- Data on /data: goodreads-work-id eval inputs + checkpoints d64/d128/d256; yambda-500m
  eval inputs + checkpoints d64/d128/d256 + `trainer/` (no timestamps). Goodreads raw:
  only a 48 MB partial `goodreads_books.json.gz` (prep log stopped at `START [1/13]`).

Baselines to beat (published `eval_quality.json`, test, full catalog; not yet
reproduced on the H100 = Gate A):
| dataset | D | NDCG@10 | R@100 |
|---|---|---|---|
| goodreads-work-id | 64 | 0.0350 | 0.1486 |
| goodreads-work-id | 128 | 0.0361 | 0.1480 |
| goodreads-work-id | 256 | 0.0354 | 0.1472 |
| yambda-500m | 64 | 0.0813 | 0.1489 |
| yambda-500m | 128 | 0.0751 | 0.1362 (no config.json / item_embs; D128_DROP05_DEFAULTS) |
| yambda-500m | 256 | 0.0753 | 0.1284 |
gSASRec does not improve with D on either dataset: the body/recipe, not D, is the lever.

## 3. Decisions (plan vs research) and rationale
- **D1 Ladder order kept: loss first on the existing block (L1).** Both sources agree
  the loss is the best-evidenced lever ("Dross into Gold": sampled CE .1857 vs BCE .1341).
- **D2 Normalized embeddings with temperature (amends plan §3).** The plan's tau=0.05 on
  raw dot products is wrong: HSTU's recipe uses cosine similarity with tau 0.05. Add
  `normalize: bool` (default on with `sampled_softmax`): L2-normalize query and item
  vectors in the loss and in `encode.py`, so the stored `[N,D]`/`[B,D]` are unit vectors
  and dot-product scoring (output contract) equals cosine. gBCE path unchanged.
- **D3 logQ is an ablation, not a default.** HSTU's recipe has no logQ; Argus uses it.
  L1 runs twice on goodreads d64: logQ off and on. Correction is per candidate from the
  mixed proposal (M in-batch ~ popularity, K uniform); the positive's own column is not
  logQ-corrected (2507.09331). Winner of the pair becomes the shared recipe.
- **D4 Isolation variant added (research delta 2).** `LlamaBlock` takes the same
  additive relative-position(+time) bias (`attn_bias: none|rab`) that `HSTUBlock` uses,
  passed as a float mask to SDPA (softmax kept). Run L3b = llama+rab only after L3 so the
  HSTU gain is split into bias vs pointwise-SiLU. One shared `rab` module, not two.
- **D5 Timestamps before goodreads prep.** Goodreads trainer inputs do not exist yet, so the
  ETL timestamp change (plan step 9) lands first and goodreads is prepped once, with
  timestamps. Column `timestamps` (list[int64], unix seconds; unit converted in ETL and
  documented in datasets.md) in train/val/test parquet. Yambda re-prep must leave
  `item_id_map.json` and `item_ids` byte-identical (hash before/after, recorded).
  This is a disjoint tree (`eval_datasets/etl/`) and CPU only: runs as W2 in parallel with W1.
- **D6 Rowwise Adagrad + bf16 table stay in the rewrite but off by default;** only the
  winner is ablated with them (<1% rel loss allowed); they are the KuaiRand enabler.
- **D7 Kept from plan:** one `Encoder` (sasrec|llama|hstu blocks, `hidden_dim` + `out_proj`),
  GSASRec deleted, `TrainConfig` renamed with no alias, shared-negatives GPU sampling,
  compile only the dense body, Gate A (4-decimal equality on published checkpoints via the
  new loader), Gate B (retrain sasrec+gbce goodreads d64 within ±0.002 NDCG@10),
  stop rule (+2% rel NDCG@10 on goodreads or regression on yambda), 6 h cap per run.
- **Rejected for now:** SCE/RECE hard negatives, FuXi-α channels, max-batch-1-neg (2608.11061),
  jagged attention, Meta Triton kernels. Revisit only if L1-L4 plateau below the bar.
- Scaling watch (2412.00430 inverted U): every ladder run logs val curve; L4 (H=512, 6 blocks)
  only if L2/L3 do not already overfit.

## 4. Dispatch plan
| id | model | worktree / branch | scope | GPU |
|---|---|---|---|---|
| W1 | fable | /scratch/wt/encoder, `dev/hstu-encoder` | plan steps 1-8 with D2-D4, D6: training/ rewrite, hub.py ignore list + snapshot pattern, tests, Gate A, Gate B, docs/system/checkpoints.md + evaluation/datasets Training section | yes (sole holder) |
| W2 | opus | /scratch/wt/etl, `dev/hstu-etl` | ETL `timestamps` column (goodreads, yambda, kuairand), goodreads download/convert/prep to /data/goodreads-work-id/trainer, yambda re-prep + hash check, datasets.md | no |
| R* | opus | reviewer in author's worktree | deslop + python-review, ruff, harness suite; fixes or findings back | no |
Then ladder runs (L1a/b, L2, L3, L3b, L4, d128, yambda) as opus run workers, one GPU job at a time.

## 5. Pending tasks
Create/push dev/hstu; dispatch W1, W2; review gates; merge; Gate A/B notes; ladder; KuaiRand go/no-go; final note.
