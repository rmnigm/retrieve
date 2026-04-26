You are training GSASRec on the Amazon Reviews 2023 dataset using the existing
pipeline at /workspace/evaluation. The data prep, model, and training loop
already support Amazon — your job is to plan the run, choose a sensible
category subset, run it, and report numbers vs. published SASRec baselines.

## Repo

Working dir: /workspace/evaluation

Key files (all already on disk; do NOT re-architect):
- data/amazon.py            — pulls McAuley-Lab/Amazon-Reviews-2023 from HF,
                              filters reviews with rating ≥ 4.0 (positive
                              implicit feedback), runs k-core (min 5
                              interactions per user/item), remaps to dense ids,
                              and does a **leave-last-out** split: per user,
                              the last item is `test`, the second-last is
                              `val`, the rest is `train`. Outputs
                              `train/val/test.parquet` (with `item_ids` and
                              optional `targets` lists) plus
                              `item_id_map.json`. Optional `--download-attrs`
                              produces `item_attrs.parquet` + attr vocab; not
                              needed for plain GSASRec, leave it off.
- training/model.py         — GSASRec (causal transformer, separate output
                              embedding, predict_last for last-pos query).
- training/losses.py        — gBCE with sampled negatives.
- training/dataset.py       — train DataLoader with uniform negatives.
- training/evaluate.py      — torch full-catalog scoring → NDCG/Recall/Coverage
                              via retrieval/metrics.py. Multi-target ready;
                              for Amazon LLO it sees a single target per user
                              (Recall@K == HitRate@K and is computed correctly
                              via num_targets=1).
- training/train_sasrec.py  — trainer with per-epoch eval, early stopping,
                              wandb logging.
- training/config.py        — GSASRecConfig dataclass.
- CHECKPOINTS.md            — doc of Yambda-500M runs (precedent).

wandb is already authed; project so far is `yambda-gsasrec`. For Amazon, use
`amazon-gsasrec`.

## What is different from the Yambda runs

1. **Split is leave-last-out, not GTS.** The target is the *last* (or
   second-last) item per user. It is NOT in the training history, but every
   other user history item is. This means:
   - **mask_history MUST be True** — otherwise the model can score historical
     items the user already engaged with, which is the wrong thing for
     next-item prediction. (Yambda Listen+ wanted history *un*masked because
     re-listens are the target. For Amazon next-purchase, you don't want
     already-purchased items in the top-K.)
   - Each user has exactly 1 target → HitRate@K = Recall@K = NDCG@K (when
     gain is binary). The accumulate_metrics fn handles this; just read
     `recall@10` and `ndcg@10` from the returned dict.

2. **Item catalog scales with category choice.** All 33 categories combined
   is millions of items and can OOM on full-catalog scoring. Common choices
   from the GSASRec literature:
   - `Beauty_and_Personal_Care` (~700K items, fast, popular benchmark)
   - `Sports_and_Outdoors` (~1.6M items)
   - `Toys_and_Games` (~600K items)
   - `Books` (~5M items, large; chunked eval recommended)
   Default to a single mid-size category for the first run; expand later.

3. **No paper baseline in the Yambda paper for Amazon.** Reference the
   gSASRec paper (Petrov & Macdonald, 2023) and the original SASRec paper
   (Kang & McAuley, 2018) for the relevant category. Expected order of
   magnitude on Beauty:
   - SASRec: NDCG@10 ≈ 0.05–0.07, HitRate@10 ≈ 0.10–0.13.
   - gSASRec / GSASRec with gBCE: typically +5–15% over SASRec.
   Don't over-fit to a specific number; aim for "in the right neighborhood".

4. **Time-axis assumptions don't apply.** No GTS, no train_end constants — the
   Amazon prep already handles everything. The Yambda time constants in
   data/yambda.py are unrelated.

## Plan (the agent should follow this; deviate when warranted)

1. **Plan mode first.** Read data/amazon.py, training/{model,evaluate,
   train_sasrec}.py, CHECKPOINTS.md. Confirm understanding by sketching the
   data-flow per file before writing code.

2. **Pick a category and prep data.** Start with three categories to allow attribute filtering later; check sized after you filtered to items which actually have actions.

3. **Smoke train.** 1 epoch, --max-batches-per-epoch 50, --no-wandb, to
   confirm shapes/memory are fine and eval runs end-to-end. Check that the
   logged val ndcg@10 is non-zero and looks like a single-target eval.

4. **Full run.** Defaults that worked at Yambda-500M:
   ```
   uv run python -m training.train_sasrec \
     --data-dir data/amazon/beauty \
     --checkpoint-dir checkpoints/gsasrec-amazon-beauty-d128-drop0.5 \
     --embedding-dim 128 --num-blocks 2 --num-heads 2 --dropout 0.5 \
     --max-seq-length 200 --batch-size 256 --negs-per-pos 256 --gbce-t 0.75 \
     --lr 1e-3 --num-epochs 200 --patience 10 \
     --eval-batch-size 256 --eval-every 2 --early-stop-metric ndcg@10 \
     --mask-history \
     --wandb --wandb-project amazon-gsasrec \
     --wandb-run-name beauty-d128-drop0.5
   ```
   Note `--mask-history` (the flag is off by default after the Yambda fix;
   pass it explicitly here so the eval semantics are right for Amazon).

5. **Final test eval + persist.** Same shape as the bottom of CHECKPOINTS.md:
   load best ckpt, run evaluate(...) on test.parquet with mask_history=True,
   write eval_quality.json and best_model.pt, save item_embs.pt.

6. **OOM contingencies.** If full-catalog scoring OOMs at >2M items:
   - Drop `--eval-batch-size` to 64.
   - Add chunked scoring in training/evaluate.py (loop over item-id chunks,
     do per-chunk topk, merge). Verify identical numbers vs the un-chunked
     path on a small sanity case before relying on it.

7. **Document.** Add a new row to CHECKPOINTS.md (or a fresh doc
   `docs/AMAZON.md` if you prefer keeping datasets separate) with:
   - Category, num_items, num_users, train/val/test row counts.
   - Full hyperparams.
   - Test metrics (ndcg@10/100, recall@10/100, coverage@10/100).
   - One-line comparison vs SASRec / gSASRec literature ballpark.
   - The exact eval command for reproducing.

## Non-goals

- Do NOT touch the `retrieve/` framework. Eval stays plain torch.
- Do NOT switch the split. Leave-last-out is the standard for Amazon SASRec
  benchmarks; do not "fix" it to a temporal split.
- Do NOT add `--download-attrs` unless the user explicitly asks; attribute
  filtering is for the silvertorch index, not for vanilla GSASRec eval.
- Do NOT change wandb_project naming; keep `amazon-gsasrec` so runs cluster.

## Output expectations

- A single concise report message at the end:
  - dataset / category / scale,
  - test NDCG@10, NDCG@100, Recall@10 (=HitRate@10), Recall@100, Coverage@10/100,
  - hyperparams headline,
  - wandb URL,
  - any deviations from the plan and why.
- Updated doc in /workspace/evaluation/.
- Saved checkpoint dir with best_model.pt + eval_quality.json + item_embs.pt.

Keep noise low. Do not narrate every tool call; only flag findings, course
corrections, and the final result.
