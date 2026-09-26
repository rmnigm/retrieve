---
chain: "e4-kuairand"
branch: "main"
parent: "2026-09-26-005934756-e4-kuairand-staged-layout-clean-training-blocked-on-gpu.md"
nextStep: "Worker on /scratch/wt/e4b (dev/e4-training): when training pid 302129 exits, publish the checkpoint from a hard-linked staging dir via `eval-data publish-checkpoint kuairand gsasrec-d128-shared --source <dir>` (NOT the raw dir: hub.py would upload the 49 GB _resume.pt), then run one filter cell, update docs, commit."
created: "2026-09-26T08:16:14Z"
---

# E4 gSASRec training running (negs 128, patience 5)

## Current work
- Worktree `/scratch/wt/e4b`, branch `dev/e4-training`. Imports forced to the worktree with
  `PYTHONPATH=/scratch/wt/e4b/evaluation:/scratch/wt/e4b/retrieve/src` (the shared `/venvs/retrieve`
  editable install points at `/workspace/retrieve`). Added gitignored symlink `evaluation/data -> /data`
  (same as the main checkout). `bench check --dataset kuairand`: `ok`.
- Training (nohup, pid 302129, log in the session scratchpad `train.log`), started 08:05 UTC:
  `PYTORCH_ALLOC_CONF=expandable_segments:True train sasrec --data-dir data/kuairand --checkpoint-dir data/kuairand/checkpoints/gsasrec-d128-shared --embedding-dim 128 --dropout 0.5 --reuse-item-embeddings --eval-max-users 4096 --negs-per-pos 128 --patience 5 --num-epochs 100 --no-wandb`
- Epoch 0: ~11 min, loss 0.0267, val ndcg@10 0.0008.

## Decisions
- `--negs-per-pos 256` (default) OOMs on the first backward (tried 6.69 GiB with 74.8 GiB allocated,
  before AdamW's moments even exist). 128 fits: probe peak `max_memory_allocated` 82.9e9 B (77.2 GiB).
- `--patience 5 --num-epochs 100` instead of 20/200: ~11 min/epoch would make patience 20 cost 3.7 h of dead epochs.
- To stop early: kill after an epoch's `_resume.pt` save, rerun the same command with `--resume --num-epochs <next_epoch>`; the loop is empty and the finalize path (best_model, item_embs, test eval) runs.

## Unresolved
- `hub.py` CKPT_ALWAYS_IGNORE does not skip `_resume.pt` (49 GB incl. optimizer state) or `encoded_queries_v2.pt`; not edited (brief forbids touching hub.py).
- Disk `/` : a finished run holds ~99 GB (resume 49 + three 16.4 GB tables); 105 GB was free at launch.
