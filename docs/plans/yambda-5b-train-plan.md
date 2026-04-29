# Yambda-5B Listen+ training plan (d=64)

## Target

Train **GSASRec** ([evaluation/training/model.py](../../evaluation/training/model.py)) on **Yambda-5B Listen+** at `embedding_dim=64`. Same recipe as the 500M-d64 best checkpoint (NDCG@10 0.0813, see [docs/checkpoints.md](../system/checkpoints.md)), scaled up to 1M users / 9.39M items.

## Data

- **Catalog**: 9,390,623 items
- **Listen+ events**: ~2.0–2.5 B (50% played-ratio threshold)
- **Sequences**: 1,000,000 (one per user, capped at last 200 ids)
- **Train parquet on disk**: ~250–400 MB zstd
- **Layout** at `data/yambda/5b-listens-plus/`:
  - `train.parquet` — col `item_ids`
  - `val.parquet` / `test.parquet` — cols `item_ids` + `targets`
  - `item_id_map.json` — dense int → raw yandex id

## Hardware fit (A100 80GB)

| | Dense AdamW | **Sparse + shared-negs** |
|---|---|---|
| Peak GPU | ~22–24 GB | ~17–19 GB |
| Optim step (embeddings) | ~120 ms | ~5 ms |
| Train batch | ~80–110 ms | ~30–50 ms |
| Train epoch (3,906 batches) | ~5 min | ~2–3 min |
| 30 epochs, eval every 2, 100K eval users | ~3.5 h | **~2 h** |

H100 only ~1.5–1.7× faster — workload is memory-bandwidth bound (embedding gather/scatter + AdamW over 9.39M rows), not matmul-bound. The algorithmic change buys more than the hardware swap.

## Why sparse + shared-negs is the right pair

- Per-position negs (current): `[B,L,K]=[256,200,256]=13.1 M` random ids → **75% of catalog touched per batch** → SparseAdam saves only ~25%.
- **Shared-batch negs** (sample once per batch, broadcast across positions): `K + B·L unique ≈ 30–40 K → 0.4% of catalog` → **~150× faster optim step on embedding tables, ~2× wall-clock overall**.
- Quality knob: bump `K` to 512 to compensate for less negative diversity (still trivial — `B·K=128 K` vs current 13.1 M). The gBCE `alpha = K/(N−1)` term in [losses.py:25](../../evaluation/training/losses.py#L25) still accounts for K correctly.
- Memory: dense grad buffers (4.8 GB total) collapse to sparse COO (~MB). Param + Adam moments stay dense in `torch.optim.SparseAdam`.

## Code changes

| File | Change |
|---|---|
| [config.py](../../evaluation/training/config.py) | Add `sparse_embeddings`, `shared_batch_negatives`, `eval_max_users`, `eval_score_chunk`. |
| [model.py:25-28,45](../../evaluation/training/model.py#L25-L45) | Pass `sparse=…` to `item_embedding` + `output_embedding` (keep `position_embedding` dense). |
| [dataset.py:35-48](../../evaluation/training/dataset.py#L35-L48) | Add `shared_batch_negatives` mode → `[K]` negatives broadcast across positions. |
| [losses.py:17-23](../../evaluation/training/losses.py#L17-L23) | Handle both `[P,K]` and broadcast `[K]` shapes. Wrap embedding gathers in `autocast(enabled=False)` for sparse path (SparseAdam needs fp32 grads). |
| [train_sasrec.py:88-93](../../evaluation/training/train_sasrec.py#L88-L93) | Two optimizers when sparse: `SparseAdam(embedding tables) + AdamW(rest, fused=True)`. Both `step()` + `zero_grad(set_to_none=True)` each iter. |
| [evaluate.py:99-103](../../evaluation/training/evaluate.py#L99-L103) | Chunk score matmul along item axis; subsample users via `eval_max_users`. |
| [train_sasrec.py:294-306](../../evaluation/training/train_sasrec.py#L294-L306) | Surface new flags as CLI options. |

Resumability (`_resume.pt` with optimizer/RNG/epoch) is on the wishlist but not blocking for a 2–4 h run.

## Run command (tmux, survives VSCode/agent close)

```bash
sudo apt-get install -y tmux  # one-time
mkdir -p /workspace/retrieve/evaluation/logs

tmux new -d -s yambda5b "cd /workspace/retrieve/evaluation && \
  python -m training.train_sasrec \
    --data-dir data/yambda/5b-listens-plus \
    --checkpoint-dir checkpoints/gsasrec-5b-listens-d64-sparse \
    --embedding-dim 64 --num-blocks 2 --num-heads 2 --dropout 0.5 \
    --batch-size 256 --lr 1e-3 \
    --negs-per-pos 512 --shared-batch-negatives --sparse-embeddings \
    --num-epochs 30 --eval-every 2 --patience 5 \
    --eval-max-users 100000 --eval-batch-size 64 \
    --wandb-run-name yambda-5b-d64-sparse \
    2>&1 | tee -a logs/yambda5b-\$(date +%Y%m%d-%H%M%S).log"

tmux attach -t yambda5b   # to view; Ctrl+b d to detach
```

- tmux survives shell exit, SSH disconnect, VSCode close, Claude agent shutdown.
- **wandb** is on by default → live remote graphs at `wandb.ai/<user>/yambda-gsasrec`.
- Fallback if no tmux: `nohup … > logs/run.log 2>&1 & disown`.

## Verification ladder

1. **Synthetic smoke** (1K users / 50K items, no real Yambda needed) — confirms code paths run end-to-end.
2. **Yambda-50M Listen+** dense baseline vs sparse+shared-negs — NDCG@10 within ±2% rel., ≥1.5× wall-clock speedup.
3. **Yambda-500M Listen+** rerun with sparse+shared-negs — must match the published d64 baseline (NDCG@10 ≈ 0.0813) within tolerance.
4. **Numerical check**: SparseAdam moment update on a single row vs hand-computed dense AdamW reference.
5. **Yambda-5B Listen+** full run.

## Blockers / preconditions

- **Disk**: container needs ≥150 GB writable for the 500M smoke and ≥500 GB for 5B raw + checkpoints. Vast.ai default disk is too small; re-rent with a bigger `Disk Space` setting, or stream from HF Hub.
- **Data prep script**: not in repo yet. Needs to stream `5b/multi_event.parquet`, filter `event_type=='listen' AND played_ratio_pct >= 50`, group-by-uid sorted by timestamp, write the three parquets + id map. Paper temporal split: train = first 10 months, val/test = month 11.
