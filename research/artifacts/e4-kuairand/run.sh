# E4 GPU half, A100 box, 2026-09-26. From evaluation/, data root /data (evaluation/data -> /data).
# negs-per-pos 256 (the default) OOMs on the first backward: probe-negs256-oom.log.
PYTORCH_ALLOC_CONF=expandable_segments:True uv run train sasrec --data-dir data/kuairand \
    --checkpoint-dir data/kuairand/checkpoints/gsasrec-d128-shared \
    --embedding-dim 128 --dropout 0.5 --reuse-item-embeddings --eval-max-users 4096 \
    --negs-per-pos 128 --patience 5 --num-epochs 100 --no-wandb
# hub.py's ignore list would also push _resume.pt (49 GB): publish from a dir of hard links.
P=data/kuairand/_publish/gsasrec-d128-shared; mkdir -p $P
for f in best_model.pt config.json eval_quality.json item_embs.pt item_id_map.json train_metrics.json; do
    ln -f data/kuairand/checkpoints/gsasrec-d128-shared/$f $P/$f; done
uv run eval-data publish-checkpoint kuairand gsasrec-d128-shared --source $P
uv run bench run --dataset kuairand --dim 128 --suite filter --algo linr_v1_filter_mask \
    --backend triton --filter-kind clause --sweep t_cat1 --mode eager --skip-perf \
    --output ../docs/artifacts/e4-kuairand/filter-kuairand-d128-cell.jsonl
