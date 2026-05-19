# Yambda HF migration (deferred follow-up)

## Context

The HF I/O refactor (per-dataset `pinkmeme/eval-<dataset>` repos, unified
`data_root()`, `eval-fetch`/`eval-publish`/`eval-publish-checkpoint` CLIs) is
complete for arxiv and goodreads. Yambda was deferred because nothing was
local on the migration host — neither the preprocessed eval data under
`data/yambda-{500m,5b}/` nor the SASRec checkpoints under
`evaluation/checkpoints/`. The standalone per-checkpoint HF repos still exist
under `pinkmeme/gsasrec-*-listens-*` and are now orphaned by the new layout.

The refactor's code paths are already in place for yambda:

- Registry entries `yambda-500m → pinkmeme/eval-yambda-500m` and
  `yambda-5b → pinkmeme/eval-yambda-5b` in
  [data/hf_io.py](evaluation/data/hf_io.py).
- All hydra configs in [conf/500m/](evaluation/conf/500m/) and
  [conf/5b/](evaluation/conf/5b/) point at the new logical paths
  (`data/yambda-500m`, `data/yambda-5b`,
  `data/yambda-{500m,5b}/checkpoints/<ckpt-id>/best_model.pt`).
- Raw upstream HF reads (`yandex/yambda`) already route through
  `hf_io.download_raw_file("yambda", ...)` → `data/_raw/yambda/`.

What remains is data movement, not code.

## Orphaned standalone HF repos to fold in

```
pinkmeme/gsasrec-500m-listens-d64-drop0.5      → checkpoints/gsasrec-d64-drop0.5/
pinkmeme/gsasrec-500m-listens-d128-drop0.5     → checkpoints/gsasrec-d128-drop0.5/
pinkmeme/gsasrec-500m-listens-d256-drop0.5     → checkpoints/gsasrec-d256-drop0.5/
pinkmeme/gsasrec-500m-listens-v1               → checkpoints/gsasrec-v1/   (legacy?)
pinkmeme/gsasrec-5b-listens-d64-v4             → checkpoints/gsasrec-d64-v4/
pinkmeme/gsasrec-5b-listens-d128-v5            → checkpoints/gsasrec-d128-v5/
```

`v1`, `v4`, `v5` suffixes deviate from the `drop0.5[-id]` convention used in
the existing 500m d{64,128,256} configs. Decide before migration whether to
keep them as separate ckpt-ids or consolidate.

The hydra configs assume:

- `conf/500m/d{64,128,256}-quality.yaml` →
  `gsasrec-d{64,128,256}-drop0.5` (so `pinkmeme/gsasrec-500m-listens-d{64,128,256}-drop0.5` map cleanly).
- `conf/5b/d64-quality.yaml` → `gsasrec-d64`
- `conf/5b/d128-quality.yaml` → `gsasrec-d128`

The 5b configs reference `gsasrec-d{64,128}` but the standalone repos are
`gsasrec-5b-listens-d64-v4` and `gsasrec-5b-listens-d128-v5`. Either
update the configs to match the suffix or strip `-v4`/`-v5` when uploading.

## Steps (when there is a host with yambda data + checkpoints)

1. **Build eval-ready datasets locally** with the existing CLIs:
   ```
   uv run yambda prep --variant 500m --output-dir data/yambda-500m
   uv run yambda prep --variant 5b   --output-dir data/yambda-5b
   ```
   These now write directly into the new logical dirs and pull raw via
   `hf_io.download_raw_file("yambda", ...)` → `data/_raw/yambda/`.
   `train.parquet`, `val.parquet`, `test.parquet`, `item_id_map.json` land in
   the right place; the eval upload's ignore_patterns drop train/val.

2. **Compute filter-bench artifacts** (item_attrs_narrow, eval_split,
   clause_is_reverse_narrow). There's no yambda equivalent of
   `arxiv.py attrs` / `goodreads.py attrs` today — confirm whether yambda
   filter sweeps are in scope. If not, the eval repos hold only quality
   artifacts (test.parquet + item_id_map.json + checkpoints).

3. **Pull each orphaned checkpoint repo and re-upload under the new layout**.
   Quickest path is `huggingface_hub.snapshot_download` → local rename →
   `eval-publish-checkpoint`:

   ```
   uv run python -c "
   from huggingface_hub import snapshot_download
   from data.hf_io import ckpt_dir
   pairs = [
       ('500m', 'gsasrec-500m-listens-d64-drop0.5',  'gsasrec-d64-drop0.5'),
       ('500m', 'gsasrec-500m-listens-d128-drop0.5', 'gsasrec-d128-drop0.5'),
       ('500m', 'gsasrec-500m-listens-d256-drop0.5', 'gsasrec-d256-drop0.5'),
       ('5b',   'gsasrec-5b-listens-d64-v4',         'gsasrec-d64'),
       ('5b',   'gsasrec-5b-listens-d128-v5',        'gsasrec-d128'),
   ]
   for variant, src_repo, ckpt_id in pairs:
       dst = ckpt_dir(f'yambda-{variant}', ckpt_id)
       snapshot_download(repo_id=f'pinkmeme/{src_repo}', repo_type='model',
                         local_dir=str(dst))
   "

   for ckpt in gsasrec-d64-drop0.5 gsasrec-d128-drop0.5 gsasrec-d256-drop0.5; do
       uv run eval-publish-checkpoint yambda-500m "$ckpt"
   done
   for ckpt in gsasrec-d64 gsasrec-d128; do
       uv run eval-publish-checkpoint yambda-5b "$ckpt"
   done
   ```

   Decide what to do with `pinkmeme/gsasrec-500m-listens-v1` (legacy, no
   matching config) — likely skip.

4. **Upload eval inputs** for both variants:
   ```
   uv run eval-publish yambda-500m --dry-run    # sanity-check
   uv run eval-publish yambda-500m
   uv run eval-publish yambda-5b
   ```

5. **Smoke-test eval against the new layout**:
   ```
   uv run evaluate --config conf/500m/d64-quality.yaml
   uv run evaluate --config conf/5b/d64-quality.yaml
   ```
   Compare NDCG/Recall to a pre-migration baseline.

6. **Delete the orphaned standalone repos** (manual, with user confirmation):
   ```
   for r in gsasrec-500m-listens-d{64,128,256}-drop0.5 \
            gsasrec-5b-listens-d64-v4 gsasrec-5b-listens-d128-v5; do
       hf repo delete pinkmeme/$r --yes
   done
   ```
   Hold off on `gsasrec-500m-listens-v1` until its provenance is checked.

## Open questions

- Does yambda have filter-bench artifacts (item_attrs_narrow etc.)? If not,
  the per-dataset repo schema for yambda is just `test.parquet +
  item_id_map.json + checkpoints/`. The 500m/5b configs are quality-only
  today (no `filters:` block), so this is consistent.
- 5b checkpoints carry `-v4`/`-v5` suffixes. Drop or keep? Configs assume
  dropped (`gsasrec-d64`, `gsasrec-d128`). If keeping, update both configs
  and the rename map above.
- Does `gsasrec-500m-listens-v1` need to be preserved anywhere?

## Critical files (no edits expected — code is already in place)

- [data/hf_io.py](evaluation/data/hf_io.py) — registry + helpers.
- [conf/500m/](evaluation/conf/500m/), [conf/5b/](evaluation/conf/5b/) — configs.
- [data/yambda.py](evaluation/data/yambda.py) — raw download path.
