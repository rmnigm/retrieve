# eval-yambda-500m

Yambda 500M (`yandex/yambda`), Listen+ events (played ratio at least 50 %): 1,866,170 tracks, 45,932 test users.
What the harness reads for `config/yambda-500m.yaml` (`data_dir: data/yambda-500m`; unfiltered cells only).

- `test.parquet`: the test split the encoders were scored on: `item_ids` (history, at most 200), `timestamps`
  (seconds since Yambda's anonymised epoch) and `targets`. The benchmark encodes the first 10,000 rows.
- `item_id_map.json`: `{yambda track id: dense id}`, ids 1..N.
- `checkpoints/sasrec-ssm-logq-d{64,128,256}/`: the sequential encoders (each with its own README).
