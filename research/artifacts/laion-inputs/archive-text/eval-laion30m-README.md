# eval-laion30m

Re-LAION (from `laion/relaion2B-en-research-safe`, parts 0-1, 32,776,409 rows): 30,000,000 items, one per distinct
caption, and 10,000 held-out caption queries drawn from the 928,288 distinct captions left out of the catalogue.
What the harness reads for `config/laion30m.yaml` and `config/laion30m-synth*.yaml` (`data_dir: data/laion30m`),
and the parent of the 1 M / 3 M / 10 M subsets.

- `content_d256/`: `text_emb_shard_000.pt` … `_029.pt` (1 M rows each, `[N, 256]` fp16, `shard_index.json`) and
  `query_emb.pt` `[10000, 256]` fp16: `nomic-ai/nomic-embed-text-v1.5` on the caption with the prefixes
  `search_document: ` / `search_query: `, bf16 weights, `max_seq_length` 128, the model card's Matryoshka recipe
  (layer norm over the 768 dims, the first 256 kept, L2-normalised). The `*.meta.json` sidecars hold the encode
  parameters and the model revisions.
- `item_attrs_narrow.pt` `[N, 6, 1]` int64 (`-1` = missing): C0 registered domain of the image URL, C1 image size
  bucket, C2 CLIP image-caption similarity bucket, C3 the domain again as a reverse clause, C4 watermark-probability
  bucket, C5 unsafe-probability bucket; `clause_is_reverse_narrow.pt`; `eval_split.parquet` (each query's own tags,
  coded with the items' vocabulary, and its target: the exact unfiltered top-1 item); `heldout.parquet` (the
  targets); `vocab.json` (code tables: the 1,430,979 registered domains and the bucket edges).
- `item_attrs_synth*.pt`, `synth_u*.pt`, `query_attrs_synth.pt`, `synth_filter*.json`: the uniform synthetic filter,
  draws 0-2 (`eval-data synth-filter --dataset laion30m-synth --draw d`).
- `subsets/laion{1,3,10}m.MANIFEST.sha256`: the sha256 of every file of the 1 M / 3 M / 10 M subsets, which are not
  shipped. `eval-data subset laion30m laion1m --n-items 1000000` (seed 20261013; likewise `laion3m`, `laion10m`)
  rebuilds each from this directory, item rows, attributes and synthetic draws byte for byte, the targets recomputed
  over the subset (checked for 1 M from this directory: every tensor and parquet file equal to the manifest). The
  JSON sidecars that name paths or the parent directory (`prep_log.json`, the two `*.meta.json`, and
  `synth_filter*.json`'s `subset_of` when the parent is not staged as `laion30m`) differ.

No caption, URL or image is included. Rebuilt with the code release's ETL (`eval-data laion`); see the archive
README for how it was checked against the final run's records.
