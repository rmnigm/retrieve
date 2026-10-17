# eval-arxiv-papers

arXiv (from `open-index/open-arxiv`, snapshot 2026-03-24): 2,988,996 papers with a non-empty abstract, 10,000 held-out
self-queries. What the harness reads for `config/arxiv.yaml`, `config/arxiv-synth*.yaml` and
`config/arxiv-corr-synth*.yaml` (`data_dir: data/arxiv-papers`).

- `content/` (d256) and `content_d128/`: `text_emb.pt` `[N + 1, D]` fp16 (row 0 padding) and `query_emb.pt`
  `[10000, D]` fp16, `nomic-ai/nomic-embed-text-v1.5` on "title. abstract" (Matryoshka, L2-normalised; d128 is the
  d256 vector sliced and renormalised), each with its `*.meta.json`.
- `heldout.parquet`: `(item_id, arxiv_id)` of the 10,000 queries; `item_id_map.json`: `{arxiv_id: dense id}`.
- `item_attrs_narrow.pt` `[N + 1, 4, 4]` int64 (row 0 padding, `-1` fill): C0 main category, C1 licence bucket,
  C2 update-year bucket, C3 version-count bucket; `clause_is_reverse_narrow.pt` (no reverse clause);
  `eval_split.parquet` (query attributes); `cat_main_vocab.json`, `license_vocab.json`, `author_vocab.json` (code tables).
- `item_attrs_synth*.pt`, `synth_u*.pt`, `query_attrs_synth.pt`, `synth_filter*.json`: the uniform synthetic filter,
  draws 0-2 (`eval-data synth-filter --dataset arxiv --draw d`).
- `item_attrs_corr.pt`, `query_attrs_corr.pt`, `synth_corr.json`: the cluster-correlated synthetic filter, draw 0
  (`eval-data synth-filter --dataset arxiv --correlated --dim 128`); `synth_corr.json` holds the centroids. Regenerated
  for this archive; see the archive README on how it compares with the run's copy.

No title, abstract or other text is included.
