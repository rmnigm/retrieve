# eval-pubmed

PubMed with MedCPT, a 10,000,000-article slice (`eval-data pubmed --keep-items 10000000 --seed 0`), scored by dot
product. What the harness reads for `config/pubmed.yaml` and `config/pubmed-synth*.yaml`
(`data_dir: data/pubmed-medcpt-dot`; fetch with `eval-data fetch pubmed --target data/pubmed-medcpt-dot`).

- `content_d768/`: the item matrix in 38 shards (`shard_index.json`, `text_emb_shard_*.pt`, fp16), the MedCPT article
  embeddings as NCBI ships them, not normalised; `query_emb.pt` `[10000, 768]` fp16: 10,000 held-out article titles
  through `ncbi/MedCPT-Query-Encoder` ([CLS]), not normalised; `*.meta.json`.
- `heldout.parquet`, `item_id_map.json` (`{pmid: dense id}`).
- `item_attrs_narrow.pt` `[N, 5, 4]` int64: C0 MeSH (4 rarest), C1 MeSH category, C2 year bucket, C3 journal (reverse clause),
  C4 has-abstract; `clause_is_reverse_narrow.pt`; `eval_split.parquet`; `*_vocab.json` and `mesh_global_freq.pt`
  (code tables: MeSH headings, journal abbreviations, years, languages).
- `item_attrs_synth*.pt`, `synth_u*.pt`, `query_attrs_synth.pt`, `synth_filter*.json`: the uniform synthetic filter,
  draws 0-2 (`eval-data synth-filter --dataset pubmed --draw d`).

No title, abstract or other text is included.
