# eval-goodreads-work-id

Goodreads (UCSD Book Graph), work-id catalogue: 797,084 works, 313,178 test users. What the harness reads for
`config/goodreads.yaml` and `config/goodreads-synth*.yaml` (`data_dir: data/goodreads-work-id`).

- `test.parquet`: the test split, one row per user: `item_ids` (input history, dense work ids, at most 200) and
  `targets` (held-out works). The benchmark encodes the first 10,000 rows (`users_limit`).
- `item_id_map.json`: `{work_id: dense id}`, ids 1..N (0 is padding).
- `item_attrs_narrow.pt` `[N + 1, 4, 4]` int64 (row 0 padding, `-1` fill): C0 genre (top 4), C1 language (reverse
  clause), C2 format, C3 year bucket; `clause_is_reverse_narrow.pt` `[4]`; `eval_split.parquet`: per test row the
  target and the query's attribute values (`query_attrs_narrow`), aligned 1:1 with `test.parquet`.
- `lang_vocab.json`, `format_vocab.json`, `author_vocab.json`: the code tables behind the attributes (codes and source
  ids only).
- `item_attrs_synth*.pt`, `synth_u*.pt`, `query_attrs_synth.pt`, `synth_filter*.json`: the uniform synthetic filter,
  draws 0-2 (`eval-data synth-filter --dataset goodreads --draw d`; seed 20261008 + d).
- `checkpoints/sasrec-ssm-logq-d128/`: the sequential encoder that produces the queries (its own README).

Derived from the UCSD Book Graph; no raw source file, text, rating, user id or timestamp is included. See the
archive's `LICENSES.md`.
