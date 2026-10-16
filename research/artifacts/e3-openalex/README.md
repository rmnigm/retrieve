# e3-openalex: the OpenAlex ETL (out of the study)

Scripts behind the OpenAlex 10 M build ([datasets](../../../docs/system/datasets.md#openalex)).

## Budget

As measured; `plan` from its
[report](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/e3-openalex/plan-2026-09-23.json):

| | 10 M catalog |
|---|---|
| read over S3 | **297.1 GB** of projected columns (all files; the slice is a hash, not a prefix) |
| `convert` wall time | **572 s** at `--workers 64`, 526 MB/s average (the 20 s probes of [`stream_probe.py`](stream_probe.py) saw 176–230 MB/s; `plan`'s threaded rate, 20 MB/s, is GIL-bound) |
| sample rate | 0.35 → 29.9 M staged; any rate that stages N × (1 + margin) gives the same catalog |
| staging parquet | 16 GB (`_raw/openalex/staging`, deletable once no smaller catalog is wanted) |
| `prep` | 374 s; `papers.parquet` 6.3 GB, `item_id_map.json` 209 MB |
| fp16 item shards | 15 GB (10 × 1 M) |
| attrs | 1.5 GB (`[10 M, 5, 4]` int64) |
| encode | a fresh encode is ~2.9 A100-hours at 958 docs/s (the 15 M one took 15,663 s); this catalog was `reshard`ed from it in 34 s |
| on the device | items fp32 30.7 GB; the cell (run before the item-chunked oracle) reserved 64 GB |

**What fits the harness.** `load_inputs` holds items fp32 on the device: N × D × 4 B plus
attrs and workspace. When the 15 M catalog was built, the then one-shot oracle added a
transposed fp32 copy, which capped 768-d at ~11–12 M items on the 80 GB A100, so E3 was cut
to 10 M (`prep --keep-items 10000000`, then `reshard --from-dir` the 15 M vectors). The
item-chunked oracle has no such copy ([evaluation](../../../docs/system/evaluation.md#oracle-blob-v4)). The
loader itself once peaked at three copies (fp16 + `.float()` + `F.normalize`, 107 GB at
15 M); `layout.load_text_items` normalises a sharded matrix shard by shard into one fp32
buffer (`load_sharded(..., normalize=True)`, `torch.equal` to the old path on 3 M real
rows), so a large catalog is written sharded.

```
data/openalex/
├── item_id_map.json            {"W<id>": item_id}
├── papers.parquet              item-id order: work_id, year, field, subfield, source, is_oa, type, title, abstract, refs
├── heldout.parquet             target item_id + the query's own attributes and relevance counts
├── queries.parquet             query_work_id, title, abstract (heldout row order)
├── qrels.parquet               query_row, item_id, passes_field_era
├── content_d768/
│   ├── text_emb_shard_NNN.pt + shard_index.json + text_emb.meta.json
│   ├── query_emb.pt + query_emb.meta.json
│   └── encode_params.json
├── item_attrs_narrow.pt        [N, 5, 4] int64
├── clause_is_reverse_narrow.pt [5] bool = [F, F, F, T, F]
├── {field,subfield,source,era}_vocab.json
├── eval_split.parquet          target_id, query_attrs_narrow [5]
└── prep_log.json
```

