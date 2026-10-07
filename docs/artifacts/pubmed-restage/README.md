# Pubmed restage for D1-E (2026-10-07)

The 10 M slice rebuilt on a fresh A100 pod under `/data/pubmed-medcpt`, CPU only,
checked against what D1's pubmed records (`d1/pubmed`, code_version `72e5a90`) ran on.

| script | what |
|---|---|
| [`source_check.py`](source_check.py) | the NCBI MedCPT source against E2's `plan-10m.json` (Hub `artifacts/e2-pubmed`): per shard, three file sizes and the npy row count, two connections at a time |
| [`restage.sh`](restage.sh) | E2's ETL steps (`docs/artifacts/e2-pubmed/run.sh`) up to `queries`, niced on NUMA node 1, ≤ 4 NCBI connections; stops before the GPU `encode_queries` |
| [`slice_identity.py`](slice_identity.py) | every per-sweep fact a record carries (n_items, n_queries, n_kept, n_queries_oracle, n_queries_heldout, n_targets_in_filter, exact pass_rate) recomputed from the restaged attrs, split and targets, compared with every record; the counting cross-checked against `clause_subset_match` on a full 10 M scan for 16 queries per sweep |

## Result

- Source: 38 shards, 35,920,666 rows, 163,658,606,313 bytes; **0 shards differ** from the
  E2 plan (files dated 2023-10-28 on the server).
- ETL: 52 min wall (17:36-18:28 UTC), no retry; `convert` 2,884 s, `medline` 2,611 s
  (39,994,988 rows, as at E2), `attrs` 201 s; 17 GB under `pubmed-medcpt/`, 0.4 GB of PMID
  lists in `_raw/pubmed/`.
- Identity: all 8 sweeps (clause `c0_mesh`, `c2_year`, `c3_journal_reverse`, `c0c2`, `all5`;
  bloom `c0_mesh`, `c2_year`, `c0c2`) **identical** on every fact for all 44 ok `d1/pubmed`
  records (pass_rate within 1e-12 relative), and for E2's `linr_v1_filter_mask` cell record.
- Not shown: the item and query **embeddings**. The records carry no embedding digest, and the
  query embeddings need `encode_queries` (GPU). The item shards follow deterministically from
  the unchanged source and the same `--keep-items 10000000 --seed 0` id map. Rerunning one exact
  cell (V1 clause `c0_mesh`, `--skip-perf`) and comparing its quality block with the D1 record
  closes this.
- `plan`'s 16-way HEAD burst draws HTTP 503 from NCBI today and `_remote_size` reads that as
  "no Content-Length", so `plan` aborts. `source_check.py` stands in for it. The ETL was not edited.
