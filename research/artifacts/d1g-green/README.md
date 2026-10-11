# d1g-green: the batching gate's companion check

D1-G's `median_ms(bs=16) < 16 × median_ms(bs=1)` fails on 25 entries of the final pass. The user declared D1-G green
with those entries listed and explained ([validation](../../validation.md)). The script compares each failing entry's
bs 16 batch time against the bs 1 time of the same queries, from the stored records only (CPU, no rerun).

| file | what |
|---|---|
| [`batch_vs_sum.py`](batch_vs_sum.py) | per failing entry: each kept bs 16 call against the sum of the kept bs 1 calls on the same 16 queries, where the stored windows allow it, and for every entry the expectation form, mean bs 16 / (16 × mean bs 1) |

**Mechanism.**
- The pools are rebuilt from the harness's own rules: `inputs.query_pool`'s seeded `randint` prefix and
  `measure.latency_group`'s counter (warmup, calibration, median window).
- The reconstruction is checked against the data. bs 1 per-call ms correlates with the query's pass count at the
  rebuilt offset (r = -0.30 to -0.40) and not at shifted ones (|r| ≤ 0.05).

**What the records lack.** The samples keep only each entry's median window: 1,000 consecutive pool entries at bs 1,
and calls at bs 16 that mostly fall on batches past the bs 1 pool. So the same-16-queries comparison exists for 2 of
the 25 entries. The other 23 would need all three windows' per-call samples, or a bs 1 pass over the bs 16 batches'
queries (a rerun).

Output: Hub `artifacts/d1g-green/batch-vs-sum-20261011` ([hub-index](../hub-index.md)). Run:
`batch_vs_sum.py MERGED_TREE CHECKS.csv OUT.csv` over `campaign-final/merged-4016` and its `checks.csv`.
