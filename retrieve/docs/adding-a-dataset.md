# Benchmarking on your own dataset

The library itself takes tensors, so any data works directly in your own code (see
[`getting-started.md`](getting-started.md)). This page is for the benchmark harness in the
source repository (`evaluation/`: `bench`, `eval-data`). The harness runs the library's
modules against an exact filtered oracle and records recall, latency and memory. To add a
dataset to it, you need vectors and attribute tables. You do not need to write an ETL.

## What you provide

| file | content |
|---|---|
| item vectors | one or more `.npy` files `[n_i, D]`, fp16 or fp32; concatenated, they give items `0 … N-1` |
| item attributes | a parquet file, row `i` = item `i`, one column per attribute. A column may be a scalar of any type (a string, an int, a bucket) or a list of one (tags). A null means "no value" |
| query vectors | one `.npy` file `[U, D]`, from the same encoder (with its query prefix, if it has one) |
| query attributes | a parquet file with the same columns, one scalar per query: the value each query filters on. A null turns that filter off for that query |
| targets (optional) | a `.npy` `[U]` of 0-indexed items, the relevant item per query. Without it, each query's exact unfiltered nearest item is used |

Vectors are L2-normalised on ingest. Scores are inner products, so cosine similarity is
what is benchmarked.

## Filters: clauses and sweeps

A **clause** is one attribute test, and an item passes a query's clause when it holds the
query's value. A **reverse** clause passes the items that do *not* hold it ("other sites",
"not this author"). A **sweep** is a set of clauses ANDed together, and every sweep is one
benchmark axis. Clauses are named on the command line as `NAME[=COLUMN][!]`: `COLUMN` defaults
to `NAME`, and `!` makes the clause reverse. Two clauses may read the same column.

## Ingest

From a clone of the repository (`uv sync --all-packages` at its root):

```bash
uv run --directory evaluation eval-data ingest mydata \
  --items /path/items_000.npy --items /path/items_001.npy \
  --item-attrs /path/item_attrs.parquet \
  --queries /path/queries.npy --query-attrs /path/query_attrs.parquet \
  --clause site --clause year --clause other_site=site! \
  --sweep c0_site=site --sweep site_year=site,year --sweep c2_other=other_site \
  --device cuda
```

This writes the staged dataset under the data root (`$RETRIEVE_DATA_ROOT/mydata`), its
config `evaluation/config/mydata.yaml`, and a `prep_log.json` that lists each clause's pass
rate. It then checks the layout and exits non-zero on any problem. `--device cuda` only
speeds up the computed targets. If the encoder uses prefixes (nomic's `search_document: ` /
`search_query: `), pass `--doc-prefix` / `--query-prefix` so the sidecars record them.

## Run

`bench` runs suites, and a suite names its datasets. Add `mydata` to a suite in
`evaluation/config/suites.yaml`: to its `datasets`, its `dims` (your `D`), and its `sweeps`
map (the sweep names you gave). Then:

```bash
uv run --directory evaluation bench check --dataset mydata
uv run --directory evaluation bench run --dataset mydata --dim 256 --suite filter --algo silvertorch
```

The library needs a CUDA GPU. The items are held fp32 on the device, so the catalog must fit:
`N × D × 4` bytes plus the index (30 M × 256 is 31 GB).
