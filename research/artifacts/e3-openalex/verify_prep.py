"""Re-derive `openalex prep`'s query side from its inputs, independently of prep's vectorised
code: for every held-out paper, its in-catalog references (staged `refs` ∩ item_id_map), the
same-field / strictly-earlier-era flag from papers.parquet, and the target's membership.
Also checks the held-out papers are not in the catalog and papers.parquet is in item-id order.

    cd evaluation && RETRIEVE_DATA_ROOT=... PYTHONPATH=. \
        python ../docs/artifacts/e3-openalex/verify_prep.py <output-dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from eval_datasets.etl.openalex import STAGING, era_of

out = Path(sys.argv[1])
id_map = json.loads((out / "item_id_map.json").read_text())
papers = pl.read_parquet(out / "papers.parquet", columns=["item_id", "work_id", "year", "field"])
heldout = pl.read_parquet(out / "heldout.parquet")
qrels = pl.read_parquet(out / "qrels.parquet")
staged = pl.read_parquet(sorted(STAGING.glob("*.parquet")), columns=["work_id", "refs"])
refs_of = dict(zip(staged["work_id"].to_list(), staged["refs"].to_list(), strict=True))

n = papers.height
assert papers["item_id"].to_list() == list(range(1, n + 1)), "papers not in item-id order"
assert len(id_map) == n, "item_id_map size != papers"
assert all(id_map[f"W{w}"] == i for w, i in zip(papers["work_id"], papers["item_id"], strict=True))
year = dict(zip(papers["item_id"], papers["year"], strict=True))
field = dict(zip(papers["item_id"], papers["field"], strict=True))

bad = 0
for q, row in enumerate(heldout.iter_rows(named=True)):
    assert row["query_work_id"] not in id_map, f"query {q} is in the catalog"
    wid = int(row["query_work_id"][1:])
    want = {id_map[f"W{r}"] for r in refs_of[wid] if f"W{r}" in id_map}
    got = qrels.filter(pl.col("query_row") == q)
    q_era = int(era_of(np.array([row["query_year"]]))[0])
    flags = {
        i: (field[i] is not None and field[i] == row["query_field"])
        and int(era_of(np.array([year[i]]))[0]) < q_era
        for i in want
    }
    ok = (
        set(got["item_id"].to_list()) == want
        and all(flags[i] == p for i, p in got.select("item_id", "passes_field_era").iter_rows())
        and flags.get(row["item_id"], False)
        and row["n_relevant"] == len(want)
        and row["n_relevant_field_era"] == sum(flags.values())
        and row["query_era"] == q_era
    )
    bad += not ok
print(json.dumps({"n_items": n, "n_queries": heldout.height, "n_qrels": qrels.height,
                  "queries_mismatched": bad}))  # fmt: skip
sys.exit(1 if bad else 0)
