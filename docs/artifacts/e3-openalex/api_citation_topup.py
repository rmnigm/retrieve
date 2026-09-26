"""Dry-run only: make a prefix-scale OpenAlex staging dir citation-connected.

A few thousand snapshot rows almost never cite each other, so `prep` finds no held-out paper
with an in-catalog reference. This script picks ~60 staged papers that will land in the
held-out pool (work-id hash above the 90th percentile of the staged hashes), pulls the works
they reference from the OpenAlex API (`filter=cited_by:W…`, ~$0.0001 per call), runs them
through the same `stage_table` as the snapshot rows, and writes them as one more staging file.
It prints the `--keep-items` that keeps the pool at the hashes above that percentile.

    cd evaluation && RETRIEVE_DATA_ROOT=/scratch/e3-dryrun \
        python ../docs/artifacts/e3-openalex/api_citation_topup.py
"""

import json
import sys
import time
import urllib.request

import numpy as np
import polars as pl
import pyarrow as pa

from eval_datasets.etl.openalex import STAGING, stage_table
from eval_datasets.etl.pubmed import pmid_hash

SELECT = (
    "id,publication_year,language,type,is_paratext,is_retracted,is_xpac,primary_topic,"
    "primary_location,open_access,title,abstract_inverted_index,referenced_works"
)
N_QUERIES = 60


def projected(rec: dict) -> dict:
    """An API work → the snapshot's projected parquet row (abstract index as a JSON string)."""
    topic = rec.get("primary_topic") or {}
    source = (rec.get("primary_location") or {}).get("source") or {}
    inv = rec.get("abstract_inverted_index")
    return {
        "id": rec["id"],
        "publication_year": rec.get("publication_year"),
        "language": rec.get("language"),
        "type": rec.get("type"),
        "is_paratext": rec.get("is_paratext"),
        "is_retracted": rec.get("is_retracted"),
        "is_xpac": rec.get("is_xpac"),
        "primary_topic": {
            "field": {"id": (topic.get("field") or {}).get("id")},
            "subfield": {"id": (topic.get("subfield") or {}).get("id")},
        },
        "primary_location": {"source": {"id": source.get("id")}},
        "open_access": {"is_oa": (rec.get("open_access") or {}).get("is_oa")},
        "title": rec.get("title"),
        "abstract_inverted_index": json.dumps(inv) if inv else None,
        "referenced_works": rec.get("referenced_works") or [],
    }


def references(work_id: int) -> list[dict]:
    out, page = [], 1
    while True:
        url = (
            f"https://api.openalex.org/works?filter=cited_by:W{work_id}&select={SELECT}"
            f"&per-page=100&page={page}"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            body = json.loads(r.read())
        out.extend(body["results"])
        if page * 100 >= body["meta"]["count"]:
            return out
        page += 1
        time.sleep(0.2)


staged = pl.read_parquet(sorted(STAGING.glob("*.parquet")))
params = json.loads((STAGING / "params.json").read_text())
h = pmid_hash(staged["work_id"].to_numpy(), params["seed"])
cut = np.quantile(h.astype(np.float64), 0.9)
cand = (
    staged.with_columns(pl.Series("h", h.astype(np.float64)))
    .filter(
        (pl.col("h") > cut)
        & (pl.col("refs").list.len() >= 3)
        & (pl.col("year") >= 2010)
        & pl.col("field").is_not_null()
    )
    .sort("h")
    .head(N_QUERIES)
)
print(f"{cand.height} pool candidates with >= 3 references", flush=True)
have = set(staged["work_id"].to_list())
rows: dict[str, dict] = {}
for wid in cand["work_id"].to_list():
    for rec in references(wid):
        if int(rec["id"].rsplit("W", 1)[1]) not in have:
            rows[rec["id"]] = projected(rec)
    time.sleep(0.2)
print(f"{len(rows)} referenced works fetched", flush=True)
topup, stats = stage_table(
    pa.Table.from_pylist(list(rows.values())),
    max_year=params["max_year"],
    sample_rate=params["sample_rate"],
    seed=params["seed"],
)
print(json.dumps(stats), flush=True)
topup.write_parquet(STAGING / "9999-99-99_api_topup.parquet", compression="zstd")
all_h = pmid_hash(
    np.concatenate([staged["work_id"].to_numpy(), topup["work_id"].to_numpy()]), params["seed"]
)
print(f"--keep-items {int((all_h.astype(np.float64) <= cut).sum())}", flush=True)
sys.exit(0)
