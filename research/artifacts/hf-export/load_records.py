"""Every JSONL record line of the fetched fr/ legs, flattened to the fields the checks need."""
import json
import sys
from pathlib import Path

import polars as pl

FR = Path("/data/work/hub/eval-results/fr")
rows = []
for leg in sorted(FR.iterdir()):
    for p in leg.rglob("*.jsonl"):
        if p.name.endswith(".samples.jsonl"):
            continue
        for line in p.open():
            r = json.loads(line)
            env = r.get("env") or {}
            rows.append({
                "leg": leg.name, "file": str(p.relative_to(leg)), "status": r.get("status"),
                "suite": r.get("suite"), "dataset": r.get("dataset"), "dim": r.get("dim"),
                "inputs": r.get("inputs"), "filter_kind": r.get("filter_kind"), "sweep": r.get("sweep"),
                "algo": r.get("algo"), "backend": r.get("backend"), "seed": r.get("seed"),
                "n_items": r.get("n_items"), "n_queries": r.get("n_queries"), "pass_rate": r.get("pass_rate"),
                "host": env.get("host"), "code_version": env.get("code_version"), "commit": env.get("commit"),
            })
df = pl.DataFrame(rows, infer_schema_length=None)
df.write_parquet("/data/work/all_records.parquet")
print(df.shape)
print(df.group_by("dataset").agg(pl.len(), pl.col("dim").unique(), pl.col("seed").unique(), pl.col("n_items").unique()).sort("dataset"))
