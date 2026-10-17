"""Every LAION record of the final run (fetched fr/ legs), flattened to what the rebuild is checked
against: counts, pass rates, quality and the result ids' sha256 per timed variant.

Usage: records.py FR_DIR OUT.parquet"""

import json
import sys
from pathlib import Path

import polars as pl


def rows_of(root: str) -> pl.DataFrame:
    rows = []
    for p in sorted(Path(root).rglob("laion*-d256.jsonl")):
        for line in p.open():
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            q = r.get("quality") or {}
            base = {
                "leg": p.relative_to(root).parts[0], "file": str(p), "suite": r["suite"],
                "dataset": r["dataset"], "filter_kind": r["filter_kind"], "sweep": r["sweep"],
                "algo": r["algo"], "backend": r["backend"], "params": json.dumps(r["params"], sort_keys=True),
                "seed": r["seed"], "n_items": r["n_items"], "n_queries": r["n_queries"], "n_kept": r["n_kept"],
                "n_queries_heldout": r["n_queries_heldout"], "n_targets_in_filter": r["n_targets_in_filter"],
                "pass_rate": r["pass_rate"], "per_query": r.get("per_query"),
                "oracle_recall": (q.get("oracle") or {}).get(f"recall@{r['k_max']}"),
                "heldout_recall": (q.get("heldout") or {}).get(f"recall@{r['k_max']}"),
            }  # fmt: skip
            for v in r.get("perf") or [{}]:
                rows.append({**base, "k": v.get("k"), "bs": v.get("bs"), "mode": v.get("mode"),
                             "ids_sha256": v.get("ids_sha256")})  # fmt: skip
    return pl.DataFrame(rows, infer_schema_length=None)


if __name__ == "__main__":
    df = rows_of(sys.argv[1])
    df.write_parquet(sys.argv[2])
    print(df.shape)
    print(
        df.group_by("dataset")
        .agg(pl.len(), pl.col("n_items").unique(), pl.col("algo").unique())
        .sort("dataset")
    )
