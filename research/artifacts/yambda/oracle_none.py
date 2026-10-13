"""Oracle recall for the unfiltered A5-yambda cells, after the run.

``bench run`` builds the exact oracle only for filter cells (``run.cell_assets``: ``fk != "none"``), so the
``none`` records carry held-out recall alone. This script builds the same oracle the filter cells use
(``bench.oracle.compute``: exact fp32, top-1000, lowest id on ties, no filter) per dim, scores every cell's
top-1000 ids from its ``_parity`` spill with the harness's own ``metrics.accumulate(ranked=True)``, and writes
``RESULTS_PARQUET`` with ``oracle_recall@{100,1000}`` filled to ``OUT/results_oracle.parquet`` (``corr.py``'s
input), plus the per-query oracle recall per cell in ``OUT/perquery_oracle/``.

usage: oracle_none.py CONFIG_DIR OUT_DIR(the run's --out) RESULTS_PARQUET OUT
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from bench import inputs, measure, metrics, oracle
from bench.config import load_matrix
from bench.run import parity_group

CFG, RUN_OUT, SRC, OUT = (Path(a) for a in sys.argv[1:5])
SUITE, DATASET, KS = "a5-yambda", "yambda-500m", [100, 1000]
dev = torch.device("cuda")
measure.setup(0)
(OUT / "perquery_oracle").mkdir(parents=True, exist_ok=True)

jobs = load_matrix(CFG / f"{DATASET}.yaml", CFG / "suites.yaml", SUITE)
recs = [
    json.loads(line)
    for f in sorted((RUN_OUT / SUITE).glob("*.jsonl"))
    for line in open(f)
]
rows = []
for dim in sorted({j.dim for j in jobs}):
    job = next(j for j in jobs if j.dim == dim)
    gt_path = OUT / f"oracle_none_d{dim}.pt"
    if gt_path.exists():
        topk = torch.load(gt_path)
    else:
        inp = inputs.load_inputs(job.data, dev, with_filters=False)
        topk = oracle.compute(
            inp["item_embs"], inp["queries"], None, None, None, max(KS), device=dev
        )["topk"]
        torch.save(topk, gt_path)
        del inp
    topk_d = topk.to(dev)
    for r in (r for r in recs if r["dim"] == dim and r["status"] == "ok"):
        assert r["n_kept"] == topk.shape[0], (
            "spill rows are the kept queries; A5 keeps every query"
        )
        key = {
            k: r[k]
            for k in (
                "dataset",
                "dim",
                "inputs",
                "suite",
                "filter_kind",
                "sweep",
                "algo",
                "params",
                "seed",
            )
        }
        key["params"] = {k: v for k, v in key["params"].items() if k != "score_path"}
        h = hashlib.sha1(
            json.dumps(key, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        z = np.load(
            RUN_OUT / "_parity" / parity_group(DATASET, dim, r["algo"]) / f"{h}.npz"
        )
        ids = torch.from_numpy(z["ids"]).long().to(dev)
        acc = metrics.accumulator(KS, dev)
        per_q = metrics.accumulate(acc, ids, topk_d, ranked=True)
        mean = metrics.finalize(acc)
        np.savez(
            OUT / "perquery_oracle" / f"d{dim}_{h}.npz",
            **{f"recall_oracle@{k}": v.cpu().numpy() for k, v in per_q.items()},
        )
        rows.append(
            {
                "dim": dim,
                "algo": r["algo"],
                "params": json.dumps(r["params"], sort_keys=True),
                **{f"oracle_recall@{k}": mean[f"recall@{k}"] for k in KS},
            }
        )
    del topk_d

ora = pl.DataFrame(rows)
df = pl.read_parquet(SRC).drop(
    [
        c
        for c in ("oracle_recall@100", "oracle_recall@1000")
        if c in pl.read_parquet(SRC).columns
    ]
)
df = df.with_columns(
    _p=pl.col("params").map_elements(
        lambda s: json.dumps(json.loads(s), sort_keys=True), return_dtype=pl.String
    )
)
out = df.join(ora.rename({"params": "_p"}), on=["dim", "algo", "_p"], how="left").drop(
    "_p"
)
assert out["oracle_recall@100"].null_count() == 0, "a record without its spill"
out.write_parquet(OUT / "results_oracle.parquet")
print(
    out.select(
        "dim",
        "algo",
        "params",
        "oracle_recall@100",
        "oracle_recall@1000",
        "heldout_recall@100",
        "heldout_recall@1000",
    )
)
