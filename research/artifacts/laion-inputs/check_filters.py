"""The rebuilt LAION inputs against the final run's records, on CPU.

Real filters: per query, the pass count from item_attrs_narrow.pt + clause_is_reverse_narrow.pt +
eval_split.parquet (clause_subset_match semantics) equals the run's per-query sidecar pass_count;
every query on single-clause sweeps, SAMPLE on multi-clause ones. Every record: n_items, n_kept,
n_targets_in_filter and pass_rate equal. Synthetic: synth_filter*.json pass count / n_items equals
every synth record's pass_rate.

Usage: check_filters.py RECORDS.parquet DATA_ROOT CONFIG_DIR"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from eval_datasets import layout

torch.set_num_threads(64)
recs = pl.read_parquet(sys.argv[1]).unique(
    ["file", "sweep", "filter_kind", "algo", "params", "n_kept"]
)
root, cfg_dir = Path(sys.argv[2]), Path(sys.argv[3])
SAMPLE = 500
SYNTH = ["p0001", "p0003", "p001", "p003", "p005", "p01", "p02", "p03", "p05", "p1"]
bad = 0

for name in ["laion1m", "laion3m", "laion10m", "laion30m"]:
    d = root / name
    cfg = yaml.safe_load((cfg_dir / f"{name}.yaml").read_text())
    r = recs.filter(pl.col("dataset") == name)
    n_items = r["n_items"].unique().to_list()
    attrs, rev = layout.load_item_attrs(
        d / "item_attrs_narrow.pt",
        d / "clause_is_reverse_narrow.pt",
        n_items[0],
        torch.device("cpu"),
    )
    qa = torch.tensor(
        pl.read_parquet(d / "eval_split.parquet")["query_attrs_narrow"].to_list()
    )
    sweeps = {**cfg["filters"]["clause"], **cfg["filters"].get("bloom", {})}
    for (sweep,), g in r.filter(pl.col("filter_kind") != "none").group_by(["sweep"]):
        clauses = sweeps[sweep]
        row = g.filter(pl.col("per_query").is_not_null()).row(0, named=True)
        z = np.load(Path(row["file"]).parent.parent / row["per_query"])
        rows, got = torch.from_numpy(z["rows"]).long(), z["pass_count"]
        idx = (
            np.arange(len(rows))
            if len(clauses) == 1
            else np.linspace(0, len(rows) - 1, SAMPLE).astype(int)
        )
        want = []
        if len(clauses) == 1:
            c = clauses[0]
            a = attrs[:, c, 0]
            qv = qa[rows, c]
            bc = torch.bincount(a[a >= 0], minlength=int(max(a.max(), qv.max())) + 2)
            got_all = torch.where(qv >= 0, bc[qv.clamp(min=0)], n_items[0])
            want = (n_items[0] - got_all if bool(rev[c]) else got_all).tolist()
        for i in idx if len(clauses) > 1 else []:
            q, ok = qa[rows[i]], torch.ones(n_items[0], dtype=torch.bool)
            for c in clauses:
                if q[c] == -1:
                    continue
                m = (attrs[:, c, :] == q[c]).any(dim=1)
                ok &= ~m if bool(rev[c]) else m
            want.append(int(ok.sum()))
        diff = int((np.array(want) != got[idx]).sum())
        stats = g.select("n_kept", "n_targets_in_filter", "pass_rate").unique()
        bad += diff > 0 or stats.height != 1
        print(
            f"{name} {sweep:18s} {len(idx)} queries, {diff} pass counts differ; "
            f"{g.height} records, distinct (n_kept, n_targets_in_filter, pass_rate): {stats.rows()}",
            flush=True,
        )
    print(
        f"{name}: n_items in records {n_items}, attrs rows {attrs.shape[0]}", flush=True
    )
    bad += n_items != [attrs.shape[0]]

synth = recs.filter(
    pl.col("dataset").str.contains("-synth") & pl.col("pass_rate").is_not_null()
)
n = 0
for row in (
    synth.select("dataset", "sweep", "pass_rate", "n_items")
    .unique()
    .iter_rows(named=True)
):
    base, _, draw = row["dataset"].partition("-synth")
    side = (
        root
        / base
        / ("synth_filter.json" if not draw else f"synth_filter_s{draw[2:]}.json")
    )
    meta = json.loads(side.read_text())
    want = meta["pass_counts"][SYNTH.index(row["sweep"])]
    n += 1
    if (
        round(row["pass_rate"] * row["n_items"]) != want
        or meta["n_items"] != row["n_items"]
    ):
        bad += 1
        print("SYNTH MISMATCH", row, want)
print(f"synth: {n} (dataset, sweep) pass rates compared")
print(f"TOTAL problems: {bad}")
