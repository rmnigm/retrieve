"""k-means test result table (bloom-runner): per (variant, n_lists, filter, sweep) cluster balance, recall@100 at the probe cap (n_lists/4)
and the smallest n_probe reaching oracle recall@100 0.90 / 0.95; random-10 = A1-laion30m (clause) and A1-bloom laion30m (bloom, recheck off),
reused per the reuse check (pass). usage: km_table.py KM_DIR A1_LAION30M_PARQUET A1_BLOOM_LAION30M_JSONL"""

import json
import sys
from pathlib import Path
import pandas as pd

km, a1, b1 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = []


def latest(path):
    last = {}
    for line in open(path):
        r = json.loads(line)
        last[
            json.dumps([r["filter_kind"], r["sweep"], r["params"]], sort_keys=True)
        ] = r
    return [r for r in last.values() if r["status"] == "ok"]


for s in ["km-4096", "km-16384"]:
    for r in latest(km / s / "laion30m-d256.jsonl"):
        p = r["params"]
        rows.append(
            dict(
                init=p["kmeans_init"],
                n_iter=p["n_iter"],
                n_lists=p["n_lists"],
                filter=r["filter_kind"],
                sweep=r["sweep"],
                n_probe=p["n_probe"],
                r100=r["quality"]["oracle"]["recall@100"],
            )
        )
d = pd.read_parquet(a1)
d = d[
    (d.dataset == "laion30m") & (d.filter_kind == "clause") & (d.status == "ok")
].drop_duplicates(["sweep", "params"])
for _, x in d.iterrows():
    p = json.loads(x.params)
    if p["n_lists"] in (4096, 16384):
        rows.append(
            dict(
                init="random",
                n_iter=10,
                n_lists=p["n_lists"],
                filter="clause",
                sweep=x.sweep,
                n_probe=p["n_probe"],
                r100=x["oracle_recall@100"],
            )
        )
for r in latest(b1):
    p = r["params"]
    if p["n_lists"] in (4096, 16384):
        rows.append(
            dict(
                init="random",
                n_iter=10,
                n_lists=p["n_lists"],
                filter="bloom",
                sweep=r["sweep"],
                n_probe=p["n_probe"],
                r100=r["quality"]["oracle"]["recall@100"],
            )
        )
df = pd.DataFrame(rows)
bal = {}
for f in (km / "centroids").glob("*.json"):
    j = json.loads(f.read_text())
    bal[j["init"], j["n_iter"], j["n_lists"]] = j
out = []
for (init, it, nl, fk, sw), g in df.groupby(
    ["init", "n_iter", "n_lists", "filter", "sweep"]
):
    g = g.sort_values("n_probe")
    cap = nl // 4

    def pick(t, g=g):
        return g[g.r100 >= t].n_probe.min() if (g.r100 >= t).any() else None

    b = bal[init, it, nl]
    out.append(
        dict(
            variant=f"{init}-{it}",
            n_lists=nl,
            filter=fk,
            sweep=sw,
            fit_s=round(b["fit_s"]),
            max_med=round(b["max_over_median"], 2),
            cv=round(b["cv"], 3),
            empty=b["empty"],
            r100_cap=round(float(g[g.n_probe == cap].r100.iloc[0]), 4)
            if (g.n_probe == cap).any()
            else None,
            np_090=pick(0.90),
            np_095=pick(0.95),
        )
    )
t = pd.DataFrame(out).sort_values(["n_lists", "filter", "sweep", "variant"])
ref = t[t.variant == "random-10"].set_index(["n_lists", "filter", "sweep"])


def flip(r):
    x = ref.loc[(r.n_lists, r["filter"], r.sweep)]
    return (
        "".join(
            f" {lab} {'reached' if pd.notna(r[c]) else 'not'} vs {'reached' if pd.notna(x[c]) else 'not'}"
            for c, lab in (("np_090", "0.90"), ("np_095", "0.95"))
            if pd.notna(r[c]) != pd.notna(x[c])
        )
        or "-"
    )


t["verdict_flip_vs_random10"] = t.apply(flip, axis=1)
print(t.to_markdown(index=False))
