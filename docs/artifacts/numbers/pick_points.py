"""Timed points from the quality-only curves (README.md): per (dataset, kind, sweep, n_lists) the smallest n_probe reaching
recall 0.90 and 0.95 (oracle recall@100 and @1000), the largest n_probe that ran (the cap), and the paper's 24 at the
dataset's smallest n_lists. Writes the `n-timed` and `n-st-k2048` suites, each SilverTorch arm interleaved with LiNR V1 / V2.

usage: pick_points.py CURVE_PARQUET [...] > suites-timed.yaml
"""

import json
import sys

import pandas as pd
import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


TARGETS = [0.90, 0.95]
K2048 = {"goodreads": "c0_genre", "arxiv": "c0_maincat", "pubmed": "c0_mesh"}

df = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
df = df[(df.backend == "official-fork") & (df.status == "ok")]
p = df.params.map(json.loads)
df = df.assign(
    n_lists=p.map(lambda d: d["n_lists"]), n_probe=p.map(lambda d: d["n_probe"])
)
df = df.drop_duplicates(["dataset", "filter_kind", "sweep", "n_lists", "n_probe"])

points = {}
for (ds, kind, sw, nl), g in df.groupby(["dataset", "filter_kind", "sweep", "n_lists"]):
    g = g.sort_values("n_probe")
    pick = {int(g.n_probe.max())}
    for col in ("oracle_recall@100", "oracle_recall@1000"):
        for t in TARGETS:
            ok = g[g[col] >= t]
            if len(ok):
                pick.add(int(ok.n_probe.iloc[0]))
    if nl == df[df.dataset == ds].n_lists.min():
        pick.add(24)
    points[ds, kind, sw, int(nl)] = sorted(pick)

datasets = sorted({k[0] for k in points})
V12 = [
    {"algo": "linr_v1_filter_mask", "backends": ["triton"]},
    {"algo": "linr_v2", "backends": ["triton"]},
]


def arms(keys):
    out = list(V12)
    for ds, kind, sw, nl in keys:
        out.append(
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "filter_kinds": [kind],
                "build": {"n_lists": [nl]},
                "datasets": {
                    ds: {"sweeps": [sw], "query": {"n_probe": points[ds, kind, sw, nl]}}
                },
            }
        )
    return out


def suite(dss, keys, ks):
    sweeps = {d: sorted({k[2] for k in keys if k[0] == d}) for d in dss}
    kinds = sorted({k[1] for k in keys})
    return {
        "datasets": dss,
        "dims": [128, 192, 256, 768],
        "filter_kinds": kinds,
        "ks": ks,
        "batch_sizes": [1, 16, 64],
        "seeds": [0],
        "sweeps": sweeps,
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms(keys),
    }


out = {"n-timed": suite(datasets, sorted(points), [100, 1000])}
k2 = [k for k in sorted(points) if K2048.get(k[0]) == k[2] and k[1] == "clause"]
if k2:
    out["n-st-k2048"] = suite(sorted({k[0] for k in k2}), k2, [2048])
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
n = sum(len(v) for v in points.values())
print(
    f"# {len(points)} (dataset, kind, sweep, n_lists) groups, {n} timed SilverTorch points",
    file=sys.stderr,
)
