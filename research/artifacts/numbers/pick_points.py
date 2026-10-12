"""Timed points from the quality-only curves (README.md): per (dataset, dim, kind, sweep, n_lists) the smallest n_probe
reaching recall 0.90 and 0.95 (oracle recall@100 and @1000), the largest n_probe that ran (the cap), and the paper's 24 at
the smallest n_lists of that (dataset, dim). Curves split over suites (n-curve-q + n-curve-q-wide) merge by
(dataset, dim, kind, sweep, n_lists, n_probe); a group whose curve stops short of the cap n_lists / 4 is skipped.

Writes one `n-timed-d<dim>` suite per dim (a dataset config may list several dims, so a suite pinned to one dim is the only
way to time exactly the picked dim) and `n-st-k2048-d<dim>` at each k-2048 dataset's primary dim; every SilverTorch arm is
interleaved with LiNR V1 / V2. A leg runs `--suite n-timed-d<dim> --dataset <ds>`.

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
K2048 = {
    ("goodreads", 128): "c0_genre",
    ("arxiv", 128): "c0_maincat",
    ("pubmed", 768): "c0_mesh",
}
V12 = [
    {"algo": "linr_v1_filter_mask", "backends": ["triton"]},
    {"algo": "linr_v2", "backends": ["triton"]},
]

df = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
CURVES = (
    "n-curve-q",
    "n-curve-q-wide",
    "n-curve-q-all",
    "n-curve-q-laion-small",
    "n-curve-q-laion-small-wide",
)  # not n-curve-q-yfcc-cap (quality only)
df = df[(df.backend == "official-fork") & (df.status == "ok") & df.suite.isin(CURVES)]
p = df.params.map(json.loads)
df = df.assign(
    n_lists=p.map(lambda d: d["n_lists"]), n_probe=p.map(lambda d: d["n_probe"])
)
df = df.drop_duplicates(
    ["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"]
)
smallest = df.groupby(["dataset", "dim"]).n_lists.min()

points = {}
for (ds, dim, kind, sw, nl), g in df.groupby(
    ["dataset", "dim", "filter_kind", "sweep", "n_lists"]
):
    g = g.sort_values("n_probe")
    if g.n_probe.max() < nl // 4:  # a split curve whose wide half has not landed yet
        print(
            f"# skipped {ds} d{dim} {kind} {sw} n_lists {nl}: curve ends at {g.n_probe.max()} < cap {nl // 4}",
            file=sys.stderr,
        )
        continue
    pick = {int(g.n_probe.max())}
    for col in ("oracle_recall@100", "oracle_recall@1000"):
        for t in TARGETS:
            ok = g[g[col] >= t]
            if len(ok):
                pick.add(int(ok.n_probe.iloc[0]))
    if nl == smallest[ds, dim]:
        pick.add(24)
    points[ds, int(dim), kind, sw, int(nl)] = sorted(pick)


def suite(dim, keys, ks):
    dss = sorted({k[0] for k in keys})
    arms = list(V12)
    for ds, _, kind, sw, nl in keys:
        arms.append(
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "filter_kinds": [kind],
                "build": {"n_lists": [nl]},
                "datasets": {
                    ds: {
                        "sweeps": [sw],
                        "query": {"n_probe": points[ds, dim, kind, sw, nl]},
                    }
                },
            }
        )
    return {
        "datasets": dss,
        "dims": [dim],
        "filter_kinds": sorted({k[2] for k in keys}),
        "ks": ks,
        "batch_sizes": [1, 16, 64],
        "seeds": [0],
        "sweeps": {d: sorted({k[3] for k in keys if k[0] == d}) for d in dss},
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms,
    }


out = {}
for dim in sorted({k[1] for k in points}):
    out[f"n-timed-d{dim}"] = suite(
        dim, sorted(k for k in points if k[1] == dim), [100, 1000]
    )
    k2 = sorted(
        k
        for k in points
        if k[1] == dim and K2048.get((k[0], dim)) == k[3] and k[2] == "clause"
    )
    if k2:
        out[f"n-st-k2048-d{dim}"] = suite(dim, k2, [2048])
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
print(
    f"# {len(points)} (dataset, dim, kind, sweep, n_lists) groups, {sum(map(len, points.values()))} timed SilverTorch points; "
    f"suites {list(out)}",
    file=sys.stderr,
)
