"""Timed points for the bloom legs (user's bloom directive, 2026-10-12): from the A1-bloom curves (SilverTorch bloom on the adapted backend,
recheck off), per (dataset, dim, sweep, n_lists) the smallest n_probe reaching recall 0.90 and 0.95 (oracle recall@k), plus 24 at the smallest
n_lists of the dataset as the paper's reference point. A group whose curve stops short of n_lists / 4 is skipped.

Writes, per dim, `n-bloom-timed-d<dim>` (k 100, priority 2), `n-bloom-timed-k1000-d<dim>` (k 1000 picks, priority 3a) and, for the synth sets,
`n-bloom-timed-synth-d<dim>` (k 100, priority 3b). Every suite runs SilverTorch bloom (recheck off) beside LiNR V1 / V2 with the exact clause filter on the same sweeps
(LiNR with a bloom filter is not in the study, user 2026-10-13), bs 1 / 16 / 64, interleaved by (algo, n_lists).

The k 100 real-filter suites also carry LiNR V3 at matched recall (user, 2026-10-13, RED 2): per sweep, the smallest candidate pool
reaching recall@100 0.90 and 0.95 on the V3 pool legs (NUMBERS L2 / L2b), in the same interleave group; a target no pool reaches
(within 20 %) gets no V3 point.

usage: pick_bloom.py CURVE_PARQUET [...] --v3 V3_POOL_PARQUET [...] > suites-bloom-timed.yaml
"""

import json
import sys

import pandas as pd
import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


TARGETS = [0.90, 0.95]
DIM = {
    "goodreads": 128,
    "arxiv": 128,
    "pubmed": 768,
}  # every other dataset here is d256
V12 = [
    {
        "algo": "linr_v1_filter_mask",
        "backends": ["triton"],
        "filter_kinds": ["clause"],
    },
    {"algo": "linr_v2", "backends": ["triton"], "filter_kinds": ["clause"]},
]

args = sys.argv[1:]
cut = args.index("--v3") if "--v3" in args else len(args)
df = pd.concat([pd.read_parquet(p) for p in args[:cut]], ignore_index=True)
v3 = (
    pd.concat([pd.read_parquet(p) for p in args[cut + 1 :]], ignore_index=True)
    if cut < len(args)
    else None
)
v3_pools = {}
if v3 is not None:
    v3 = v3[(v3.algo == "linr_v3") & (v3.status == "ok") & (v3.filter_kind == "clause")]
    v3 = v3.assign(
        pool=v3.params.map(lambda s: json.loads(s)["candidate_pool_frac"])
    ).drop_duplicates(["dataset", "sweep", "pool"])
    for (ds, sw), g in v3.groupby(["dataset", "sweep"]):
        g = g.sort_values("pool")
        pick = sorted(
            {
                float(g[g["oracle_recall@100"] >= t].pool.iloc[0])
                for t in TARGETS
                if (g["oracle_recall@100"] >= t).any()
            }
        )
        if pick:
            v3_pools[ds, sw] = pick
df = df[
    (df.backend == "official-fork")
    & (df.status == "ok")
    & (df.filter_kind == "bloom")
    & df.suite.str.startswith("n-bloom-curve")
]
df = df[df.dim == df.dataset.map(lambda d: DIM.get(d.removesuffix("-synth"), 256))]
p = df.params.map(json.loads)
df = df.assign(
    n_lists=p.map(lambda d: d["n_lists"]), n_probe=p.map(lambda d: d["n_probe"])
)
df = df.drop_duplicates(["dataset", "dim", "sweep", "n_lists", "n_probe"])
smallest = df.groupby(["dataset", "dim"]).n_lists.min()

points = {100: {}, 1000: {}}
for (ds, dim, sw, nl), g in df.groupby(["dataset", "dim", "sweep", "n_lists"]):
    g = g.sort_values("n_probe")
    if g.n_probe.max() < nl // 4:
        print(
            f"# skipped {ds} d{dim} {sw} n_lists {nl}: curve ends at {g.n_probe.max()}",
            file=sys.stderr,
        )
        continue
    for k in points:
        pick = {24} if nl == smallest[ds, dim] and k == 100 else set()
        for t in TARGETS:
            ok = g[g[f"oracle_recall@{k}"] >= t]
            if len(ok):
                pick.add(int(ok.n_probe.iloc[0]))
        if pick:
            points[k][ds, int(dim), sw, int(nl)] = sorted(pick)


def suite(dim, keys, k, pts, with_v3=False):
    dss = sorted({key[0] for key in keys})
    arms = list(V12)
    if with_v3:
        for ds, sw in sorted({(key[0], key[2]) for key in keys}):
            if (ds, sw) in v3_pools:
                arms.append(
                    {
                        "algo": "linr_v3",
                        "backends": ["triton"],
                        "filter_kinds": ["clause"],
                        "datasets": {
                            ds: {
                                "sweeps": [sw],
                                "query": {"candidate_pool_frac": v3_pools[ds, sw]},
                            }
                        },
                    }
                )
    for ds, _, sw, nl in keys:
        arms.append(
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "filter_kinds": ["bloom"],
                "build": {"n_lists": [nl], "recheck": [False]},
                "datasets": {
                    ds: {"sweeps": [sw], "query": {"n_probe": pts[ds, dim, sw, nl]}}
                },
            }
        )
    return {
        "datasets": dss,
        "dims": [dim],
        "filter_kinds": ["clause", "bloom"],
        "ks": [k],
        "batch_sizes": [1, 16, 64],
        "seeds": [0],
        "sweeps": {d: sorted({key[2] for key in keys if key[0] == d}) for d in dss},
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms,
    }


out = {}
for k, name in [(100, "n-bloom-timed"), (1000, "n-bloom-timed-k1000")]:
    pts = points[k]
    for dim in sorted({key[1] for key in pts}):
        real = sorted(
            key for key in pts if key[1] == dim and not key[0].endswith("synth")
        )
        if real:
            out[f"{name}-d{dim}"] = suite(dim, real, k, pts, with_v3=k == 100)
        synth = sorted(key for key in pts if key[1] == dim and key[0].endswith("synth"))
        if synth and k == 100:
            out[f"n-bloom-timed-synth-d{dim}"] = suite(dim, synth, k, pts)
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
print(
    f"# suites {list(out)}; SilverTorch points k100 {sum(map(len, points[100].values()))}, k1000 {sum(map(len, points[1000].values()))}",
    file=sys.stderr,
)
