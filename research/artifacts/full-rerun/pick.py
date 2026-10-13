"""FULL-RERUN stage 2 (README.md): the timed suites, picked from stage 1's seed-0 curves (controller C4, 2026-10-13) and run at
every seed (synth draw d at seed d). Per (dataset, dim, kind, sweep, n_lists) the smallest n_probe reaching oracle recall 0.90 and
0.95, plus the papers' 24 at the dataset's smallest n_lists; a curve that stops short of n_lists / 4 is skipped.

- `fr-timed-bloom-d<dim>` (k 100) and `-k1000-d<dim>` (k 1000): SilverTorch bloom, recheck off, beside LiNR V1 / V2 with the exact
  clause filter (LiNR's bloom is not in the study); the k 100 suites also carry LiNR V3 at matched recall (the smallest seed-0 pool
  of `fr-v3pool*` reaching recall@100 0.90 / 0.95; a target no pool reaches within 20 % gets no V3 point).
- `fr-timed-bloom-k2048-d<dim>`: the papers' own k on one sweep per real dataset, at the union of its k 100 / k 1000 picks.
- `fr-timed-bloom-synth-s<d>-d<dim>`: the synth datasets with a bloom curve, draw d at seed d.
- `fr-timed-clause-d<dim>`: SilverTorch's exact clause mode (our adaptation, a reference row only) at its k 100 picks, every seed.
Every suite is bs 1 / 16 / 64, interleaved by (algo, n_lists).

usage: pick.py STAGE1_PARQUET [...] > suites-timed.yaml
"""

import json
import sys

import polars as pl
import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


TARGETS = [0.90, 0.95]
SEEDS = [0, 1, 2]
K2048 = {"goodreads": "c0_genre", "arxiv": "c0_maincat", "pubmed": "c0_mesh"}
V12 = [
    {"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["clause"]},
    {"algo": "linr_v2", "backends": ["triton"], "filter_kinds": ["clause"]},
]
COLS = ["suite", "dataset", "dim", "filter_kind", "sweep", "algo", "backend", "seed", "status", "params",
        "oracle_recall@100", "oracle_recall@1000"]  # fmt: skip


def param(name, dtype=pl.Int64):
    return pl.col("params").map_elements(
        lambda s: dtype.to_python()(json.loads(s)[name]), return_dtype=dtype
    )


def read(path):
    t = pl.read_parquet(path)
    return t.select(
        [
            pl.col(c) if c in t.columns else pl.lit(None, pl.Float64).alias(c)
            for c in COLS
        ]
    )


df = pl.concat([read(p) for p in sys.argv[1:]], how="vertical_relaxed")
df = df.filter((pl.col("status") == "ok") & (pl.col("seed") == 0))
v3 = (
    df.filter(
        (pl.col("algo") == "linr_v3") & pl.col("suite").str.starts_with("fr-v3pool")
    )
    .with_columns(pool=param("candidate_pool_frac", pl.Float64))
    .unique(["dataset", "sweep", "pool"], keep="first")
    .sort("pool")
)
v3_pools = {}
for (ds, sw), g in v3.group_by(["dataset", "sweep"]):
    pick = sorted({g.filter(pl.col("oracle_recall@100") >= t)["pool"][0] for t in TARGETS
                   if (g["oracle_recall@100"] >= t).any()})  # fmt: skip
    if pick:
        v3_pools[ds, sw] = pick

cur = (
    df.filter(
        (pl.col("algo") == "silvertorch")
        & (pl.col("backend") == "official-fork")
        & pl.col("suite").str.starts_with("fr-curve")
    )  # fmt: skip
    .with_columns(n_lists=param("n_lists"), n_probe=param("n_probe"))
    .unique(
        ["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"], keep="first"
    )
    .sort("n_probe")
)
smallest = {
    (ds, dim): n
    for ds, dim, n in cur.group_by(["dataset", "dim"])
    .agg(pl.col("n_lists").min())
    .iter_rows()
}

points = {}  # (k, kind) -> {(ds, dim, sweep, n_lists): [n_probe]}
for (ds, dim, kind, sw, nl), g in cur.group_by(
    ["dataset", "dim", "filter_kind", "sweep", "n_lists"]
):
    if g["n_probe"].max() < nl // 4:
        print(
            f"# skipped {ds} d{dim} {kind} {sw} n_lists {nl}: curve ends at {g['n_probe'].max()}",
            file=sys.stderr,
        )
        continue
    for k in (100, 1000):
        pick = {24} if nl == smallest[ds, dim] and k == 100 else set()
        for t in TARGETS:
            ok = g.filter(pl.col(f"oracle_recall@{k}") >= t)
            if len(ok):
                pick.add(int(ok["n_probe"][0]))
        if pick:
            points.setdefault((k, kind), {})[ds, int(dim), sw, int(nl)] = sorted(pick)


def suite(dim, keys, ks, pts, kind, seeds, with_v3=False, rename=lambda d: d):
    dss = sorted({key[0] for key in keys})
    arms = list(V12)
    for ds, sw in sorted({(key[0], key[2]) for key in keys}) if with_v3 else []:
        if (ds, sw) in v3_pools:
            arms.append(
                {"algo": "linr_v3", "backends": ["triton"], "filter_kinds": ["clause"],
                 "datasets": {ds: {"sweeps": [sw], "query": {"candidate_pool_frac": v3_pools[ds, sw]}}}}
            )  # fmt: skip
    for ds, _, sw, nl in keys:
        build = {"n_lists": [nl], **({"recheck": [False]} if kind == "bloom" else {})}
        arms.append(
            {"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": [kind], "build": build,
             "datasets": {rename(ds): {"sweeps": [sw], "query": {"n_probe": pts[ds, dim, sw, nl]}}}}
        )  # fmt: skip
    return {
        "datasets": [rename(d) for d in dss],
        "dims": [dim],
        "filter_kinds": sorted({"clause", kind}),
        "ks": ks,
        "batch_sizes": [1, 16, 64],
        "seeds": seeds,
        "sweeps": {
            rename(d): sorted({key[2] for key in keys if key[0] == d}) for d in dss
        },
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms,
    }


def by_dim(pts, synth):
    out = {}
    for key in pts:
        if key[0].endswith("synth") == synth:
            out.setdefault(key[1], []).append(key)
    return {dim: sorted(keys) for dim, keys in sorted(out.items())}


out = {}
b100, b1000 = points.get((100, "bloom"), {}), points.get((1000, "bloom"), {})
for dim, keys in by_dim(b100, False).items():
    out[f"fr-timed-bloom-d{dim}"] = suite(
        dim, keys, [100], b100, "bloom", SEEDS, with_v3=True
    )
for dim, keys in by_dim(b1000, False).items():
    out[f"fr-timed-bloom-k1000-d{dim}"] = suite(
        dim, keys, [1000], b1000, "bloom", SEEDS
    )
both = {
    key: sorted(set(b100.get(key, [])) | set(b1000.get(key, [])))
    for key in {*b100, *b1000}
    if K2048.get(key[0]) == key[2]
}
for dim, keys in by_dim(both, False).items():
    out[f"fr-timed-bloom-k2048-d{dim}"] = suite(dim, keys, [2048], both, "bloom", SEEDS)
for dim, keys in by_dim(b100, True).items():
    for d in SEEDS:
        rename = (lambda ds: ds) if d == 0 else (lambda ds, d=d: f"{ds}-s{d}")
        out[f"fr-timed-bloom-synth-s{d}-d{dim}"] = suite(
            dim, keys, [100], b100, "bloom", [d], rename=rename
        )
c100 = points.get((100, "clause"), {})
for dim, keys in by_dim(c100, False).items():
    out[f"fr-timed-clause-d{dim}"] = suite(dim, keys, [100], c100, "clause", SEEDS)
yaml.dump(out, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
n = {f"{k}/{kind}": sum(map(len, v.values())) for (k, kind), v in points.items()}
print(
    f"# suites {list(out)}; SilverTorch points {n}; V3 sweeps {len(v3_pools)}",
    file=sys.stderr,
)
