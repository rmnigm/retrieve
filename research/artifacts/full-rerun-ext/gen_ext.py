"""TUNE-EXT phase 1 (user 2026-10-14 15:30Z, `.chains/control/2026-10-14-090000000-user-fr-scope-seed0-plus-variance.md`): the
cells whose seed-0 SilverTorch curve never reaches oracle recall@100 0.90 within n_probe <= n_lists / 4, per (dataset, dim, filter
kind, sweep) over every n_lists the curve used, get two more quality-only points at n_probe = n_lists / 2 and 3 n_lists / 4, at the
largest n_lists that cell used. Same arm shape as full-rerun's `fr-curve` suites (adapted backend, bloom with the recheck off, k 100 /
1000, bs 16, seed 0). One suite per (pod that timed the dataset at seed 0, dim, real / synth); goodreads-synth is pod e's, the rest
pod 1's. Reads the `fr/curve*-s0` legs' results.parquet; writes the suites and the cell table, no hand-picked points.

usage: gen_ext.py CURVE_PARQUET [...] > suites-ext-curve.yaml     (the cell table goes to stderr)
"""

import json
import sys

import polars as pl
import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


TARGET = 0.90
POD_E = {"goodreads", "arxiv", "laion30m", "goodreads-synth", "arxiv-synth", "arxiv-corr-synth", "laion30m-synth"}


def param(name):
    return pl.col("params").map_elements(lambda s: int(json.loads(s)[name]), return_dtype=pl.Int64)


cols = ["suite", "dataset", "dim", "filter_kind", "sweep", "algo", "backend", "seed", "status", "params", "oracle_recall@100"]
df = pl.concat([pl.read_parquet(p).select(cols) for p in sys.argv[1:]], how="vertical_relaxed")
cur = (
    df.filter(
        (pl.col("status") == "ok") & (pl.col("seed") == 0) & (pl.col("algo") == "silvertorch")
        & (pl.col("backend") == "official-fork") & pl.col("suite").str.starts_with("fr-curve")
    )  # fmt: skip
    .with_columns(n_lists=param("n_lists"), n_probe=param("n_probe"))
    .filter(pl.col("n_probe") <= pl.col("n_lists") // 4)
)
key = ["dataset", "dim", "filter_kind", "sweep"]
cells = (
    cur.group_by(key)
    .agg(pl.col("oracle_recall@100").max().alias("best"), pl.col("n_lists").max().alias("n_lists"))
    .filter(pl.col("best") < TARGET)
    .sort(key)
)
print(cells.with_columns(pod=pl.col("dataset").map_elements(lambda d: "pode" if d in POD_E else "pod1", return_dtype=pl.Utf8)),
      file=sys.stderr)  # fmt: skip

out = {}
for (pod, dim, synth), g in cells.with_columns(
    pod=pl.col("dataset").map_elements(lambda d: "pode" if d in POD_E else "pod1", return_dtype=pl.Utf8),
    synth=pl.col("dataset").str.ends_with("synth"),
).group_by(["pod", "dim", "synth"]):
    arms, sweeps = [], {}
    for (kind, nl), h in g.group_by(["filter_kind", "n_lists"]):
        ds = {}
        for d, sw in h.select("dataset", "sweep").iter_rows():
            ds.setdefault(d, {"sweeps": [], "query": {"n_probe": [nl // 2, 3 * nl // 4]}})["sweeps"].append(sw)
            sweeps.setdefault(d, set()).add(sw)
        build = {"n_lists": [int(nl)], **({"recheck": [False]} if kind == "bloom" else {})}
        arms.append({"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": [kind], "build": build,
                     "datasets": {d: {"sweeps": sorted(v["sweeps"]), "query": v["query"]} for d, v in sorted(ds.items())}})  # fmt: skip
    arms.sort(key=lambda a: (a["filter_kinds"][0], a["build"]["n_lists"][0]))
    name = f"fr-ext-curve{'-synth' if synth else ''}-{pod}-d{dim}"
    out[name] = {
        "perf": False,
        "datasets": sorted(sweeps),
        "dims": [int(dim)],
        "filter_kinds": sorted(set(g["filter_kind"])),
        "ks": [100, 1000],
        "batch_sizes": [16],
        "seeds": [0],
        "sweeps": {d: sorted(s) for d, s in sorted(sweeps.items())},
        "arms": arms,
    }
yaml.dump(dict(sorted(out.items())), sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
print(f"# suites {sorted(out)}; cells {len(cells)}", file=sys.stderr)
