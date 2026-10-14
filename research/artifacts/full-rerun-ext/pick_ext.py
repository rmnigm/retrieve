"""TUNE-EXT phase 2: picks for the TUNE-EXT cells only, by pick.py's rule (the smallest n_probe reaching oracle recall 0.90 and 0.95, at
k 100 and k 1000) over the cell's whole seed-0 curve (the fr-curve points plus the fr-ext-curve points at n_lists / 2 and 3 n_lists /
4). Only points not already in the seed-0 picks (`$FR/picks.yaml`) are new: existing picks and records stay unchanged. Writes, seed 0
only (no seed variance for TUNE-EXT points):
- `fr-ext-timed-bloom-<pod>-d<dim>` (k 100, LiNR V1 / V2 exact and V3 at the seed-0 pools re-timed in the same run) and
  `fr-ext-timed-bloom-k1000-<pod>-d<dim>` (k 1000, V1 / V2), SilverTorch bloom with the recheck off; synth datasets in
  `fr-ext-timed-bloom-synth-<pod>-d<dim>`. Every suite bs 1 / 16 / 64, interleaved by (algo, n_lists), as pick.py's.
- `fr-ext-timed-clause[-synth]-<pod>-d<dim>`: the clause-row points at k 100, SilverTorch exact arms only (the row is lighter: user 2026-10-14
  11:00Z); synth cells too, since the LAION synth TUNE-EXT cells are clause only (controller ~21:45Z: phase 2 for those cells).
- `grade-<pod>-d<dim>.json`: the new bloom points for grade_bloom.py, [[dataset, sweep, n_lists, [n_probe...]], ...].

usage: pick_ext.py PICKS_YAML OUTDIR CURVE_PARQUET [...]   (fr/curve*-s0, fr/v3pool*-s0 and fr/ext-curve*-s0 legs' results.parquet)
"""

import json
import sys
from pathlib import Path

import polars as pl
import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


TARGETS = [0.90, 0.95]
CELL_TARGET = 0.90
POD_E = {"goodreads", "arxiv", "laion30m", "goodreads-synth", "arxiv-synth", "arxiv-corr-synth", "laion30m-synth"}
V12 = [
    {"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["clause"]},
    {"algo": "linr_v2", "backends": ["triton"], "filter_kinds": ["clause"]},
]
COLS = ["suite", "dataset", "dim", "filter_kind", "sweep", "algo", "backend", "seed", "status", "params",
        "oracle_recall@100", "oracle_recall@1000"]  # fmt: skip


def pod(ds):
    return "pode" if ds in POD_E else "pod1"


def param(name, dtype=pl.Int64):
    return pl.col("params").map_elements(lambda s: dtype.to_python()(json.loads(s)[name]), return_dtype=dtype)


def read(path):
    t = pl.read_parquet(path)
    return t.select([pl.col(c) if c in t.columns else pl.lit(None, pl.Float64).alias(c) for c in COLS])


picks_yaml, outdir = yaml.safe_load(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
df = pl.concat([read(p) for p in sys.argv[3:]], how="vertical_relaxed")
df = df.filter((pl.col("status") == "ok") & (pl.col("seed") == 0))

# the seed-0 V3 pools, as pick.py
v3 = (
    df.filter((pl.col("algo") == "linr_v3") & pl.col("suite").str.starts_with("fr-v3pool"))
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

st = (
    df.filter((pl.col("algo") == "silvertorch") & (pl.col("backend") == "official-fork")
              & (pl.col("suite").str.starts_with("fr-curve") | pl.col("suite").str.starts_with("fr-ext-curve")))  # fmt: skip
    .with_columns(n_lists=param("n_lists"), n_probe=param("n_probe"))
    .unique(["dataset", "dim", "filter_kind", "sweep", "n_lists", "n_probe"], keep="first")
    .sort("n_probe")
)
# the TUNE-EXT cells: never 0.90 within n_probe <= n_lists / 4 on the stage-1 curve (gen_ext.py's rule), at their largest n_lists
base = st.filter(pl.col("suite").str.starts_with("fr-curve") & (pl.col("n_probe") <= pl.col("n_lists") // 4))
key = ["dataset", "dim", "filter_kind", "sweep"]
cells = (
    base.group_by(key).agg(pl.col("oracle_recall@100").max().alias("best"), pl.col("n_lists").max().alias("n_lists"))
    .filter(pl.col("best") < CELL_TARGET)
)


def old_points(ds, dim, kind, sw, nl, k):
    """n_probe values already timed for this (dataset, kind, sweep, n_lists) at k in the seed-0 picks."""
    out = set()
    for name, s in picks_yaml.items():
        if int(dim) not in s.get("dims", []) or k not in s.get("ks", []):
            continue
        for a in s["arms"]:
            if a["algo"] != "silvertorch" or a["filter_kinds"] != [kind] or a["build"]["n_lists"] != [nl]:
                continue
            for d, v in a["datasets"].items():
                if d == ds and sw in v["sweeps"]:
                    out |= set(v["query"]["n_probe"])
    return out


points, table = {}, []  # points[(k, kind)][(ds, dim, sw, nl)] = [n_probe]
for ds, dim, kind, sw, best, nl in cells.sort(key).iter_rows():
    g = st.filter((pl.col("dataset") == ds) & (pl.col("dim") == dim) & (pl.col("filter_kind") == kind)
                  & (pl.col("sweep") == sw) & (pl.col("n_lists") == nl))  # fmt: skip
    for k in (100, 1000):
        new = set()
        for t in TARGETS:
            ok = g.filter(pl.col(f"oracle_recall@{k}") >= t)
            if len(ok):
                new.add(int(ok["n_probe"][0]))
        new -= old_points(ds, dim, kind, sw, nl, k)
        top = g[f"oracle_recall@{k}"].max()
        table.append((ds, dim, kind, sw, nl, k, round(best, 4), round(top, 4) if top is not None else None, sorted(new)))
        if new:
            points.setdefault((k, kind), {})[ds, int(dim), sw, int(nl)] = sorted(new)


def suite(dim, keys, ks, pts, kind, linr, with_v3=False):
    dss = sorted({key[0] for key in keys})
    arms = list(V12) if linr else []
    for ds, sw in sorted({(key[0], key[2]) for key in keys}) if with_v3 else []:
        if (ds, sw) in v3_pools:
            arms.append({"algo": "linr_v3", "backends": ["triton"], "filter_kinds": ["clause"],
                         "datasets": {ds: {"sweeps": [sw], "query": {"candidate_pool_frac": v3_pools[ds, sw]}}}})  # fmt: skip
    for ds, _, sw, nl in keys:
        build = {"n_lists": [nl], **({"recheck": [False]} if kind == "bloom" else {})}
        arms.append({"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": [kind], "build": build,
                     "datasets": {ds: {"sweeps": [sw], "query": {"n_probe": pts[ds, dim, sw, nl]}}}})  # fmt: skip
    return {
        "datasets": dss,
        "dims": [dim],
        "filter_kinds": sorted({"clause", kind}) if linr else [kind],
        "ks": ks,
        "batch_sizes": [1, 16, 64],
        "seeds": [0],
        "sweeps": {d: sorted({key[2] for key in keys if key[0] == d}) for d in dss},
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms,
    }


def groups(pts):
    out = {}
    for key in pts:
        out.setdefault((pod(key[0]), key[1], key[0].endswith("synth")), []).append(key)
    return {g: sorted(keys) for g, keys in sorted(out.items())}


out, grade = {}, {}
b100, b1000 = points.get((100, "bloom"), {}), points.get((1000, "bloom"), {})
for (p, dim, synth), keys in groups(b100).items():
    name = f"fr-ext-timed-bloom{'-synth' if synth else ''}-{p}-d{dim}"
    out[name] = suite(dim, keys, [100], b100, "bloom", linr=True, with_v3=not synth)
for (p, dim, synth), keys in groups(b1000).items():
    if not synth:
        out[f"fr-ext-timed-bloom-k1000-{p}-d{dim}"] = suite(dim, keys, [1000], b1000, "bloom", linr=True)
for (p, dim, _), keys in groups({**b100, **b1000}).items():
    for ds, _, sw, nl in keys:
        grade.setdefault((p, dim), {}).setdefault((ds, sw, nl), set()).update(b100.get((ds, dim, sw, nl), []) + b1000.get((ds, dim, sw, nl), []))
c100 = points.get((100, "clause"), {})
for (p, dim, synth), keys in groups(c100).items():
    out[f"fr-ext-timed-clause{'-synth' if synth else ''}-{p}-d{dim}"] = suite(dim, keys, [100], c100, "clause", linr=False)

outdir.mkdir(parents=True, exist_ok=True)
(outdir / "suites-ext-timed.yaml").write_text(yaml.dump(out, sort_keys=False, width=140, Dumper=NoAlias))
for (p, dim), d in grade.items():
    (outdir / f"grade-{p}-d{dim}.json").write_text(json.dumps([[ds, sw, nl, sorted(v)] for (ds, sw, nl), v in sorted(d.items())]))
print("dataset dim kind sweep n_lists k best@<=25% best_with_ext new_points", file=sys.stderr)
for row in table:
    print(*row, file=sys.stderr)
print(f"# suites {list(out)}; grade files {sorted(f'{p}-d{dim}' for p, dim in grade)}", file=sys.stderr)
