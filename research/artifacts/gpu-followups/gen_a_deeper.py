"""Run a, second pass: where no re-run point reaches a head-to-head target the final run reached (the inner-product k-means
moves recall at a fixed probe), the same lists probed deeper: 1.5x, 2x and 4x the deepest re-run n_probe,
capped at n_lists / 4 (the head-to-head's budget). Suites as gen_a.py's, named gf-a2-d<dim>.

usage: gen_a_deeper.py POINTS.csv (compare_a.py) SELECTIONS.csv DATASETS OUT_SUITES.yaml"""

import sys

import polars as pl
import yaml

pts, sel = pl.read_csv(sys.argv[1]), pl.read_csv(sys.argv[2])
only = set(sys.argv[3].split(","))
DIM = {"goodreads": 128, "arxiv": 128, "goodreads-synth": 128, "arxiv-synth": 128, "pubmed": 768,
       "pubmed-synth": 768}  # fmt: skip
sel = sel.filter(
    (pl.col("k") == 100)
    & pl.col("batch").is_in([1, 16])
    & pl.col("dataset").is_in(only)
)
extra = set()
for r in sel.filter(pl.col("silvertorch_reached_target")).iter_rows(named=True):
    c = pts.filter(
        (pl.col("dataset") == r["dataset"])
        & (pl.col("sweep") == r["filter"])
        & (pl.col("bs") == r["batch"])
    )
    if (
        c.is_empty()
        or (
            c.filter(pl.col("n_probe") <= pl.col("n_lists") // 4)["new_recall"]
            >= r["recall_target"]
        ).any()
    ):
        continue
    for nl in c["n_lists"].unique().to_list():
        deepest = c.filter(pl.col("n_lists") == nl)["n_probe"].max()
        for f in (1.5, 2, 4):
            p = min(int(deepest * f), nl // 4)
            if p > deepest:
                extra.add((r["dataset"], r["filter"], nl, p))
suites = {"bloom": {"m_bits": 1024, "k_hash": 5}}
for ds, f, nl, npb in sorted(extra):
    dim = DIM.get(ds, 256)
    s = suites.setdefault(
        f"gf-a2-d{dim}",
        {
            "datasets": [],
            "dims": [dim],
            "filter_kinds": ["bloom"],
            "ks": [100],
            "batch_sizes": [1, 16],
            "seeds": [0],
            "sweeps": {},
            "arms": [],
        },
    )
    if ds not in s["datasets"]:
        s["datasets"].append(ds)
    if f not in s["sweeps"].setdefault(ds, []):
        s["sweeps"][ds].append(f)
    arm = next(
        (
            a
            for a in s["arms"]
            if a["build"]["n_lists"] == [nl]
            and ds in a["datasets"]
            and a["datasets"][ds]["sweeps"] == [f]
        ),
        None,
    )
    if arm is None:
        arm = {
            "algo": "silvertorch",
            "backends": ["cuda"],
            "filter_kinds": ["bloom"],
            "build": {"n_lists": [nl], "recheck": [False]},
            "datasets": {ds: {"sweeps": [f], "query": {"n_probe": []}}},
        }
        s["arms"].append(arm)
    arm["datasets"][ds]["query"]["n_probe"].append(npb)
yaml.safe_dump(suites, open(sys.argv[4], "w"), sort_keys=False)
print(sorted(extra))
