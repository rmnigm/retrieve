"""Run a's suites: the final run's head-to-head SilverTorch cells (bloom 1,024 / 5, recheck off, graph, k 100,
bs 1 and 16, seed 0) re-run on current code. Per (dataset, filter): the points the head-to-head selected at
recall 0.90 / 0.95 at either batch, or, where SilverTorch did not reach a target, its best-recall timed point
inside the probe budget (n_probe <= n_lists / 4); plus n_probe 24 at each of those n_lists.

usage: gen_a.py RECORDS.parquet SELECTIONS.csv OUT_SUITES.yaml OUT_POINTS.csv"""

import json
import sys

import polars as pl
import yaml

recs, sel = pl.read_parquet(sys.argv[1]), pl.read_csv(sys.argv[2])
SYNTH = {"goodreads-synth": ["p001", "p01"], "arxiv-synth": ["p001", "p01"],
         "pubmed-synth": ["p001", "p01"], "laion30m-synth": ["p01", "p05"]}  # fmt: skip
DIM = {"goodreads": 128, "arxiv": 128, "goodreads-synth": 128, "arxiv-synth": 128, "pubmed": 768,
       "pubmed-synth": 768}  # fmt: skip

st = recs.filter(
    (pl.col("experiment") == "head_to_head")
    & (pl.col("algo") == "silvertorch")
    & (pl.col("seed") == 0)
    & (pl.col("perf_k") == 100)
    & pl.col("perf_bs").is_in([1, 16])
).with_columns(
    n_lists=pl.col("params").map_elements(
        lambda p: json.loads(p)["n_lists"], return_dtype=pl.Int64
    ),
    n_probe=pl.col("params").map_elements(
        lambda p: json.loads(p)["n_probe"], return_dtype=pl.Int64
    ),
)
sel = sel.filter((pl.col("k") == 100) & pl.col("batch").is_in([1, 16]))
wanted = sel.filter(
    (pl.col("filter_type") == "real")
    | pl.struct("dataset", "filter").map_elements(
        lambda r: r["filter"] in SYNTH.get(r["dataset"], []), return_dtype=pl.Boolean
    )
)

points = set()
for r in wanted.iter_rows(named=True):
    if r["silvertorch_reached_target"]:
        points.add(
            (
                r["dataset"],
                r["filter"],
                r["silvertorch_n_lists"],
                r["silvertorch_n_probe"],
            )
        )
        continue
    c = st.filter(
        (pl.col("dataset") == r["dataset"])
        & (pl.col("sweep") == r["filter"])
        & (pl.col("perf_bs") == r["batch"])
        & (pl.col("n_probe") <= pl.col("n_lists") // 4)
    )
    best = c.sort("oracle_recall@100", descending=True).row(0, named=True)
    points.add((r["dataset"], r["filter"], best["n_lists"], best["n_probe"]))
for ds, f, nl in {(p[0], p[1], p[2]) for p in points}:
    points.add((ds, f, nl, 24))

suites = {"bloom": {"m_bits": 1024, "k_hash": 5}}
for ds, f, nl, npb in sorted(points):
    dim = DIM.get(ds, 256)
    s = suites.setdefault(
        f"gf-a-d{dim}",
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
yaml.safe_dump(suites, open(sys.argv[3], "w"), sort_keys=False)
pl.DataFrame(
    sorted(points), schema=["dataset", "filter", "n_lists", "n_probe"], orient="row"
).write_csv(sys.argv[4])
print(f"{len(points)} points:", {k: len(v.get("arms", [])) for k, v in suites.items()})
