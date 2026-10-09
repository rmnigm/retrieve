"""Scratch config dirs for the H2H protocol diagnostic: the real `h2h` suite cut to goodreads
`c0_genre` bloom, k 100, bs 16, seed 0. OUT/all keeps the three arms (one interleave group);
OUT/triton, OUT/fp16, OUT/int32 hold one arm each, for one process per arm. Suite and dataset names
are the real ones (dataset configs symlinked), so record keys equal the campaign's.
Usage (from evaluation/): python h2h_diag_config.py OUT"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
h2h = copy.deepcopy(real["h2h"])
h2h.update(
    datasets=["goodreads"],
    filter_kinds=["bloom"],
    ks=[100],
    batch_sizes=[16],
    seeds=[0],
)
h2h["sweeps"] = {"goodreads": ["c0_genre"]}
triton, official = h2h["arms"]
assert triton["backends"] == ["triton"] and official["backends"] == ["official"]
def one(path):
    return {**official, "build": {**official.get("build", {}), "score_path": [path]}}

variants = {
    "all": h2h["arms"],
    "triton": [triton],
    "fp16": [one("fp16")],
    "int32": [one("int32")],
}
for name, arms in variants.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    suites = {"h2h": {**h2h, "arms": arms}, "bloom": real["bloom"]}
    (d / "suites.yaml").write_text(yaml.safe_dump(suites, sort_keys=False))
    for ds in Path("config").glob("*.yaml"):
        if ds.name != "suites.yaml" and not (d / ds.name).exists():
            (d / ds.name).symlink_to(ds.resolve())
    jobs = load_matrix(d / "goodreads.yaml", d / "suites.yaml", "h2h")
    print(name, [(j.backend, j.build, q) for j in jobs for q in j.query])
