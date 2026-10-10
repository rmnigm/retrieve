"""Scratch config for the timed tune phase at campaign-v2.11 (controller 2026-10-12): suite `tune-timed`, SilverTorch triton at the
candidates in tune-timed-candidates.json ([dataset, sweep, filter_kind, n_lists, n_probe]: goodreads / arXiv per (sweep, kind, n_lists) the
first n_probe reaching 0.80 / 0.90 / 0.95 / 0.99 and the knee, from the v2.10 quality grid; YFCC the v2.9 Pareto points and knees), grouped one
arm per (dataset, sweep, kind, n_lists) so each index is built once; k 100, bs {1, 16, 64}, seed 0. Usage (from evaluation/):
python tune_timed_config.py OUT CANDIDATES.json"""

import json
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

out, cands = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
groups: dict[tuple, list[int]] = {}
for ds, sw, fk, nl, npb in cands:
    groups.setdefault((ds, sw, fk, nl), []).append(npb)
datasets = sorted({g[0] for g in groups})
real = yaml.safe_load(Path("config/suites.yaml").read_text())
arms = [{"algo": "silvertorch", "backends": ["triton"], "filter_kinds": [fk], "build": {"n_lists": [nl]}, "query": {"n_probe": sorted(set(nps))},
         "datasets": {ds: {"sweeps": [sw]}}} for (ds, sw, fk, nl), nps in sorted(groups.items())]  # fmt: skip
suite = {"datasets": datasets, "dims": [128, 192], "filter_kinds": ["clause", "bloom"], "ks": [100], "batch_sizes": [1, 16, 64], "seeds": [0],
         "sweeps": {ds: sorted({g[1] for g in groups if g[0] == ds}) for ds in datasets}, "arms": arms}  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(
    yaml.safe_dump({"tune-timed": suite, "bloom": real["bloom"]}, sort_keys=False)
)
for f in Path("config").glob("*.yaml"):
    if f.name != "suites.yaml" and not (out / f.name).exists():
        (out / f.name).symlink_to(f.resolve())
for ds in datasets:
    jobs = load_matrix(out / f"{ds}.yaml", out / "suites.yaml", "tune-timed")
    print(ds, len(jobs), "jobs", sum(len(j.query) for j in jobs), "cells")
