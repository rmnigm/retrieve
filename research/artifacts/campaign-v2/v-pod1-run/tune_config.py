"""Scratch config for night-queue item 8 (controller 2026-10-12): ANN tuning on YFCC 10 M d192. Suite `tune-q` (quality only): SilverTorch
triton, n_lists {2048, 4096, 8192, 16384} x n_probe {8 ... 4096} (n_probe <= n_lists), k 100, seed 0; clause on yfcc10m `tags_and` and on
yfcc10m-synth p 0.01 / 0.1 / 1, bloom on the synth rates only (the real yfcc10m config has no bloom block). With a FRONTIER json
([[dataset, sweep, filter_kind, n_lists, n_probe], ...]) it also writes suite `tune`: those cells timed at bs {1, 16}.
Usage (from evaluation/): python tune_config.py OUT [FRONTIER.json]"""

import copy
import json
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

N_LISTS = [2048, 4096, 8192, 16384]
N_PROBE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SWEEPS = {"yfcc10m": ["tags_and"], "yfcc10m-synth": ["p001", "p01", "p1"]}
out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
base = {"datasets": list(SWEEPS), "dims": [192], "filter_kinds": ["clause", "bloom"], "ks": [100], "batch_sizes": [1, 16],
        "seeds": [0], "sweeps": copy.deepcopy(SWEEPS)}  # fmt: skip
suites = {"bloom": real["bloom"]}
suites["tune-q"] = {**copy.deepcopy(base), "perf": False, "arms": [
    {"algo": "silvertorch", "backends": ["triton"], "filter_kinds": ["clause"], "build": {"n_lists": N_LISTS}, "query": {"n_probe": N_PROBE}},
    {"algo": "silvertorch", "backends": ["triton"], "filter_kinds": ["bloom"], "build": {"n_lists": N_LISTS}, "query": {"n_probe": N_PROBE},
     "datasets": {"yfcc10m-synth": {}}},
]}  # fmt: skip
if len(sys.argv) > 2:
    groups: dict[tuple, list[int]] = {}
    for ds, sw, fk, nl, npb in json.loads(Path(sys.argv[2]).read_text()):
        groups.setdefault((ds, sw, fk, nl), []).append(npb)
    arms = [
        {
            "algo": "silvertorch",
            "backends": ["triton"],
            "filter_kinds": [fk],
            "build": {"n_lists": [nl]},
            "query": {"n_probe": sorted(nps)},
            "datasets": {ds: {"sweeps": [sw]}},
        }
        for (ds, sw, fk, nl), nps in sorted(groups.items())
    ]  # fmt: skip (one build per (dataset, sweep, kind, n_lists))
    suites["tune"] = {**copy.deepcopy(base), "arms": arms}
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(yaml.safe_dump(suites, sort_keys=False))
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (out / ds.name).exists():
        (out / ds.name).symlink_to(ds.resolve())
for suite in [s for s in ("tune-q", "tune") if s in suites]:
    for ds in SWEEPS:
        jobs = load_matrix(out / f"{ds}.yaml", out / "suites.yaml", suite)
        print(suite, ds, len(jobs), "jobs", sum(len(j.query) for j in jobs), "cells")
