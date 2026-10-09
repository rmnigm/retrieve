"""Scratch config for V-YFCC deep's calibration cell: the real `deep` suite cut to yfcc10m, the linr_v3 arm only,
candidate_pool_frac 0.01, seed 0; bs, k and everything else as the suite (one timed V3 cell at 10 M).
Usage (from evaluation/): python yfcc_calib_config.py OUT"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
deep = copy.deepcopy(real["deep"])
deep.update(
    datasets=["yfcc10m"], dims=[192], seeds=[0], sweeps={"yfcc10m": ["tags_and"]}
)
deep["arms"] = [
    {**a, "query": {"candidate_pool_frac": [0.01]}}
    for a in deep["arms"]
    if a["algo"] == "linr_v3"
]
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(
    yaml.safe_dump({"deep": deep, "bloom": real["bloom"]}, sort_keys=False)
)
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (out / ds.name).exists():
        (out / ds.name).symlink_to(ds.resolve())
jobs = load_matrix(out / "yfcc10m.yaml", out / "suites.yaml", "deep")
print(
    [
        (
            j.algo,
            j.backend,
            j.filter_kind,
            j.sweep,
            j.seed,
            j.query,
            j.ks,
            j.batch_sizes,
        )
        for j in jobs
    ]
)
