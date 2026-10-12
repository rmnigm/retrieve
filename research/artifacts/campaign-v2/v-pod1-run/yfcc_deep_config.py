"""Scratch config for V-YFCC deep's claims-first cut: the real `deep` suite on yfcc10m, SilverTorch triton only, n_lists 4096,
n_probe {24, 256, 1024}, seed 0, k 100 (controller ruling 2026-10-10). Usage (from evaluation/): python yfcc_deep_config.py OUT"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
deep = copy.deepcopy(real["deep"])
deep.update(
    datasets=["yfcc10m"],
    dims=[192],
    seeds=[0],
    ks=[100],
    sweeps={"yfcc10m": ["tags_and"]},
)
arm = next(
    a
    for a in deep["arms"]
    if a["algo"] == "silvertorch" and a["backends"] == ["triton"]
)
deep["arms"] = [{"algo": "silvertorch", "backends": ["triton"], "build": {"n_lists": [4096]},
                 "query": {"n_probe": [24, 256, 1024]}, "filter_kinds": arm.get("filter_kinds", ["clause"])}]  # fmt: skip
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
        (j.algo, j.backend, j.filter_kind, j.build, j.query, j.ks, j.batch_sizes)
        for j in jobs
    ]
)
