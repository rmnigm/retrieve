"""Scratch config for AFTER-QUEUE 2 on YFCC (controller 2026-10-12): the real `synth` suite with yfcc10m-synth's sweep list
widened from the suite's five rates to nine (adds 0.001 / 0.003 / 0.03 / 0.05, which the 10-rate attrs carry), k 100 at every rate.
Usage (from evaluation/): python yfcc_synth_config.py OUT"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

RATES = ["p0001", "p0003", "p001", "p003", "p005", "p01", "p02", "p05", "p1"]
out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
synth = copy.deepcopy(real["synth"])
synth["sweeps"]["yfcc10m-synth"] = RATES
synth["ks_by_sweep"]["yfcc10m-synth"] = {r: [100] for r in RATES}
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(
    yaml.safe_dump({"synth": synth, "bloom": real["bloom"]}, sort_keys=False)
)
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (out / ds.name).exists():
        (out / ds.name).symlink_to(ds.resolve())
jobs = load_matrix(out / "yfcc10m-synth.yaml", out / "suites.yaml", "synth")
print(sorted({(j.algo, j.backend, j.filter_kind, j.sweep) for j in jobs}))
