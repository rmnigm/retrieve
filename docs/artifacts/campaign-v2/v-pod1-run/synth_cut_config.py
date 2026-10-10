"""Scratch config for a synth leg on one dataset: the real `synth` suite with that dataset's sweep list set to RATES at k 100 and,
optionally, seeds 0-2 at the --seeds-at rates (repeats for a CI). Other datasets' keys unchanged. Usage (from evaluation/):
python synth_cut_config.py OUT DATASET RATE... [--seeds-at RATE...]"""

import argparse
import copy
from pathlib import Path

import yaml
from bench.config import load_matrix

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("dataset")
ap.add_argument("rates", nargs="+")
ap.add_argument("--seeds-at", nargs="*", default=[])
a = ap.parse_args()
real = yaml.safe_load(Path("config/suites.yaml").read_text())
synth = copy.deepcopy(real["synth"])
synth["sweeps"][a.dataset] = a.rates
synth["ks_by_sweep"][a.dataset] = {r: [100] for r in a.rates}
if a.seeds_at:
    synth.setdefault("seeds_by_sweep", {})[a.dataset] = {
        r: [0, 1, 2] for r in a.seeds_at
    }
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "suites.yaml").write_text(
    yaml.safe_dump({"synth": synth, "bloom": real["bloom"]}, sort_keys=False)
)
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (a.out / ds.name).exists():
        (a.out / ds.name).symlink_to(ds.resolve())
jobs = load_matrix(a.out / f"{a.dataset}.yaml", a.out / "suites.yaml", "synth")
print(sorted({(j.algo, j.sweep, j.seed) for j in jobs if j.algo.startswith("linr_v")}))
