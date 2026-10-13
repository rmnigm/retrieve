"""B2 gate (bloom-runner): load every n-bloom-timed* suite in CONFIG_DIR/suites.yaml on its datasets; refuse LiNR bloom arms, YFCC, or
SilverTorch bloom without recheck=False. Prints per (suite, dataset): jobs, cells, arms.
usage: check_b2.py CONFIG_DIR"""
import sys
from collections import Counter
from pathlib import Path

import yaml
from bench.algos import official_config
from bench.config import load_matrix

c = Path(sys.argv[1])
suites = {k: v for k, v in yaml.safe_load((c / "suites.yaml").read_text()).items() if k.startswith("n-bloom-timed")}
bad = []
for s, spec in suites.items():
    for ds in spec["datasets"]:
        if ds.startswith("yfcc"):
            bad.append(f"{s}/{ds}: YFCC is out of the study")
            continue
        js = load_matrix(c / f"{ds}.yaml", c / "suites.yaml", s)
        arms = Counter((j.algo, j.backend, j.filter_kind) for j in js)
        for j in js:
            if j.algo.startswith("linr") and j.filter_kind != "clause":
                bad.append(f"{s}/{ds}: LiNR {j.algo} {j.filter_kind}")
            if j.algo == "silvertorch" and (j.filter_kind != "bloom" or official_config(j.algo, j.filter_kind, j.backend, j.build).recheck_for(j.backend)):
                bad.append(f"{s}/{ds}: silvertorch {j.filter_kind} recheck not off")
        print(s, ds, "jobs", len(js), dict(arms))
print("REFUSE" if bad else "OK", *sorted(set(bad)), sep="\n")
sys.exit(1 if bad else 0)
