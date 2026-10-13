"""grade_bloom.py POINTS_JSON per dim from the k 100 bloom-timed suites: SilverTorch arms -> [[dataset, sweep, n_lists, [n_probe...]], ...].
usage: grade_points.py SUITES_YAML OUTDIR   (writes points-d<dim>.json for every n-bloom-timed-d<dim>)"""
import json
import sys
from pathlib import Path

import yaml

suites = yaml.safe_load(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
for name, spec in suites.items():
    if not name.startswith("n-bloom-timed-d"):
        continue
    pts = {}
    for arm in spec["arms"]:
        if arm["algo"] != "silvertorch":
            continue
        for nl in arm["build"]["n_lists"]:
            for ds, d in arm["datasets"].items():
                for sw in d["sweeps"]:
                    pts.setdefault((ds, sw, nl), set()).update(d["query"]["n_probe"])
    rows = [[ds, sw, nl, sorted(p)] for (ds, sw, nl), p in sorted(pts.items())]
    (out / f"points-{name.removeprefix('n-bloom-timed-')}.json").write_text(json.dumps(rows))
    print(name, len(rows), "groups", sum(len(r[3]) for r in rows), "points")
