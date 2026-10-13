"""grade_bloom.py's POINTS_JSON from the picked timed suites: every SilverTorch bloom arm of the `fr-timed-bloom-*` suites (k 100, 1000,
2048; real filters) as [dataset, sweep, n_lists, [n_probe, ...]], merged per (dataset, sweep, n_lists), for the datasets named.

usage: points.py PICKS_YAML DATASET [...] > points.json
"""

import json
import sys

import yaml

picks, wanted = yaml.safe_load(open(sys.argv[1])), set(sys.argv[2:])
pts = {}
for name, suite in picks.items():
    if not name.startswith("fr-timed-bloom-") or "-synth-" in name:
        continue
    for arm in suite["arms"]:
        if arm["algo"] != "silvertorch":
            continue
        (nl,) = arm["build"]["n_lists"]
        for ds, over in arm["datasets"].items():
            if ds in wanted:
                for sw in over["sweeps"]:
                    pts.setdefault((ds, sw, nl), set()).update(over["query"]["n_probe"])
json.dump(
    [[ds, sw, nl, sorted(p)] for (ds, sw, nl), p in sorted(pts.items())], sys.stdout
)
