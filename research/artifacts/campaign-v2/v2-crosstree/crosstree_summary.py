"""Per (group, sweep, bs, mode, arm): old / new median ms per repeat, the paired new / old ratio (pair i =
old-r<i> vs new-r<i>, run back to back) as geometric mean with a t 95 % CI over the pairs, window sm_mhz
ranges; V1 is the control. Then V2's top device kernels per tree from the profile processes.
Usage: python crosstree_summary.py OUT_DIR"""

import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}
out = Path(sys.argv[1])
runs = defaultdict(dict)  # (group, tree) -> {rep: results}
for f in sorted(out.glob("*-r*.json")):
    g, tree, rep = re.match(r"(\w+)-(old|new)-r(\d+)\.json", f.name).groups()
    d = json.loads(f.read_text())
    runs[g, tree][int(rep)] = d["results"]
    print(f.name, d["code_version"][:8], d["retrieve"])
for g in sorted({g for g, _ in runs}):
    print(
        f"\n== {g}: key | old ms per rep | new ms per rep | new/old geomean [95 % CI] over pairs | sm_mhz old / new"
    )
    reps = sorted(set(runs[g, "old"]) & set(runs[g, "new"]))
    for key in runs[g, "old"][reps[0]]:
        o = [runs[g, "old"][r][key]["median_ms"] for r in reps]
        n = [runs[g, "new"][r][key]["median_ms"] for r in reps]
        logs = [math.log(b / a) for a, b in zip(o, n, strict=True)]
        m = st.mean(logs)
        h = (
            T95.get(len(logs) - 1, 2.0) * st.stdev(logs) / math.sqrt(len(logs))
            if len(logs) > 1
            else float("nan")
        )
        mhz = lambda t: [x for r in reps for x in runs[g, t][r][key]["window_sm_mhz"]]  # noqa: E731
        print(
            f"{key} | {' '.join(f'{x:.3f}' for x in o)} | {' '.join(f'{x:.3f}' for x in n)} | "
            f"{math.exp(m):.3f} [{math.exp(m - h):.3f}, {math.exp(m + h):.3f}] | "
            f"{min(mhz('old')):.0f}-{max(mhz('old')):.0f} / {min(mhz('new')):.0f}-{max(mhz('new')):.0f}"
        )
for f in sorted(out.glob("*-prof.json")):
    print(f"\n== {f.name} (V2 graph bs 16, top 5 kernels, us per call)")
    for key, ks in json.loads(f.read_text())["results"].items():
        print(
            key, "; ".join(f"{k['kernel'][:60]} {k['us_per_call']:.1f}" for k in ks[:5])
        )
