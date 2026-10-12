"""Three-tree V2 check: per (sweep, bs, mode, arm), each tree's median ms per round and each tree pair's paired ratio (round i of
both trees) as geomean with a t 95 % CI over the rounds, window sm_mhz ranges, and whether every tree's ids hash is the same.
Usage: python crosstree3_summary.py OUT_DIR"""

import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}
out = Path(sys.argv[1])
runs = defaultdict(dict)
for f in sorted(out.glob("ax3-*-r*.json")):
    tree, rep = re.match(r"ax3-(v\d+)-r(\d+)\.json", f.name).groups()
    d = json.loads(f.read_text())
    runs[tree][int(rep)] = d["results"]
    print(f.name, d["code_version"][:8], d["retrieve"])
trees = sorted(runs)
reps = sorted(set.intersection(*(set(runs[t]) for t in trees)))
pairs = [("v24", "v21"), ("v25", "v24"), ("v25", "v21")]
print(f"\n{len(reps)} rounds; ratio = first / second tree, geomean [t 95 % CI]")
for key in runs[trees[0]][reps[0]]:
    med = {t: [runs[t][r][key]["median_ms"] for r in reps] for t in trees}
    ids = {runs[t][r][key].get("ids_sha256") for t in trees for r in reps}
    mhz = [x for t in trees for r in reps for x in runs[t][r][key]["window_sm_mhz"]]
    cells = []
    for a, b in pairs:
        logs = [math.log(x / y) for x, y in zip(med[a], med[b], strict=True)]
        m = st.mean(logs)
        h = (
            T95.get(len(logs) - 1, 2.0) * st.stdev(logs) / math.sqrt(len(logs))
            if len(logs) > 1
            else float("nan")
        )
        cells.append(
            f"{a}/{b} {math.exp(m):.3f} [{math.exp(m - h):.3f}, {math.exp(m + h):.3f}]"
        )
    print(f"{key:28} " + " ".join(f"{t} {st.median(med[t]):.3f}" for t in trees) + " | " + " | ".join(cells)
          + f" | ids {'same' if len(ids) == 1 else f'{len(ids)} distinct'} | MHz {min(mhz):.0f}-{max(mhz):.0f}")  # fmt: skip
