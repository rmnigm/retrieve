"""v2.6 regression split: paired ratios v26 / v25, v26p / v25, v26p / v26 (geomean, t 95 % CI over rounds), ids sameness and MHz;
per profiled key the device kernels whose µs per call differ by more than 5 µs between v2.5 and v2.6 (prepared); the two trees'
8-cell `bench run` medians side by side. Usage: python crosstree4_summary.py OUT_DIR"""

import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}
out = Path(sys.argv[1])
runs = defaultdict(dict)
for f in sorted(out.glob("ax3-*-r*.json")):
    tree, rep = re.match(r"ax3-(v\d+p?)-r(\d+)\.json", f.name).groups()
    d = json.loads(f.read_text())
    runs[tree][int(rep)] = d["results"]
    print(f.name, d["code_version"][:8], d["prep"])
trees = sorted(runs)
reps = sorted(set.intersection(*(set(runs[t]) for t in trees)))
pairs = [("v26", "v25"), ("v26p", "v25"), ("v26p", "v26")]
print(f"\n{len(reps)} rounds; ratio = first / second, geomean [t 95 % CI]")
for key in runs[trees[0]][reps[0]]:
    med = {t: [runs[t][r][key]["median_ms"] for r in reps] for t in trees}
    ids = {runs[t][r][key].get("ids_sha256") for t in trees for r in reps}
    mhz = [x for t in trees for r in reps for x in runs[t][r][key]["window_sm_mhz"]]
    cells = []
    for a, b in pairs:
        logs = [math.log(x / y) for x, y in zip(med[a], med[b], strict=True)]
        m, h = st.mean(logs), T95[len(logs) - 1] * st.stdev(logs) / math.sqrt(len(logs))
        cells.append(
            f"{a}/{b} {math.exp(m):.3f} [{math.exp(m - h):.3f}, {math.exp(m + h):.3f}]"
        )
    print(f"{key:36} " + " ".join(f"{t} {st.median(med[t]):.3f}" for t in trees) + " | " + " | ".join(cells)
          + f" | ids {'same' if len(ids) == 1 else f'{len(ids)} distinct'} | MHz {min(mhz):.0f}-{max(mhz):.0f}")  # fmt: skip

print(
    "\nprofile, bs 16: device kernels differing by > 5 us per call, v25 -> v26p (us per call, calls)"
)
prof = {
    t: json.loads((out / f"prof-{t}.json").read_text())["results"]
    for t in ("v25", "v26", "v26p")
}
for key in prof["v25"]:
    k = {t: {e["kernel"]: e for e in prof[t][key]} for t in prof}
    tot = {t: sum(e["us_per_call"] for e in k[t].values()) for t in k}
    print(
        f"{key:36} total us v25 {tot['v25']:.1f} v26 {tot['v26']:.1f} v26p {tot['v26p']:.1f}"
    )
    for name in sorted(set(k["v25"]) | set(k["v26p"])):
        a, b = k["v25"].get(name), k["v26p"].get(name)
        ua, ub = (a or {}).get("us_per_call", 0.0), (b or {}).get("us_per_call", 0.0)
        if abs(ub - ua) > 5:
            print(
                f"    {name[:90]:90} {ua:9.1f} -> {ub:9.1f}  calls {(a or {}).get('calls', 0)} -> {(b or {}).get('calls', 0)}"
            )

print("\nbench run, median ms: v25 tree | v26 tree | ratio")
bench = {}
for t in ("v25", "v26"):
    for line in (
        (out / f"bench-{t}/synth/arxiv-synth-d128.jsonl").read_text().splitlines()
    ):
        r = json.loads(line)
        for p in r["perf"]:
            bench.setdefault(f"{r['sweep']}/bs{p['bs']}/{p['mode']}/{r['algo']}", {})[
                t
            ] = p["median_ms"]
for key, v in sorted(bench.items()):
    print(f"{key:40} {v['v25']:.3f} | {v['v26']:.3f} | {v['v26'] / v['v25']:.3f}")
