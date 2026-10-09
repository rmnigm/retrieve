"""Same-box anchor for crosstree3: pod 1's earlier v2.1 records (Hub artifacts/v-ax-synth-group1) against today's
v2.1 and v2.5 trees, k 100, seed 0, clause triton. Usage: python crosstree3_group1.py GROUP1_JSONL CROSSTREE3_DIR"""

import json
import statistics
import sys
from pathlib import Path

g = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if (r["seed"], r["backend"], r["filter_kind"]) != (0, "triton", "clause"):
        continue
    if r["algo"] in ("linr_v1_filter_mask", "linr_v2") and r["sweep"] in (
        "p0001",
        "p01",
        "p1",
    ):
        for p in r["perf"]:
            if p["k"] == 100:
                g[f"{r['sweep']}/bs{p['bs']}/{p['mode']}/{r['algo']}"] = p["median_ms"]
ct = {}
for f in sorted(Path(sys.argv[2]).glob("ax3-v2[15]-r*.json")):
    for k, v in json.loads(f.read_text())["results"].items():
        ct.setdefault((f.name[4:7], k), []).append(v["median_ms"])
for k, ms in sorted(g.items()):
    v21, v25 = (statistics.median(ct[(t, k)]) for t in ("v21", "v25"))
    print(
        f"{k:40s} group1 v2.1 {ms:.3f} | today v2.1 {v21:.3f} ({v21 / ms:.3f}) | today v2.5 {v25:.3f} ({v25 / ms:.3f})"
    )
