"""Summarise interleave.sh output: per case the median over rounds of each side's median, the
new/base ratio, the noise band (the larger side's max-min over its rounds, relative), and the
sampled SM clock range and unstable flags of both sides.

    python compare.py OUTDIR > table.md
"""

import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
runs = {"base": [], "new": []}
for p in sorted(out.glob("*.json")):
    runs[p.stem.split("-")[0]].append(json.loads(p.read_text()))

print(
    "| case | base µs | new µs | new/base | noise band | base sm MHz | new sm MHz | unstable (base/new) |"
)
print("|---|---|---|---|---|---|---|---|")
for case in runs["base"][0]["cases"]:
    row = {}
    for side, rs in runs.items():
        meds = [r["cases"][case]["us_median"] for r in rs if case in r["cases"]]
        mhz = [m for r in rs if case in r["cases"] for m in r["cases"][case]["sm_mhz"]]
        row[side] = (
            statistics.median(meds),
            (max(meds) - min(meds)) / statistics.median(meds),
            (min(mhz), max(mhz)),
            sum(r["cases"][case]["unstable"] for r in rs if case in r["cases"]),
            len(meds),
        )
    b, n = row["base"], row["new"]
    print(
        f"| {case} | {b[0]:.1f} | {n[0]:.1f} | {n[0] / b[0]:.3f} | {max(b[1], n[1]):.3f} | "
        f"{b[2][0]}-{b[2][1]} | {n[2][0]}-{n[2][1]} | {b[3]}/{b[4]} , {n[3]}/{n[4]} |"
    )
