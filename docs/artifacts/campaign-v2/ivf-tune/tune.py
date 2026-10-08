"""IVF-TUNE table from the tuning records: per (n_lists, n_probe) recall_oracle@100 (seed 0),
items scanned (n_probe * N / n_lists + n_lists) and seconds per cell; per n_lists the smallest
n_probe reaching 0.95; the pick (fewest items scanned, ties to the smaller n_lists).
`--reached L` prints nothing and exits 0 iff n_lists L has a point at >= 0.95 (the driver's stop).

    python tune.py RECORDS_JSONL [--reached N_LISTS]
"""

import json
import sys

TARGET = 0.95
pts, n_items = {}, None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["status"] != "ok":
        continue
    n_items = r["n_items"]
    nl, npr = r["params"]["n_lists"], r["params"]["n_probe"]
    pts[nl, npr] = (
        r["quality"]["oracle"]["recall@100"],
        r["elapsed_s"],
        r["pass_rate"],
    )
if len(sys.argv) > 2 and sys.argv[2] == "--reached":
    nl = int(sys.argv[3])
    sys.exit(0 if any(v[0] >= TARGET for (a, _), v in pts.items() if a == nl) else 1)


def scanned(nl, npr):
    return npr * n_items / nl + nl


print(f"N {n_items:,}  pass_rate {next(iter(pts.values()))[2]:.5f}")
print("| n_lists | n_probe | recall_oracle@100 | items scanned | s |")
print("|---|---|---|---|---|")
for (nl, npr), (rec, s, _) in sorted(pts.items()):
    print(f"| {nl} | {npr} | {rec:.4f} | {scanned(nl, npr):,.0f} | {s:.0f} |")
n95 = {}
for nl in sorted({a for a, _ in pts}):
    reach = sorted(npr for (a, npr), v in pts.items() if a == nl and v[0] >= TARGET)
    n95[nl] = reach[0] if reach else None
pick = min(((scanned(nl, x), nl, x) for nl, x in n95.items() if x), default=None)
print(json.dumps({"n95": n95, "pick": None if pick is None else
                  {"n_lists": pick[1], "n_probe": pick[2], "items_scanned": round(pick[0])}}))  # fmt: skip
