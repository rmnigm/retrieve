"""Readout of `tune-t` (PubMed 10 M d768, v2.11): per (sweep, filter kind) and per (mode, bs), every timed point (n_lists, n_probe,
recall_oracle@100, median ms), the recall-latency Pareto front (no other point has recall >= and ms <=, one of them strictly), and
the tuned setting: the fastest point at the highest of the targets 0.80 / 0.90 / 0.95 / 0.99 that any point reaches.
Graph is the reading; eager is listed where timed.

    python pareto.py [RESULTS]   (default /scratch/campaign-v2.11/results)
"""

import collections
import json
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "/scratch/campaign-v2.11/results"
TARGETS = (0.80, 0.90, 0.95, 0.99)

pts = collections.defaultdict(list)
unstable = collections.Counter()
for line in open(f"{R}/tune-t/pubmed-d768.jsonl"):
    r = json.loads(line)
    p, rec = r["params"], r["quality"]["oracle"]["recall@100"]
    for x in r["perf"]:
        if x["k"] == 100 and x["median_ms"]:
            pts[r["sweep"], r["filter_kind"], x["mode"], x["bs"]].append((p["n_lists"], p["n_probe"], rec, x["median_ms"]))
            unstable[x["mode"], x["bs"]] += bool(x.get("unstable"))
print("unstable points per (mode, bs):", dict(unstable))
for (sweep, fk, mode, bs), v in sorted(pts.items()):
    front = sorted((a for a in v if not any(b[2] >= a[2] and b[3] <= a[3] and (b[2] > a[2] or b[3] < a[3]) for b in v)),
                   key=lambda a: a[3])  # fmt: skip
    reached = [t for t in TARGETS if any(a[2] >= t for a in v)]
    tuned = min((a for a in v if a[2] >= reached[-1]), key=lambda a: a[3]) if reached else max(v, key=lambda a: a[2])
    label = f">= {reached[-1]:.2f}" if reached else "none reached: best recall"
    head = f"\n## {sweep} / {fk} / {mode} bs {bs}: tuned ({label})"
    print(f"{head} n_lists {tuned[0]} n_probe {tuned[1]} recall {tuned[2]:.4f} {tuned[3]:.3f} ms")
    print("front: " + "; ".join(f"{a[0]}/{a[1]} {a[2]:.4f} {a[3]:.3f} ms" for a in front))
    for t in TARGETS:
        ok = [a for a in v if a[2] >= t]
        if ok:
            b = min(ok, key=lambda a: a[3])
            print(f"  fastest >= {t:.2f}: {b[0]}/{b[1]} {b[2]:.4f} {b[3]:.3f} ms")
