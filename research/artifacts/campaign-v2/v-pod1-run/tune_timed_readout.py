"""Readout for the v2.11 timed tune: joins each timed cell (graph bs 1/16/64, eager bs 16, median ms) with its oracle recall@100
from the quality grids (recall is unchanged by ST-TOPK, bit-exact), prints per (dataset, sweep, kind) the recall-latency frontier
at graph bs 16 and the fastest setting reaching each recall target. Usage: python tune_timed_readout.py TIMED_DIR QUALITY_JSONL..."""

import json
import sys
from pathlib import Path

TARGETS = (0.80, 0.90, 0.95, 0.99)
key = lambda r: (r["dataset"], r["sweep"], r["filter_kind"], r["params"]["n_lists"], r["params"]["n_probe"])  # noqa: E731
recall = {}
for f in sys.argv[2:]:
    for line in open(f):
        r = json.loads(line)
        if r.get("quality") and r["quality"].get("oracle") and r["algo"] == "silvertorch":
            recall[key(r)] = r["quality"]["oracle"]["recall@100"]
rows = {}
for f in sorted(Path(sys.argv[1]).glob("*-d*[0-9].jsonl")):
    for line in open(f):
        r = json.loads(line)
        ms = {(p["mode"], p["bs"]): p["median_ms"] for p in r["perf"]}
        rows.setdefault(key(r)[:3], []).append((key(r)[3:], recall.get(key(r)), ms, r["unstable"]))
missing = 0
for g, cells in sorted(rows.items()):
    cells.sort(key=lambda c: c[2][("graph", 16)])
    missing += sum(c[1] is None for c in cells)
    front, best = [], -1.0
    for c in cells:
        if c[1] is not None and c[1] > best:
            front.append(c)
            best = c[1]
    print(f"\n{' '.join(g)}  cells {len(cells)}  frontier (graph bs16):")
    for (nl, npb), rc, ms, _ in front:
        print(f"  nl {nl:5d} np {npb:5d}  R {rc:.4f}  g1 {ms[('graph', 1)]:.3f}  g16 {ms[('graph', 16)]:.3f}  "
              f"g64 {ms[('graph', 64)]:.3f}  e16 {ms[('eager', 16)]:.3f}")  # fmt: skip
    for t in TARGETS:
        hit = [c for c in cells if c[1] is not None and c[1] >= t]
        if hit:
            for bs in (1, 16, 64):
                (nl, npb), rc, ms, _ = min(hit, key=lambda c: c[2][("graph", bs)])
                print(f"  R>={t:.2f} bs{bs:<2d}: nl {nl} np {npb} ({ms[('graph', bs)]:.3f} ms, R {rc:.4f})")
        else:
            print(f"  R>={t:.2f}: not reached")
print(f"\ncells without a recall match: {missing}")
