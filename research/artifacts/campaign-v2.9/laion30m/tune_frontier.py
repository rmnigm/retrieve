"""Night-queue item 8, phase 2 on the tuning records (CPU): per (dataset, sweep, filter kind) the recall curve per n_lists, the smallest
n_probe reaching recall_oracle@100 {0.80, 0.90, 0.95, 0.99} and the knee (smallest n_probe within 0.01 of that n_lists' best), the
recall-latency Pareto set over all n_lists per bs (graph median), and the tuned setting = the fastest point at the highest target reached.

    python tune_frontier.py RESULTS_DIR [--json OUT.json]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TARGETS = (0.80, 0.90, 0.95, 0.99)
pts = defaultdict(
    dict
)  # (dataset, sweep, kind) -> {(n_lists, n_probe): (recall, {bs: ms})}
for f in sorted(Path(sys.argv[1]).rglob("*.jsonl")):
    if f.name.endswith(".samples.jsonl"):
        continue
    for line in open(f):
        r = json.loads(line)
        if r["status"] != "ok" and r["partial_reasons"] != [
            "modes"
        ]:  # graph-only runs are partial by modes
            continue
        ms = {p["bs"]: p["median_ms"] for p in r["perf"] or [] if p["mode"] == "graph"}
        key = (r["dataset"], r["sweep"], r["filter_kind"])
        cell = (r["params"]["n_lists"], r["params"]["n_probe"])
        old = pts[key].get(cell, (None, {}))[1]
        pts[key][cell] = (r["quality"]["oracle"]["recall@100"], {**old, **ms})

out = {}
for key in sorted(pts):
    cells, res = (
        pts[key],
        {"curves": {}, "targets": {}, "knees": {}, "pareto": {}, "tuned": {}},
    )
    for nl in sorted({a for a, _ in cells}):
        curve = sorted((npb, cells[nl, npb][0]) for a, npb in cells if a == nl)
        res["curves"][nl] = curve
        best = max(rec for _, rec in curve)
        res["knees"][nl] = next(npb for npb, rec in curve if rec >= best - 0.01)
        res["targets"][nl] = {
            t: next((npb for npb, rec in curve if rec >= t), None) for t in TARGETS
        }
    for bs in sorted({b for _, ms in cells.values() for b in ms}):
        cand = sorted(
            (ms[bs], -rec, nl, npb)
            for (nl, npb), (rec, ms) in cells.items()
            if bs in ms
        )
        front, top = [], -1.0
        for t, negrec, nl, npb in cand:
            if -negrec > top:
                front.append((nl, npb, -negrec, t))
                top = -negrec
        res["pareto"][bs] = front
        reached = [t for t in TARGETS if any(rec >= t for _, _, rec, _ in front)]
        if reached:
            hi = reached[-1]
            nl, npb, rec, t = min((p for p in front if p[2] >= hi), key=lambda p: p[3])
            res["tuned"][bs] = {
                "target": hi,
                "n_lists": nl,
                "n_probe": npb,
                "recall": rec,
                "ms": t,
            }
    out["/".join(key)] = res
    print(f"## {'/'.join(key)}")
    for nl, curve in res["curves"].items():
        tg = " ".join(f"{t}:{res['targets'][nl][t] or '-'}" for t in TARGETS)
        print(
            f"n_lists {nl}: best {max(r for _, r in curve):.4f} knee {res['knees'][nl]} targets {tg}"
        )
    for bs, tuned in res["tuned"].items():
        print(
            f"bs {bs}: tuned {tuned}; pareto {[(a, b, round(c, 4), round(d, 3)) for a, b, c, d in res['pareto'][bs]]}"
        )
if len(sys.argv) > 3 and sys.argv[2] == "--json":
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=str))
