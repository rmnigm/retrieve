"""Controller rule for night-queue item 8 (2026-10-12): run n_lists 65536 only if 32768 lands on the Pareto front at some recall target
where 16384 does not, i.e. for some (dataset, sweep, filter kind, bs) and target in {0.80, 0.90, 0.95, 0.99} the fastest timed cell
reaching the target has n_lists 32768. Prints each such case and exits 0 when 65536 should run, 1 otherwise.

    python tune_need_65536.py RESULTS_DIR
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TARGETS = (0.80, 0.90, 0.95, 0.99)
cells = defaultdict(
    list
)  # (dataset, sweep, kind, bs) -> [(ms, recall, n_lists, n_probe)]
for f in sorted(Path(sys.argv[1]).rglob("*.jsonl")):
    if f.name.endswith(".samples.jsonl"):
        continue
    for line in open(f):
        r = json.loads(line)
        if r["status"] != "ok" and r["partial_reasons"] != [
            "modes"
        ]:  # graph-only runs are partial by modes
            continue
        for p in r["perf"] or []:
            if p["mode"] == "graph":
                key = (r["dataset"], r["sweep"], r["filter_kind"], p["bs"])
                rec = r["quality"]["oracle"]["recall@100"]
                cells[key].append(
                    (
                        p["median_ms"],
                        rec,
                        r["params"]["n_lists"],
                        r["params"]["n_probe"],
                    )
                )
wins = []
for key, pts in sorted(cells.items()):
    for t in TARGETS:
        ok = [c for c in pts if c[1] >= t]
        if ok:
            best = min(ok)
            print(
                f"{'/'.join(map(str, key))} target {t}: fastest n_lists {best[2]} n_probe {best[3]} ({best[0]:.3f} ms, recall {best[1]:.4f})"
            )
            if best[2] == 32768:
                wins.append((key, t))
print(
    f"32768 fastest at {len(wins)} (key, target) cases: {'run 65536' if wins else 'skipped: 32768 not on the front'}"
)
sys.exit(0 if wins else 1)
