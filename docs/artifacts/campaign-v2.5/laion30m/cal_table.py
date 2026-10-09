"""V-LAION30 calibration table: per record (sweep, arm, n_probe) pass rate, recall_oracle@100, items scanned
(SilverTorch: n_probe * N / n_lists + n_lists), elapsed s per cell, and per timed entry (bs, mode) the median ms and sm_mhz.

    python cal_table.py RESULTS_DIR...
"""

import json
import sys
from pathlib import Path

print(
    "| sweep | arm | n_probe | pass_rate | recall_oracle@100 | items scanned | cell s | bs | mode | median ms | sm_mhz |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    for f in sorted(Path(d).rglob("*.jsonl")):
        if f.name.endswith(".samples.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r["status"] != "ok":
                print(
                    f"| {r['sweep']} | {r['algo']} | {r['params'].get('n_probe', '')} | {r['status']} |||||||"
                )
                continue
            p, n = r["params"], r["n_items"]
            scanned = (
                p["n_probe"] * n / p["n_lists"] + p["n_lists"] if "n_lists" in p else n
            )
            row = (
                f"| {r['sweep']} | {r['algo']} | {p.get('n_probe', '')} | {r['pass_rate']:.5f} "
                f"| {r['quality']['oracle']['recall@100']:.4f} | {scanned:,.0f} | {r['elapsed_s']:.0f} "
            )
            for t in r.get("perf") or [{}]:
                ms = f"{t['median_ms']:.3f}" if "median_ms" in t else ""
                print(
                    f"{row}| {t.get('bs', '')} | {t.get('mode', '')} | {ms} | {t.get('sm_mhz', '')} |"
                )
