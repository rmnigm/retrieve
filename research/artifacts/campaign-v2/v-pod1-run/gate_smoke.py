"""The v2.3 candidate's smoke cells: per record, status, code_version, eager and graph median ms, the graph entry's
capture (a median and no `reason`), eager == graph ids. Exits 1 when any record failed or any graph entry is missing.
Usage: python gate_smoke.py TREE"""

import json
import sys
from pathlib import Path

bad = 0
for f in sorted(Path(sys.argv[1]).glob("filter/*-d128.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        perf = {p["mode"]: p for p in r.get("perf") or []}
        e, g = perf.get("eager", {}), perf.get("graph", {})
        captured = g.get("median_ms") is not None and not g.get("reason")
        same = e.get("ids_sha256") is not None and e.get("ids_sha256") == g.get(
            "ids_sha256"
        )
        bad += r["status"] != "ok" or not captured
        print(
            r["algo"], r["filter_kind"], r["params"], r["status"], r["env"]["code_version"][:8],
            f"eager {e.get('median_ms')} graph {g.get('median_ms')}", "captured" if captured else f"NOT captured {g.get('reason')}",
            "ids eager==graph" if same else "ids differ", f"recall@100 {(r['quality'].get('oracle') or {}).get('recall@100')}",
        )  # fmt: skip
sys.exit(1 if bad else 0)
