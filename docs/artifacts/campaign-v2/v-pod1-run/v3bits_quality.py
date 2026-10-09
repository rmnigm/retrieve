"""V-V3BITS quality table from a v3bits tree: recall_oracle@100 / @1000 per (dataset, sweep, filter kind, pool frac),
k_bits 64 vs 128, median over seeds. Usage: python v3bits_quality.py TREE"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(list)
for f in sorted(Path(sys.argv[1]).glob("v3bits/*.jsonl")):
    if f.name.endswith(".samples.jsonl"):
        continue
    for line in f.read_text().splitlines():
        r = json.loads(line)
        q = (r.get("quality") or {}).get("oracle") or {}
        key = (r["dataset"], r["sweep"], r["filter_kind"], r["params"]["candidate_pool_frac"], r["params"]["k_bits"])
        rows[key].append((q.get("recall@100"), q.get("recall@1000"), r["status"]))
print("dataset sweep kind pool | k_bits 64 r@100 r@1000 | k_bits 128 r@100 r@1000 | statuses")
for key in sorted({k[:4] for k in rows}):
    cells = []
    for kb in (64, 128):
        v = rows.get((*key, kb), [])
        med = [st.median(x[i] for x in v if x[i] is not None) if any(x[i] is not None for x in v) else None for i in (0, 1)]
        cells.append(" ".join("-" if m is None else f"{m:.4f}" for m in med))
    stat = sorted({x[2] for kb in (64, 128) for x in rows.get((*key, kb), [])})
    print(" ".join(map(str, key)), "|", cells[0], "|", cells[1], "|", ",".join(stat))
