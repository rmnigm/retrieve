"""Distinct string values per key path across every record / samples line and every JSON sidecar."""
import json
import sys
from collections import defaultdict
from pathlib import Path

FR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/work/hub/eval-results/fr")
vals = defaultdict(set)


def walk(o, path):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        if o and isinstance(o[0], (dict, list, str)):
            for v in o[:50]:
                walk(v, path + "[]")
    elif isinstance(o, str):
        vals[path].add(o)


for p in FR.rglob("*"):
    if not p.is_file() or "logs" in p.parts:
        continue
    kind = "samples" if p.name.endswith(".samples.jsonl") else "jsonl" if p.suffix == ".jsonl" else "json" if p.suffix == ".json" and p.name != "MANIFEST.json" else None
    if kind is None:
        continue
    lines = p.open() if kind != "json" else [p.read_text()]
    for line in lines:
        walk(json.loads(line), kind if kind != "json" else "json:" + p.name.split("-")[0])
for k in sorted(vals):
    v = sorted(vals[k])
    if k.endswith(".per_query") or k.endswith("started"):
        print(f"{k}: {len(v)} values, e.g. {v[0]}")
        continue
    print(f"{k}: {len(v)}: {v[:12]}{' …' if len(v) > 12 else ''}")
