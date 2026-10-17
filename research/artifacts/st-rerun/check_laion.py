"""The restaged LAION inputs against the final run's records, before any timing.

usage (from evaluation/): python ../research/artifacts/st-rerun/check_laion.py FR_DIR...

For every LAION (dataset, sweep) the rerun reads: the final run's seed-0 records (from the fetched `fr/`
trees) give `n_items`, `n_kept`, `pass_rate` and, in their per-query sidecars, each kept query's exact
pass count; the oracle blob `bench oracle` built on the restaged data (data/<dataset>/gt_d256) gives
the same. Equal means equal: counts exact, pass rate to the last bit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

fr = {}
for tree in map(Path, sys.argv[1:]):
    for f in tree.glob("*/*.jsonl"):
        if f.name.endswith(".samples.jsonl"):
            continue
        for line in open(f):
            r = json.loads(line)
            if r["status"] != "ok" or r["seed"] != 0 or not r["dataset"].startswith("laion") or not r.get("per_query"):
                continue
            fr.setdefault((r["dataset"], r["sweep"]), (r, tree / r["per_query"]))

DATA = {"laion1m": "laion1m", "laion3m": "laion3m", "laion10m": "laion10m", "laion30m": "laion30m"}
bad = 0
for (ds, sweep), (r, pq) in sorted(fr.items()):
    base = ds.removesuffix("-synth")
    blobs = sorted(Path(f"data/{DATA[base]}/gt_d256").glob(f"oracle_v4_{sweep}_*.pt"))
    if not blobs:
        print(f"{ds} {sweep}: no blob built (not in the rerun)")
        continue
    side = np.load(pq)
    for b in blobs:
        blob = torch.load(b, weights_only=False)
        counts = blob["pass_counts"].numpy()
        kept = np.flatnonzero(counts >= 0)
        ok = (
            blob["n_items"] == r["n_items"]
            and blob["n_kept"] == r["n_kept"]
            and np.array_equal(kept, side["rows"])
            and np.array_equal(counts[kept], side["pass_count"])
            and blob["pass_rate"] == r["pass_rate"]
        )
        bad += not ok
        print(
            f"{ds} {sweep} {b.name}: n_items {blob['n_items']} / {r['n_items']}, n_kept {blob['n_kept']} / {r['n_kept']}, "
            f"pass_rate {blob['pass_rate']!r} / {r['pass_rate']!r}, per-query counts "
            f"{'equal' if np.array_equal(counts[kept], side['pass_count']) else 'DIFFER'} -> {'EQUAL' if ok else 'MISMATCH'}"
        )
print("all equal" if not bad else f"{bad} MISMATCH")
sys.exit(bool(bad))
