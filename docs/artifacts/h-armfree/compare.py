"""H-ARMFREE gate (from H-QLOOP's): every record of tree A has a record with the same key in tree B whose `quality` is identical (JSON,
key order normalised) and whose per-query sidecar holds bit-identical arrays (same names, dtype, shape and raw bytes; the
.npz container itself carries zip timestamps, so its file bytes differ run to run). Exits 1 on any difference or missing record.
Usage: python compare.py TREE_A TREE_B"""

import json
import sys

import numpy as np
from pathlib import Path

from bench import records

a_root, b_root = map(Path, sys.argv[1:])
bad = n = 0
for fa in sorted(a_root.glob("*/*-d*.jsonl")):
    if fa.name.endswith(".samples.jsonl"):
        continue
    fb = b_root / fa.relative_to(a_root)
    b = (
        {records.record_key(r): r for r in records.read_records(fb)}
        if fb.exists()
        else {}
    )
    for ra in records.read_records(fa):
        n += 1
        key = records.record_key(ra)
        rb = b.get(key)
        tag = f"{fa.parent.name}/{fa.stem} {ra['algo']}/{ra['backend']} {ra['filter_kind']} {ra['sweep']} {ra['params']}"
        if rb is None:
            print("MISSING in B:", tag)
            bad += 1
            continue
        same_q = json.dumps(ra["quality"], sort_keys=True) == json.dumps(
            rb["quality"], sort_keys=True
        )
        pa, pb = ra.get("per_query"), rb.get("per_query")
        files = 0
        same_pq = pa == pb
        if same_pq and pa:
            with np.load(a_root / pa) as za, np.load(b_root / pb) as zb:
                files = len(za.files)
                same_pq = sorted(za.files) == sorted(zb.files) and all(
                    za[n].dtype == zb[n].dtype and za[n].shape == zb[n].shape
                    and za[n].tobytes() == zb[n].tobytes() for n in za.files
                )  # fmt: skip
        ok = same_q and same_pq and ra["status"] == rb["status"]
        bad += not ok
        print("OK  " if ok else "DIFF", tag, f"quality {'=' if same_q else '!='}", f"sidecar arrays {files} {'=' if same_pq else '!='}",
              f"elapsed A {ra['elapsed_s']:.0f}s B {rb['elapsed_s']:.0f}s")  # fmt: skip
print(f"{n} records, {bad} differing or missing")
sys.exit(1 if bad else 0)
