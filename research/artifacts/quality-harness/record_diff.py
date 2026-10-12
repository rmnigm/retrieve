"""Compare the records of two results trees with timing and provenance stripped: same keys in the
same order, same values. Usage: record_diff.py <before-dir> <after-dir>."""

import json
import sys
from pathlib import Path

TIMING = {"elapsed_s", "build_s", "compile_s", "memory_reserved_mib", "unstable", "env", "perf"}
PERF_KEEP = ("k", "bs", "mode", "query_prep_ms", "ids_sha256", "ids_sha256_canon")


def load(root):
    recs = [json.loads(ln) for p in sorted(Path(root).glob("*/*.jsonl"))
            if not p.name.endswith(".samples.jsonl") for ln in p.read_text().splitlines() if ln]
    out = []
    for r in recs:
        perf = [{k: e.get(k) for k in PERF_KEEP if k != "query_prep_ms"} for e in r.get("perf") or []]
        out.append(({k: v for k, v in r.items() if k not in TIMING}, list(r), perf))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
assert len(a) == len(b) and a, (len(a), len(b))
for (ra, ka, pa), (rb, kb, pb) in zip(a, b, strict=True):
    assert ka == kb, ("key order", ka, kb)
    assert ra == rb, ("values", {k for k in ra if ra[k] != rb.get(k)})
    assert pa == pb, ("perf ids", pa, pb)
print(f"{len(a)} record(s): same keys in the same order, same values (timing/env stripped), same ids_sha256")
