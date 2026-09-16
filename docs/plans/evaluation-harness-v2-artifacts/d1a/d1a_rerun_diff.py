#!/usr/bin/env python3
"""Compare the `quality` block of every rerun record in <dir> against the stage-a campaign
record with the same key. Byte-identical = `json.dumps(..., sort_keys=True)` equal, with the
parity fields (`jaccard_vs_first@100`, `score_max_abs_diff`, `parity`) dropped: the rerun
writes its own spill file, so it is always its group's reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEY = ("dataset", "dim", "suite", "filter_kind", "sweep", "algo", "backend", "params", "seed")
DROP = ("jaccard_vs_first@100", "score_max_abs_diff", "parity")
RESULTS = Path(__file__).resolve().parents[4] / "evaluation" / "results" / "filter"


def key_of(r):
    return tuple(json.dumps(r[f], sort_keys=True) if f == "params" else r[f] for f in KEY)


def qual(r):
    return json.dumps(
        {k: v for k, v in (r.get("quality") or {}).items() if k not in DROP}, sort_keys=True
    )


def main() -> int:
    base = {}
    for p in RESULTS.glob("*-d128.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["status"] == "ok":
                    base[key_of(r)] = r
    bad = 0
    n = 0
    for p in sorted(Path(sys.argv[1]).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = key_of(r)
            if k not in base:
                print(f"NO CAMPAIGN RECORD for {k}")
                bad += 1
                continue
            n += 1
            same = qual(r) == qual(base[k])
            bad += not same
            print(f"{'IDENTICAL' if same else 'DIFFERS  '}  {k[0]} {k[5]} {k[6]} {k[3]}/{k[4]} "
                  f"{k[7]}")
            if not same:
                a, b = json.loads(qual(base[k])), json.loads(qual(r))
                for kk in sorted(set(a) | set(b)):
                    if a.get(kk) != b.get(kk):
                        print(f"     {kk}: campaign {a.get(kk)!r} != rerun {b.get(kk)!r}")
    print(f"\n{n} cells rerun: {'PASS' if not bad else f'FAIL ({bad})'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
