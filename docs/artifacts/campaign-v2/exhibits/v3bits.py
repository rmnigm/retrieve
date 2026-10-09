"""C2: LiNR V3 by k_bits x candidate pool x pass rate (README.md § v3bits): recall_oracle@100 and graph p50
next to the same pool at the default k_bits = D (V-PILOT, a different code_version: labelled, never one curve).

usage: v3bits.py OUT TREE [TREE ...]
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

from load import cv, load, pass_p, perf, recall


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    recs = [
        r
        for r in load(sys.argv[2:])
        if r["algo"] == "linr_v3"
        and r["dataset"] == "goodreads-synth"
        and r["status"] == "ok"
        and "candidate_pool_frac" in r["params"]
    ]
    g = collections.defaultdict(list)
    for r in recs:
        kb = r["params"].get("k_bits", "D")
        g[(cv(r), r["filter_kind"], kb, r["params"]["candidate_pool_frac"], pass_p(r))].append(r)
    rows = []
    for (c, fk, kb, pool, p), rs in sorted(g.items(), key=lambda x: tuple(map(str, x[0]))):
        row = {
            "code_version": c,
            "filter": fk,
            "k_bits": kb,
            "pool": pool,
            "p": p,
            "seeds": len(rs),
            "recall@100": round(st.median(recall(r) for r in rs), 4),
        }
        for bs in (1, 16):
            v = [perf(r, bs, 100, "graph") for r in rs]
            v = [e for e in v if e]
            row[f"p50_bs{bs}"] = round(st.median(e["median_ms"] for e in v), 4) if v else ""
            row[f"unstable_bs{bs}"] = sum(bool(e.get("unstable")) for e in v)
            row[f"spread_max_bs{bs}"] = round(max(e.get("spread") or 0 for e in v), 3) if v else ""
            row[f"sm_mhz_bs{bs}"] = "/".join(
                sorted({str(int(e["sm_mhz"])) for e in v if e.get("sm_mhz")})
            )
        rows.append(row)
    with open(out / "v3bits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
