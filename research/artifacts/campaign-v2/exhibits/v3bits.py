"""C2: LiNR V3 by k_bits x candidate pool x pass rate (README.md § v3bits): recall_oracle@100 and graph p50
next to the same pool at the default k_bits = D (V-PILOT, a different code_version: labelled, never one curve),
on goodreads-synth and PubMed, with V1 / V2 of the same code_version, box and sweep as the exact reference.

usage: v3bits.py OUT TREE [TREE ...]
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

from load import ALGO, EXACT, box, cv, load, pass_p, perf, recall

DATASETS = ("goodreads-synth", "pubmed")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    allr = [r for r in load(sys.argv[2:]) if r["dataset"] in DATASETS and r["status"] == "ok"]
    v3 = [r for r in allr if r["algo"] == "linr_v3" and "candidate_pool_frac" in r["params"]]
    sel = {(r["dataset"], cv(r), box(r), r["filter_kind"], r["sweep"]) for r in v3}
    ref = [
        r
        for r in allr
        if r["algo"] in EXACT
        and r["backend"] == "triton"
        and (r["dataset"], cv(r), box(r), r["filter_kind"], r["sweep"]) in sel
    ]
    g = collections.defaultdict(list)
    for r in v3 + ref:
        a = "V3" if r["algo"] == "linr_v3" else ALGO[r["algo"]]
        kb = r["params"].get("k_bits", "D") if a == "V3" else ""
        pool = r["params"].get("candidate_pool_frac", "")
        g[(r["dataset"], cv(r), box(r), r["filter_kind"], r["sweep"], a, kb, pool)].append(r)
    rows = []
    for (ds, c, bx, fk, sw, a, kb, pool), rs in sorted(
        g.items(), key=lambda x: tuple(map(str, x[0]))
    ):
        row = {
            "dataset": ds,
            "code_version": c,
            "box": bx,
            "filter": fk,
            "sweep": sw,
            "arm": a,
            "k_bits": kb,
            "pool": pool,
            "p": pass_p(rs[0]) if ds.endswith("-synth") else round(rs[0]["pass_rate"], 6),
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
