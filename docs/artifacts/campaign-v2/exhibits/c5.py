"""C5: co-design (`bloom_path` partial) vs full mask, per backend (README.md § c5).

usage: c5.py OUT TREE [TREE ...]

Paired full / partial ratio of p50 over the interleaved rounds (seed x window, `stats.paired_ratio_ci`)
per (code_version, backend, dataset, sweep, n_probe, bs, mode); > 1 means co-design is faster, the
paper's direction (S-11: 1.79-2.15x). Also: recall equality between the paths (S-13), and the
clocks behind any `unstable` flag (spread, sampled SM MHz).
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

from load import cv, load, official_build, recall, redo, score_path

from bench import stats


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    recs = [
        r for r in load(sys.argv[2:]) if r["suite"].startswith("codesign") and r["status"] == "ok"
    ]
    pairs = collections.defaultdict(dict)
    for r in recs:
        k = (
            cv(r),
            r["backend"],
            r["dataset"],
            r["sweep"],
            r["params"]["n_probe"],
            r["params"].get("n_lists"),
            r["seed"],
            official_build(r),
            score_path(r),
        )
        pairs[k][r["params"]["bloom_path"]] = r
    cells = collections.defaultdict(
        lambda: {
            "f": [],
            "p": [],
            "rec": [],
            "mhz": set(),
            "spread": [],
            "unst": 0,
            "prep_f": [],
            "prep_p": [],
        }
    )
    for k, d in pairs.items():
        if set(d) != {"full", "partial"}:
            continue
        f, p = d["full"], d["partial"]
        same_group = (f.get("interleave") or {}).get("group") == (p.get("interleave") or {}).get(
            "group"
        )
        for ef in f["perf"] or []:
            ep = next(
                (
                    e
                    for e in p["perf"] or []
                    if (e["bs"], e["k"], e["mode"]) == (ef["bs"], ef["k"], ef["mode"])
                ),
                None,
            )
            if not ep or ef.get("median_ms") is None or ep.get("median_ms") is None:
                continue
            c = cells[k[:6] + (ef["bs"], ef["mode"], k[7], k[8])]
            if same_group and len(ef["window_medians_ms"]) == len(ep["window_medians_ms"]):
                c["f"] += ef["window_medians_ms"]
                c["p"] += ep["window_medians_ms"]
            c["rec"].append((recall(f), recall(p)))
            c["redo"] = redo(f, ef["bs"], ef["k"])
            c["mhz"] |= {
                int(x)
                for x in (ef.get("window_sm_mhz") or []) + (ep.get("window_sm_mhz") or [])
                if x
            }
            c["spread"] += [ef.get("spread") or 0, ep.get("spread") or 0]
            c["unst"] += bool(ef.get("unstable")) + bool(ep.get("unstable"))
            # from v2.6 the query-side filter encoding is out of the timed forward (evaluation.md)
            for side, e in (("prep_f", ef), ("prep_p", ep)):
                if e.get("query_prep_ms") is not None:
                    c[side].append(e["query_prep_ms"])
    rows = []
    for k, c in sorted(cells.items(), key=lambda x: tuple(map(str, x[0]))):
        ci = stats.paired_ratio_ci(c["f"], c["p"]) if c["f"] else None
        eq = sum(a == b for a, b in c["rec"])
        rows.append(
            {
                "code_version": k[0],
                "backend": k[1],
                "dataset": k[2],
                "sweep": k[3],
                "n_probe": k[4],
                "n_lists": k[5],
                "bs": k[6],
                "mode": k[7],
                "official_build": k[8],
                "score_path": k[9],
                "triton_redo": c.get("redo", ""),
                "rounds": len(c["f"]),
                "full_over_partial": "" if not ci else round(ci[0], 3),
                "ci_lo": "" if not ci else round(ci[1], 3),
                "ci_hi": "" if not ci else round(ci[2], 3),
                "differs": "" if not ci else stats.differs(ci),
                "p50_full_ms": round(st.median(c["f"]), 4) if c["f"] else "",
                "p50_partial_ms": round(st.median(c["p"]), 4) if c["p"] else "",
                "recall_equal": f"{eq}/{len(c['rec'])}",
                "unstable_entries": c["unst"],
                "spread_max": round(max(c["spread"]), 3),
                "sm_mhz": "/".join(map(str, sorted(c["mhz"]))),
                "query_prep_ms_full": round(st.median(c["prep_f"]), 4) if c["prep_f"] else "",
                "query_prep_ms_partial": round(st.median(c["prep_p"]), 4) if c["prep_p"] else "",
            }
        )
    with open(out / "c5.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
