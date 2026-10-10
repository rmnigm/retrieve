"""C7 outside `h2h` (README.md § c7_filter): official / Triton SilverTorch per cell, like for like.
Official is eager-only (not capturable), so the ratio is eager over eager, paired over the
interleaved windows when both arms ran in one group; Triton graph sits beside it as context, never in
the ratio. Each row names Meta's score path (`load.score_path`: fp16 default vs int32 parity) and
build; ours scores through an int32 dot and an fp32 epilogue on every cell.

usage: c7_filter.py OUT TREE [TREE ...]
"""

import collections
import csv
import sys
from pathlib import Path

from load import box, cv, load, official_build, recall, redo_st_wide2, score_path
from bench import stats


def cell(r):
    p = {k: v for k, v in r["params"].items() if k != "score_path"}
    return (
        cv(r),
        box(r),
        r["suite"],
        r["dataset"],
        r["filter_kind"],
        r["sweep"],
        tuple(sorted(p.items())),
        r["seed"],
    )


def main():
    out = Path(sys.argv[1])
    recs = [
        r
        for r in load(sys.argv[2:])
        if r["algo"] == "silvertorch"
        and r["suite"] != "h2h"
        # a run narrowed to eager (`modes`) or to fewer k / bs is complete for an eager ratio
        and (r["status"] == "ok" or set(r.get("partial_reasons") or ["?"]) <= {"modes", "ks_bs"})
    ]
    by = collections.defaultdict(lambda: {"triton": None, "official": []})
    for r in recs:
        if r["backend"] == "triton":
            by[cell(r)]["triton"] = r
        elif r["backend"] == "official":
            by[cell(r)]["official"].append(r)
    rows = []
    for k, d in sorted(by.items(), key=lambda x: tuple(map(str, x[0]))):
        t = d["triton"]
        for o in d["official"]:
            if t is None:
                continue
            same = (t.get("interleave") or {}).get("group") == (o.get("interleave") or {}).get(
                "group"
            ) and t.get("interleave")
            for eo in o["perf"] or []:
                if eo["mode"] != "eager" or eo.get("median_ms") is None:
                    continue
                pick = {(e["bs"], e["k"], e["mode"]): e for e in t["perf"] or []}
                et, eg = (
                    pick.get((eo["bs"], eo["k"], "eager")),
                    pick.get((eo["bs"], eo["k"], "graph")),
                )
                if not et or et.get("median_ms") is None:
                    continue
                paired = same and len(et["window_medians_ms"]) == len(eo["window_medians_ms"])
                ci = (
                    stats.paired_ratio_ci(eo["window_medians_ms"], et["window_medians_ms"])
                    if paired
                    else None
                )
                rows.append(
                    {
                        "code_version": k[0],
                        "box": k[1],
                        "suite": k[2],
                        "dataset": k[3],
                        "filter": k[4],
                        "sweep": k[5],
                        "params": ",".join(f"{a}={b}" for a, b in k[6]),
                        "seed": k[7],
                        "bs": eo["bs"],
                        "k": eo["k"],
                        "score_path": score_path(o),
                        "official_build": official_build(o),
                        "official_eager_ms": round(eo["median_ms"], 4),
                        "triton_eager_ms": round(et["median_ms"], 4),
                        "official_over_triton_eager": round(eo["median_ms"] / et["median_ms"], 3),
                        "paired_ci": "" if not ci else f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]",
                        "triton_graph_ms": ""
                        if not eg or eg.get("median_ms") is None
                        else round(eg["median_ms"], 4),
                        "triton_redo": redo_st_wide2(t, eo["bs"]),
                        "recall_official": recall(o),
                        "recall_triton": recall(t),
                        "unstable": bool(eo.get("unstable")) + bool(et.get("unstable")),
                    }
                )
    with open(out / "c7-filter.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
