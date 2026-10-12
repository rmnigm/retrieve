"""The fixed-p scaling view across N (README.md § scaling): uniform synth legs at 0.8 / 3 / 10 / 30 M.

usage: scaling.py OUT TREE_OR_LEG [...]

Per (pass rate p, dataset): SilverTorch Triton clause recall@100 at n_probe 24 and at the n_probe nearest to
6.25 % of the lists scanned; V2 / V1 and SilverTorch(n_probe 24) / V1 graph p50 ratios at bs 1 and 16
**inside one leg** (one box). Absolute latencies are never compared across N: the scales ran on different
boxes (decisions, 2026-10-10). Every row says its box, code_version, and whether its clause timings are
pre-CLAUSE-SKIP (10-clause tables) or its clock fields unknown (pod d GPU 1).
"""

import collections
import csv
import sys
from pathlib import Path

from load import box, clock_unknown, cv, load, pass_p, perf, pre_clause_skip, recall

SYNTH = ("goodreads-synth", "arxiv-synth", "yfcc10m-synth", "laion30m-synth")


def g(r, bs):
    e = perf(r, bs, 100, "graph")
    return e["median_ms"] if e else None


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    recs = [
        r
        for r in load(sys.argv[2:])
        if r["dataset"] in SYNTH
        and r["filter_kind"] == "clause"
        and r["status"] in ("ok", "partial")
        and r["seed"] == 0
    ]
    legs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        legs[(r["dataset"], r["n_items"], r["dim"], cv(r), box(r), pass_p(r))][r["algo"]].append(r)
    rows = []
    for (ds, n, d, c, bx, p), arms in sorted(
        legs.items(), key=lambda x: (x[0][1], x[0][5], x[0][3])
    ):
        st = {
            r["params"]["n_probe"]: r
            for r in arms.get("silvertorch", [])
            if r["backend"] == "triton"
        }
        n_lists = next((r["params"].get("n_lists", 1024) for r in st.values()), None)
        r24 = st.get(24)
        frac = {np_: np_ / n_lists for np_ in st} if n_lists else {}
        near = min(frac, key=lambda x: abs(frac[x] - 0.0625)) if frac else None
        v1 = next(
            (r for r in arms.get("linr_v1_filter_mask", []) if r["backend"] == "triton"), None
        )
        v2 = next((r for r in arms.get("linr_v2", []) if r["backend"] == "triton"), None)
        row = {
            "p": p,
            "N_M": round(n / 1e6, 2),
            "d": d,
            "dataset": ds,
            "code_version": c,
            "box": bx,
            "n_lists": n_lists,
            "st_recall_np24": "" if not r24 else round(recall(r24), 4),
            "st_np_at_6pct": "" if near is None else f"{near} ({frac[near]:.2%})",
            "st_recall_at_6pct": "" if near is None else round(recall(st[near]), 4),
            "v1_recall": "" if not v1 else round(recall(v1), 4),
        }
        for bs in (1, 16):
            row[f"v2_over_v1_bs{bs}"] = (
                round(g(v2, bs) / g(v1, bs), 3) if v1 and v2 and g(v1, bs) and g(v2, bs) else ""
            )
            row[f"st24_over_v1_bs{bs}"] = (
                round(g(r24, bs) / g(v1, bs), 3) if v1 and r24 and g(v1, bs) and g(r24, bs) else ""
            )
        rs = [r for a in arms.values() for r in a]
        row["flags"] = " ".join(
            f
            for f, on in (
                ("pre-CLAUSE-SKIP", any(pre_clause_skip(r) for r in rs)),
                ("clock-unknown", any(clock_unknown(r) for r in rs)),
            )
            if on
        )
        rows.append(row)
    cols = list(rows[0])
    with open(out / "scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Fixed-p scaling across N (uniform synth, NOT CITABLE)\n",
        "Ratios are inside one leg (one box); absolute latencies are not compared across N.\n",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "scaling.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
