"""V2 before / after Fix A by pass rate (README.md § v2_old_new): the old-side V1 / V2 table from the
fetched trees and the V2-FIX-A artifact's ABAB ratios and `programs` sweep.

usage: v2_old_new.py TREE_ROOT FIX_A_DIR    (TREE_ROOT holds v2/ and d1/; FIX_A_DIR = artifacts/v2-fix-a)
"""

import collections
import json
import statistics as st
import sys
from pathlib import Path

LEGS = [
    "d1/filter/arxiv-d128",
    "d1/filter/pubmed-d768",
    "d1/filter/yfcc10m-d192",
    "v2/filter/goodreads-d128",
    "v2/synth/goodreads-synth-d128",
]
ARMS = ("linr_v1_filter_mask", "linr_v2")


def med(x):
    return st.median(x) if x else float("nan")


def old_side(root):
    print("## old side: k 100 p50 ms (graph), V2/V1 in the same run")
    print(
        "| leg | N | d | filter | sweep | pass | V1 bs1 | V2 bs1 | V1 bs16 | V2 bs16 | V2/V1 bs1 | V2/V1 bs16 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for leg in LEGS:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in open(root / f"{leg}.jsonl"):
            r = json.loads(line)
            if r["status"] != "ok" or r["algo"] not in ARMS or r["backend"] != "triton":
                continue
            c = g[(r["n_items"], r["dim"], r["filter_kind"], r["sweep"])]
            c["p"].append(r["pass_rate"])
            for e in r["perf"]:
                if e["k"] == 100 and e["mode"] == "graph" and e.get("median_ms"):
                    c[(r["algo"], e["bs"])].append(e["median_ms"])
        for (n, d, fk, sw), c in sorted(g.items(), key=lambda x: (x[0][2], med(x[1]["p"]))):
            v = {(a, bs): med(c[(a, bs)]) for a in ARMS for bs in (1, 16)}
            print(
                f"| {leg.split('/')[0]} | {n:,} | {d} | {fk} | {sw} | {med(c['p']):.4f} "
                + " | ".join(f"{v[(a, bs)]:.3f}" for bs in (1, 16) for a in ARMS)
                + f" | {v[('linr_v2', 1)] / v[('linr_v1_filter_mask', 1)]:.2f}"
                + f" | {v[('linr_v2', 16)] / v[('linr_v1_filter_mask', 16)]:.2f} |"
            )


def fix_a(d):
    g = json.load(open(d / "gate_abab.json"))
    print(
        f"\n## V2-FIX-A ABAB, goodreads-synth 0.8 M d128 (old 408b1188 vs tree {g['env']['commit'][:8]}, "
        f"programs {g['env']['config']['programs']})"
    )
    print("| algo | filter | sweep | bs | old ms | new ms | new/old [95 % CI] |")
    print("|---|---|---|---|---|---|---|")
    for a in g["abab"]:
        r = a["new_over_old"]
        print(
            f"| {a['algo']} | {a['filter']} | {a['sweep']} | {a['bs']} | {a['old_ms']:.3f} | {a['new_ms']:.3f} "
            f"| {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] |"
        )
    s = json.load(open(d / "profile_sweep.json"))["sweep"]
    keys = [k for k in s[0]["ms"] if k != "old"]
    print("\n## fused_masked_knn_topk kernel ms vs `programs` (V2 clause, 0.8 M d128)")
    print("| bs | sweep | old | " + " | ".join(keys) + " |")
    print("|---|---|---|" + "---|" * len(keys))
    for x in s:
        print(
            f"| {x['bs']} | {x['sweep']} | {x['ms']['old']:.3f} | "
            + " | ".join(f"{x['ms'][k]:.3f}" for k in keys)
            + " |"
        )


if __name__ == "__main__":
    old_side(Path(sys.argv[1]))
    fix_a(Path(sys.argv[2]))
