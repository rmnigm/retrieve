"""D3 PubMed summary (pod c's d3-arxiv/summary.py, pointed at pubmed): bloom FPR / recall / memory per width, and the
surprise check against d1/pubmed. Usage: python summary.py RESULTS_TREE [D1_FILTER_JSONL]"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

tree = Path(sys.argv[1])
recs = [
    json.loads(line)
    for p in tree.glob("*/*.jsonl")
    if not p.name.endswith(".samples.jsonl")
    for line in p.open()
]
print(
    f"{len(recs)} records, status {dict((s, sum(r['status'] == s for r in recs)) for s in {r['status'] for r in recs})}"
)

g = defaultdict(list)
for r in recs:
    b = r.get("bloom") or {}
    g[r["suite"], r["backend"], r["sweep"], b.get("m_bits"), b.get("k_hash")].append(r)


def q(r, k):
    return r["quality"]["oracle"].get(f"recall@{k}")


def ms(r, mode):
    return next(
        (
            p["median_ms"]
            for p in r.get("perf") or []
            if p["mode"] == mode and p["bs"] == 16 and p["k"] == 100
        ),
        None,
    )


print(
    "suite backend sweep m_bits k_hash n | fp_rate recall@100 recall@1000 index_mib | median_ms bs16 k100 eager graph"
)
for key in sorted(g, key=lambda k: tuple(str(x) for x in k)):
    rs = g[key]
    fps = [r.get("bloom_fp_rate") for r in rs]
    t = {
        m: [x for x in (ms(r, m) for r in rs) if x is not None]
        for m in ("eager", "graph")
    }
    print(
        *key,
        len(rs),
        "|",
        f"{median(fps) if None not in fps else fps[0]}",
        f"{median(q(r, 100) for r in rs):.4f}",
        f"{median(q(r, 1000) for r in rs):.4f}"
        if all(q(r, 1000) is not None for r in rs)
        else "-",
        f"{rs[0].get('index_mib'):.1f}",
        "|",
        *(f"{median(t[m]):.3f}" if t[m] else "-" for m in ("eager", "graph")),
    )

if len(sys.argv) > 2:
    print(
        "\nsurprise check: default bloom (m_bits 1024, k_hash 5) vs d1/arxiv filter, silvertorch bloom, per seed"
    )
    d1 = [json.loads(line) for line in open(sys.argv[2])]
    for r0 in d1:
        if (
            r0.get("algo") != "silvertorch"
            or r0.get("filter_kind") != "bloom"
            or r0["params"].get("n_probe") != 24
        ):
            continue
        mine = [
            r
            for r in recs
            if r["backend"] == r0["backend"]
            and r["sweep"] == r0["sweep"]
            and r["seed"] == r0["seed"]
            and (r["backend"] == "official" or r["bloom"] == r0["bloom"])
            and (r["bloom"] or {}).get("k_hash") == r0["bloom"]["k_hash"]
        ]
        for r in mine:
            print(
                r0["backend"],
                r0["sweep"],
                r0["seed"],
                r["suite"],
                f"fp {r0.get('bloom_fp_rate')} -> {r.get('bloom_fp_rate')}",
                f"r@100 {q(r0, 100):.4f} -> {q(r, 100):.4f}",
                f"idx {r0.get('index_mib'):.1f} -> {r.get('index_mib'):.1f}",
            )
