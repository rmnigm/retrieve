"""Chart data for the results story page (NOT CITABLE): pagedata.py OUT_DIR DATA_JSON, OUT_DIR = build_results.py output."""

import json
import pandas as pd
import sys

OUT = sys.argv[1].rstrip("/") + "/"
h = pd.read_csv(OUT + "h2h.csv")
m = pd.read_csv(OUT + "matched.csv")
D = {}
lab = {
    "goodreads": "Goodreads 0.8M",
    "arxiv": "arXiv 3M",
    "pubmed": "PubMed 10M",
    "yfcc10m": "YFCC 10M",
    "goodreads-synth": "goodreads-synth 0.8M",
    "arxiv-synth": "arxiv-synth 3M",
}
# 1: at n_probe 24, k100, clause: speedup and recall per real dataset
r = []
for ds in ["goodreads", "arxiv", "pubmed", "yfcc10m"]:
    x = h[(h.dataset == ds) & (h.n_probe == 24) & (h.k == 100) & (h.kind == "clause")]
    for sw, g in x.groupby("sweep"):
        r.append(
            dict(
                ds=lab[ds],
                sweep=sw,
                pass_rate=round(g.pass_rate.iloc[0], 4),
                recall=round(g.recall.iloc[0], 3),
                sp1=round(1 / g[g.bs == 1].st_over_linr.iloc[0], 1),
                sp16=round(1 / g[g.bs == 16].st_over_linr.iloc[0], 1),
            )
        )
D["prod"] = r
# 2: arxiv-synth & goodreads-synth latency vs pass rate
c = []
for ds in ["arxiv-synth", "goodreads-synth"]:
    for bs in [1, 16]:
        x = h[
            (h.dataset == ds) & (h.bs == bs) & (h.k == 100) & (h.kind == "clause")
        ].sort_values("pass_rate")
        base = x.drop_duplicates("sweep")
        series = [
            dict(
                name="LiNR V1",
                pts=[
                    [p, round(v, 4), None] for p, v in zip(base.pass_rate, base.V1_ms)
                ],
            ),
            dict(
                name="LiNR V2",
                pts=[
                    [p, round(v, 4), None] for p, v in zip(base.pass_rate, base.V2_ms)
                ],
            ),
        ]
        for npb, g in x.groupby("n_probe"):
            series.append(
                dict(
                    name=f"SilverTorch {int(npb)} probes",
                    pts=[
                        [p, round(v, 4), round(rc, 3)]
                        for p, v, rc in zip(g.pass_rate, g.ms, g.recall)
                    ],
                )
            )
        c.append(dict(ds=ds, label=lab[ds], bs=bs, series=series))
D["curves"] = c
# 3: matched map at 0.95 and 0.9
mm = []
for t in [0.9, 0.95]:
    for r_ in m[m.target == t].itertuples():
        mm.append(
            dict(
                t=t,
                ds=lab[r_.dataset],
                kind=r_.kind,
                sweep=r_.sweep,
                bs=int(r_.bs),
                k=int(r_.k),
                pass_rate=round(r_.pass_rate, 4),
                win="ST" if r_.winner == "SilverTorch" else "LiNR",
                ratio=None if pd.isna(r_.st_over_linr) else round(r_.st_over_linr, 2),
                np=None if pd.isna(r_.n_probe) else int(r_.n_probe),
                maxrec=round(r_.max_recall, 3),
                linr=r_.linr_best,
            )
        )
D["matched"] = mm
# 4: C1 curves
cc = pd.read_csv(OUT + "c1_curves.csv")
cx = pd.read_csv(OUT + "c1_crossover.csv")
pick = [("goodreads-synth", 128), ("laion3m-synth", 256), ("laion30m-synth", 256)]
D["c1"] = [
    dict(
        name=f"{d} d{dim}",
        n=int(g.n_items.iloc[0]),
        bs=int(bs),
        pts=[[round(p, 4), round(v, 3)] for p, v in zip(g.pass_rate, g.v2_over_v1)],
    )
    for (d, dim) in pick
    for bs, g in cc[(cc.dataset == d) & (cc.dim == dim)]
    .sort_values("pass_rate")
    .groupby("bs")
]
D["c1x"] = json.loads(cx.round(3).to_json(orient="records"))
# 5: C5 Meta by N
c5 = pd.read_csv(OUT + "c5_meta.csv")
c5 = c5[c5.suite.isin(["nsweep-codesign"])]
D["c5"] = [
    dict(
        bs=int(bs),
        pts=[
            [
                int(n),
                round(g2.full_over_partial.median(), 2),
                round(g2.full_over_partial.min(), 2),
                round(g2.full_over_partial.max(), 2),
            ]
            for n, g2 in g.groupby("n_items")
        ],
    )
    for bs, g in c5.groupby("perf_bs")
]
a = pd.read_csv(OUT + "c5_adapted.csv")
D["c5a"] = [
    dict(
        ds=f"{d} d{dim}",
        bs=int(bs),
        med=round(g.graph_full_over_partial.median(), 2),
        mx=round(g.graph_full_over_partial.max(), 2),
    )
    for (d, dim, bs), g in a.groupby(["dataset", "dim", "bs"])
]
ad = pd.read_csv(OUT + "adaptation.csv")
D["adapt"] = json.loads(ad.round(3).to_json(orient="records"))
json.dump(D, open(sys.argv[2], "w"))
print(len(json.dumps(D)))
for r in D["prod"]:
    print(r)
print(D["c5"])
