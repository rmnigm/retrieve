"""Chart data for the results story page: pagedata.py OUT_DIR DATA_JSON, OUT_DIR = build_results.py output."""

import json
import sys

import pandas as pd

OUT = sys.argv[1].rstrip("/") + "/"
h = pd.read_csv(OUT + "h2h.csv")
tn = pd.read_csv(OUT + "tuned.csv")
REAL = [
    "goodreads",
    "laion1m",
    "arxiv",
    "laion3m",
    "pubmed",
    "yfcc10m",
    "laion10m",
    "laion30m",
]
SYN = [
    "goodreads-synth",
    "laion1m-synth",
    "arxiv-synth",
    "laion3m-synth",
    "yfcc10m-synth",
    "laion10m-synth",
    "laion30m-synth",
]
lab = {
    "goodreads": "Goodreads 0.8M",
    "laion1m": "LAION 1M",
    "arxiv": "arXiv 3M",
    "laion3m": "LAION 3M",
    "pubmed": "PubMed 10M",
    "yfcc10m": "YFCC 10M",
    "laion10m": "LAION 10M",
    "laion30m": "LAION 30M",
}
lab.update({s: lab[s.replace("-synth", "")].replace(" ", "-synth ", 1) for s in SYN})
lab["arxiv-corr-synth"] = "arXiv-corr-synth 3M"
D = {}


def num(v, nd=3):
    return None if pd.isna(v) else round(float(v), nd)


# 1: the tuned point per dataset size (real filters, clause, k 100), targets 0.90 / 0.95
x = tn[
    (tn.filter_kind == "clause")
    & (tn.perf_k == 100)
    & tn.dataset.isin(REAL)
    & (tn.target < 0.99)
]
size = []
for (t, ds, sw), g in x.groupby(["target", "dataset", "sweep"]):
    g = g.set_index("perf_bs")
    b = g.dropna(subset=["n_probe"])
    size.append(
        dict(
            t=t,
            ds=lab[ds],
            n=int(g.n_items.iloc[0]),
            sweep=sw,
            pass_rate=num(g.pass_rate.iloc[0], 4),
            top=num(g.max_recall.max()),
            cfg=None
            if b.empty
            else f"{int(b.n_lists.iloc[0])} lists / {int(b.n_probe.iloc[0])} probes",
            recall=None if b.empty else num(b.recall.iloc[0]),
            sp={
                int(bs): num(1 / g.st_over_linr[bs], 2)
                if not pd.isna(g.st_over_linr[bs])
                else None
                for bs in g.index
            },
            linr=g.linr_best.iloc[0],
        )
    )
D["size"] = sorted(size, key=lambda r: (r["t"], r["n"], r["ds"], r["sweep"]))
# 2: at 24 probes, smallest n_lists, clause, k 100
a = h[
    (h.n_probe == 24)
    & (h.filter_kind == "clause")
    & (h.perf_k == 100)
    & h.dataset.isin(REAL)
]
a = a[a.n_lists == a.groupby("dataset").n_lists.transform("min")]
D["prod"] = [
    dict(
        ds=lab[ds],
        n=int(g.n_items.iloc[0]),
        sweep=sw,
        pass_rate=num(g.pass_rate.iloc[0], 4),
        recall=num(g.recall.iloc[0]),
        sp1=num(1 / g[g.perf_bs == 1].st_over_linr.iloc[0], 1),
        sp16=num(1 / g[g.perf_bs == 16].st_over_linr.iloc[0], 1),
    )
    for (n, ds, sw), g in a.groupby(["n_items", "dataset", "sweep"])
]
# 3: latency against pass rate, synthetic sets, smallest n_lists
c = []
for ds in ["goodreads-synth", "arxiv-synth", "laion10m-synth", "laion30m-synth"]:
    for bs in [1, 16, 64]:
        x = h[
            (h.dataset == ds)
            & (h.perf_bs == bs)
            & (h.perf_k == 100)
            & (h.filter_kind == "clause")
        ]
        base = x.drop_duplicates("sweep").sort_values("pass_rate")
        series = [
            dict(
                name="LiNR V1",
                pts=[[p, num(v, 4), None] for p, v in zip(base.pass_rate, base.V1)],
            ),
            dict(
                name="LiNR V2",
                pts=[[p, num(v, 4), None] for p, v in zip(base.pass_rate, base.V2)],
            ),
        ]
        nl = x.n_lists.min()
        g = x[(x.n_lists == nl) & (x.n_probe == 24)].sort_values("pass_rate")
        series.append(
            dict(
                name="SilverTorch, 24 probes",
                pts=[
                    [p, num(v, 4), num(r)]
                    for p, v, r in zip(g.pass_rate, g.st_ms, g.recall)
                ],
            )
        )
        for t in [0.9, 0.95]:
            g = tn[
                (tn.dataset == ds)
                & (tn.perf_bs == bs)
                & (tn.perf_k == 100)
                & (tn.filter_kind == "clause")
            ]
            g = g[(g.target == t) & g.st_ms.notna()].sort_values("pass_rate")
            series.append(
                dict(
                    name=f"SilverTorch, tuned for {t:.2f}",
                    pts=[
                        [p, num(v, 4), num(r)]
                        for p, v, r in zip(g.pass_rate, g.st_ms, g.recall)
                    ],
                )
            )
        c.append(dict(ds=ds, label=lab[ds], bs=bs, series=series))
D["curves"] = c
# 4: matched map, every dataset and filter kind, bs 1 / 16 / 64 x k 100 / 1000
mm = []
for r in tn[tn.target < 0.99].itertuples():
    mm.append(
        dict(
            t=r.target,
            ds=lab[r.dataset],
            n=int(r.n_items),
            kind=r.filter_kind,
            sweep=r.sweep,
            bs=int(r.perf_bs),
            k=int(r.perf_k),
            pass_rate=num(r.pass_rate, 4),
            win="ST" if r.winner == "SilverTorch" else "LiNR",
            ratio=num(r.st_over_linr, 2),
            np=None if pd.isna(r.n_probe) else int(r.n_probe),
            nl=None if pd.isna(r.n_lists) else int(r.n_lists),
            maxrec=num(r.max_recall),
            linr=r.linr_best,
            flag=bool(r.flagged) if not pd.isna(r.flagged) else False,
        )
    )
D["matched"] = [m for m in mm if m["k"] in (100, 1000)]
# 5: LiNR V2 / V1 (final pass)
cc = pd.read_csv(OUT + "c1_curves.csv")
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
# 6: co-design, Meta's code as published, by N
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
json.dump(D, open(sys.argv[2], "w"))
print(len(json.dumps(D)), {k: len(v) for k, v in D.items()})
