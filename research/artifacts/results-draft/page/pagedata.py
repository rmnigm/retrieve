"""Chart data for the results story page: pagedata.py OUT_DIR DATA_JSON, OUT_DIR = build_results.py output (FULL-RERUN)."""

import json
import sys

import pandas as pd

OUT = sys.argv[1].rstrip("/") + "/"
h = pd.read_csv(OUT + "h2h.csv")
tn = pd.read_csv(OUT + "tuned.csv")
top = pd.read_csv(OUT + "c6_top.csv")
REAL = ["goodreads", "laion1m", "arxiv", "laion3m", "pubmed", "laion10m", "laion30m"]
SYN = [
    "goodreads-synth",
    "arxiv-synth",
    "pubmed-synth",
    "laion10m-synth",
    "laion30m-synth",
]
lab = {
    "goodreads": "Goodreads 0.8M",
    "laion1m": "LAION 1M",
    "arxiv": "arXiv 3M",
    "laion3m": "LAION 3M",
    "pubmed": "PubMed 10M",
    "laion10m": "LAION 10M",
    "laion30m": "LAION 30M",
}
lab.update({s: lab[s.replace("-synth", "")].replace(" ", "-synth ", 1) for s in SYN})
lab["arxiv-corr-synth"] = "arXiv-corr-synth 3M"
D = {}


def num(v, nd=3):
    return None if pd.isna(v) else round(float(v), nd)


def rng(s, nd=2):
    s = s.dropna()
    lo, hi = s.min(), s.max()
    return (
        f"{lo:.{nd}f}" if round(lo, nd) == round(hi, nd) else f"{lo:.{nd}f}–{hi:.{nd}f}"
    )


def curve_top(ds, sweep, k):
    t = top[(top.dataset == ds) & (top.sweep == sweep) & (top.filter_kind == "bloom")]
    col = "top_recall100"
    if k != 100:
        c = pd.read_csv(OUT + "c6_curves.csv")
        c = c[(c.dataset == ds) & (c.sweep == sweep) & (c.filter_kind == "bloom")]
        return num(c[f"oracle_recall@{min(k, 1000)}"].max()) if len(c) else None
    return num(t[col].max()) if len(t) else None


s0 = tn[tn.seed == 0]
h0 = h[h.seed == 0]
# 1: the tuned point per dataset size (real filters, k 100), targets 0.90 / 0.95, against the best LiNR at that recall
x = s0[(s0.perf_k == 100) & s0.dataset.isin(REAL)]
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
            top=curve_top(ds, sw, 100),
            cfg=None
            if b.empty
            else f"{int(b.n_lists.iloc[0])} lists / {int(b.n_probe.iloc[0])} probes",
            recall=None if b.empty else num(b.recall.iloc[0]),
            sp={
                int(bs): None
                if pd.isna(g.st_over_matched[bs])
                else num(1 / g.st_over_matched[bs], 2)
                for bs in g.index
            },
            v3={int(bs): bool(g.matched_best[bs] == "V3") for bs in g.index},
            linr=g.matched_best.iloc[0],
        )
    )
D["size"] = sorted(size, key=lambda r: (r["t"], r["n"], r["ds"], r["sweep"]))
# 2: 24 probes at the smallest n_lists, k 100, real filters
a = h0[(h0.n_probe == 24) & (h0.perf_k == 100) & h0.dataset.isin(REAL)]
a = a[a.n_lists == a.groupby("dataset").n_lists.transform("min")]
D["prod"] = [
    dict(
        ds=lab[ds],
        n=int(n),
        sweep=sw,
        pass_rate=num(g.pass_rate.iloc[0], 4),
        recall=num(g.recall.iloc[0]),
        sp1=num(1 / g[g.perf_bs == 1].st_over_linr.iloc[0], 1),
        sp16=num(1 / g[g.perf_bs == 16].st_over_linr.iloc[0], 1),
        sp64=num(1 / g[g.perf_bs == 64].st_over_linr.iloc[0], 1),
    )
    for (n, ds, sw), g in a.groupby(["n_items", "dataset", "sweep"])
]
# 3: latency against pass rate on the synthetic sets with a timed bloom leg
c = []
for ds in [d for d in SYN if d in set(h0.dataset)]:
    for bs in [1, 16, 64]:
        x = h0[(h0.dataset == ds) & (h0.perf_bs == bs) & (h0.perf_k == 100)]
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
        g = x[(x.n_lists == x.n_lists.min()) & (x.n_probe == 24)].sort_values(
            "pass_rate"
        )
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
            g = s0[
                (s0.dataset == ds)
                & (s0.perf_bs == bs)
                & (s0.perf_k == 100)
                & (s0.target == t)
                & s0.st_ms.notna()
            ].sort_values("pass_rate")
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
D["synth_pending"] = [lab[d] for d in SYN if d not in set(h0.dataset)]
# 4: every cell, against the best LiNR at that recall
D["matched"] = [
    dict(
        t=r.target,
        ds=lab[r.dataset],
        n=int(r.n_items),
        sweep=r.sweep,
        bs=int(r.perf_bs),
        k=int(r.perf_k),
        pass_rate=num(r.pass_rate, 4),
        win="ST" if r.winner_matched == "SilverTorch" else "LiNR",
        ratio=num(r.st_over_matched, 2),
        np=None if pd.isna(r.n_probe) else int(r.n_probe),
        nl=None if pd.isna(r.n_lists) else int(r.n_lists),
        maxrec=curve_top(r.dataset, r.sweep, int(r.perf_k)),
        linr=r.matched_best,
        flag=bool(r.flagged) if not pd.isna(r.flagged) else False,
    )
    for r in s0.itertuples()
]
reached = s0[s0.st_ms.notna()]
D["counts"] = dict(
    cells=int(len(s0)),
    reached=int(len(reached)),
    st_wins=int((reached.winner_matched == "SilverTorch").sum()),
    linr_wins=[
        f"{lab[r.dataset]} {r.sweep}, k {int(r.perf_k)}, bs {int(r.perf_bs)}, target {r.target:.2f}: {r.st_over_matched:.2f}× LiNR {r.matched_best}"
        for r in reached[reached.winner_matched != "SilverTorch"].itertuples()
    ],
    v3_best=int((reached.matched_best == "V3").sum()),
    ratio_range=rng(reached.st_over_matched),
)
# 5: seed variance of the tuned head-to-head
v = pd.read_csv(OUT + "variance_tuned.csv")
v = v[v.seeds > 1]
D["variance"] = [
    dict(
        ds=lab[r.dataset],
        sweep=r.sweep,
        bs=int(r.perf_bs),
        k=int(r.perf_k),
        t=r.target,
        seeds=int(r.seeds),
        reached=int(r.reached),
        med=num(r.matched_med, 2),
        lo=num(r.matched_min, 2),
        hi=num(r.matched_max, 2),
    )
    for r in v.itertuples()
]
# 4b: bloom against SilverTorch's exact clause filter (clause reference row), graph, seed 0
cb = pd.read_csv(OUT + "clause_vs_bloom.csv")
cb = cb[(cb.perf_mode == "graph") & (cb.seed == 0)]
D["clause_bloom"] = [
    dict(
        ds=lab.get(ds, ds),
        sweep=sw,
        bs=int(bs),
        n=len(g),
        med=num(g.bloom_over_clause.median(), 2),
        lo=num(g.bloom_over_clause.min(), 2),
        hi=num(g.bloom_over_clause.max(), 2),
        cost=num(g.recall_cost.median(), 3),
    )
    for (ds, sw, bs), g in cb.groupby(["dataset", "sweep", "perf_bs"])
]
# 6: LiNR V2 / V1
cc = pd.read_csv(OUT + "c1_curves.csv")
pick = [
    d
    for d in ["goodreads-synth", "arxiv-synth", "laion30m-synth", "laion10m-synth"]
    if d in set(cc.dataset)
][:3]
D["c1"] = [
    dict(
        name=f"{d} d{dim}",
        n=int(g.n_items.iloc[0]),
        bs=int(bs),
        pts=[[round(p, 4), round(v, 3)] for p, v in zip(g.pass_rate, g.v2_over_v1)],
    )
    for d in pick
    for dim in sorted(set(cc[cc.dataset == d].dim))
    for bs, g in cc[(cc.dataset == d) & (cc.dim == dim)]
    .sort_values("pass_rate")
    .groupby("bs")
]
# 7: co-design, Meta's code as published, by N
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
# 8: claims, numbers from the tables
c1t = pd.read_csv(OUT + "c1_crossover.csv")
lr = pd.read_csv(OUT + "linr_real.csv")
lr = lr[lr.perf_k == 100]
c2 = pd.read_csv(OUT + "c2_v3.csv")
c2 = c2.loc[c2.groupby(["leg", "dataset", "sweep", "perf_bs"]).pool.idxmin()]
c3 = pd.read_csv(OUT + "c3_torch.csv").dropna(subset=["torch_over_triton"])
bw = pd.read_csv(OUT + "c4_bloomwidth.csv")
ctop = top[(top.filter_kind == "bloom") & top.dataset.isin(REAL)]
ctop_clause = top[(top.filter_kind == "clause") & top.dataset.isin(REAL)]
ym = pd.read_csv(OUT + "yambda.csv")
ym = ym[ym.algo == "silvertorch"]
ym95 = ym[ym["oracle_recall@100"] >= 0.95].groupby("n_lists").n_probe.min()
reach95 = sorted(
    {
        lab[r.dataset] + " " + r.sweep
        for r in ctop_clause.itertuples()
        if r.top_recall100 >= 0.95
    }
)
miss95 = sorted(
    {
        lab[r.dataset] + " " + r.sweep
        for r in ctop_clause.itertuples()
        if r.top_recall100 < 0.95
    }
)
claims = []
b1 = c1t[c1t.bs == 1].crossover_p
b16 = c1t[c1t.bs == 16].crossover_p
claims.append(
    [
        "C1",
        "part",
        "shape, not location",
        f"<b>LiNR: V2 beats V1 at low pass rates.</b> For one query at a time the crossover sits at a pass rate of {rng(b1)}, not LiNR's 0.11, so V2 wins at LiNR's operating point. "
        f"At bs 16 V2 wins only below {rng(b16, 3)} passing; at bs 64 and 256 V1 wins on every filter, synthetic and real. On the real filters at bs 1, V2 takes {rng(lr[(lr.perf_bs == 1) & (lr.pass_rate < 0.99)].v2_over_v1)} of V1's time.",
    ]
)
if len(c2):
    r1 = c2[c2.perf_bs == 1].v3_over_v2
    r16 = c2[c2.perf_bs > 1].v3_over_v2
    claims.append(
        [
            "C2",
            "part",
            "not comparable",
            f"<b>LiNR V3 (1-bit, keep 1 %): about 10 % faster at near-equal recall.</b> LiNR's recall is end-task recall with an unpublished protocol. Under ours, the smallest pool reaching 0.90 (0.5–20 % of the items) "
            f"takes {rng(r1)} of V2's time at one query and {rng(r16)} at bs 16 and 64, where V1 is faster than both.",
        ]
    )
if len(c3):
    e1 = c3[(c3.perf_bs == 1) & (c3.perf_mode == "eager")].torch_over_triton
    e16 = c3[(c3.perf_bs == 16) & (c3.perf_mode == "eager")].torch_over_triton
    claims.append(
        [
            "C3",
            "part",
            "direction only",
            f"<b>LiNR: native masking is about 100× slower than a custom kernel.</b> Against torch.compile (max-autotune, timed eagerly) the custom kernel is {rng(e1)}× faster at one query and {rng(e16)}× at bs 16, on PubMed and LAION-synth 30M so far. The rows against plain PyTorch and on Goodreads and arXiv are still running.",
        ]
    )
pm = bw[bw.dataset == "pubmed"].groupby("m_bits").recall.max()
claims.append(
    [
        "C4",
        "pend",
        "pending",
        f"<b>SilverTorch bloom filter: about 0.067 % false positives at 1024 bits.</b> The false-positive count on SilverTorch's own bloom is still running. At a fixed probe setting, recall stops improving at 256–512 bits: "
        f"PubMed's rarest filter reaches {pm.max():.2f} there but only {pm.min():.2f} at 64 bits; arXiv and Goodreads are flat from 64 bits.",
    ]
)
claims.append(
    [
        "C5",
        "pend",
        "pending",
        "<b>SilverTorch co-design: 1.79–2.15× faster.</b> The rerun of Meta's code as published across 1–30 M items is still running.",
    ]
)
claims.append(
    [
        "C6",
        "part",
        "partially",
        f"<b>SilverTorch int8 cannot reach 0.95 recall; selective filters need more probes.</b> It does reach 0.95 within a quarter of the clusters on {len(reach95)} of {len(reach95) + len(miss95)} real filters "
        f"(Goodreads, arXiv, LAION's tag filter at 10–30 M, PubMed's rarest filter). It does not on LAION's domain filter at any size (top {ctop_clause[ctop_clause.sweep == 'c0_domain'].top_recall100.max():.2f}), "
        f"LAION's tag filter at 1–3 M, or PubMed's two broad filters (top {ctop_clause[ctop_clause.sweep.isin(['all5', 'c3_journal_reverse'])].top_recall100.max():.2f})."
        + (
            f" On Yambda, with no filter, it reaches 0.95 at {int(ym95.min())}–{int(ym95.max())} probes for 1,024–16,384 clusters."
            if len(ym95)
            else ""
        ),
    ]
)
try:
    gl = pd.read_csv(OUT + "gls_adapted.csv")
    claims[-1][3] += (
        f" On Goodreads a query's recall follows its local pass rate (how much of its unfiltered top-100 passes; Spearman up to {gl.spearman_lq.max():.2f}) "
        f"more than the global pass rate (up to {gl.spearman_pq.max():.2f})."
    )
except FileNotFoundError:
    pass
claims.append(
    [
        "C7",
        "ok",
        "holds",
        "<b>Our adaptation keeps Meta's results.</b> Identical ids and scores to Meta's code as published in all 1,536 gate cells (an earlier gate, not part of this rerun).",
    ]
)
D["claims"] = claims
json.dump(D, open(sys.argv[2], "w"))
print(len(json.dumps(D)), {k: len(v) for k, v in D.items()})
print(json.dumps(D["counts"], indent=1))
print("\n".join(c[3] for c in claims))
