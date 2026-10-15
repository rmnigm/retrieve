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


TG = [0.9, 0.95]


def pct_(p):
    return f"{100 * p:.1f} %" if p < 0.1 else f"{100 * p:.0f} %"


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
            half=bool((b.probe_frac > 0.25).any())
            if "probe_frac" in b and len(b)
            else False,
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
        ex=num(r.st_over_linr, 2),
        half=bool(getattr(r, "probe_frac", 0) > 0.25),
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
    quarter=int((reached.probe_frac <= 0.25).sum()),
    quarter_wins=int(
        ((reached.probe_frac <= 0.25) & (reached.winner_matched == "SilverTorch")).sum()
    ),
    beyond=int((reached.probe_frac > 0.25).sum()),
    beyond_wins=int(
        ((reached.probe_frac > 0.25) & (reached.winner_matched == "SilverTorch")).sum()
    ),
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
        nl=int(nl),
        bs=int(bs),
        n=len(g),
        med=num(g.bloom_over_clause.median(), 2),
        lo=num(g.bloom_over_clause.min(), 2),
        hi=num(g.bloom_over_clause.max(), 2),
        cost=num(g.recall_cost.median(), 3),
        mhz_b=num(g.perf_sm_mhz_bloom.median(), 0),
        mhz_c=num(g.perf_sm_mhz_clause.median(), 0),
    )
    for (ds, sw, nl, bs), g in cb.groupby(["dataset", "sweep", "n_lists", "perf_bs"])
]
fpp = pd.read_csv(OUT + "c4_fp_points.csv")
D["fp"] = dict(
    points=int(len(fpp)),
    zero=int((fpp.rejects_over_survivors == 0).sum()),
    share_max=num(100 * fpp.rejects_over_survivors.max(), 3),
    fail_max=float(f"{fpp.rejects_over_exact_failing.max():.1g}"),
    fn=int(fpp.false_negative_rows.sum()),
    cost_max=f"{cb.recall_cost.abs().max():.6f}",
    graph_n=int(len(cb)),
)
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
    e = c3[c3.perf_mode == "eager"]
    pt = e[e.variant == "eager torch"]
    ma = e[e.variant == "max-autotune"]
    claims.append(
        [
            "C3",
            "no",
            "does not hold as stated",
            f"<b>LiNR: native masking is about 100× slower than a custom kernel.</b> Same direction, far smaller. Against plain PyTorch the custom kernel is {rng(pt[pt.perf_bs == 1].torch_over_triton, 1)}× faster at one query and {rng(pt[pt.perf_bs == 16].torch_over_triton, 1)}× at bs 16 (Goodreads-synth, timed eagerly). "
            f"Against torch.compile (max-autotune) it is {rng(ma[ma.perf_bs == 1].torch_over_triton)}× at one query, so compiled PyTorch is sometimes faster, and {rng(ma[ma.perf_bs == 16].torch_over_triton)}× at bs 16, across Goodreads, arXiv, PubMed and the synthetic sets.",
        ]
    )
pm = bw[bw.dataset == "pubmed"].groupby("m_bits").recall.max()
fw = pd.read_csv(OUT + "c4_fp_width.csv")
f1024 = fw[fw.m_bits == 1024]
claims.append(
    [
        "C4",
        "ok",
        "holds on Meta's bloom as adapted",
        f"<b>SilverTorch bloom filter: about 0.067 % false positives at 1024 bits.</b> At Meta's default of 1024 bits and 5 hashes, SilverTorch's own bloom lets through at most {100 * f1024[f1024.k_hash == 5].fp_rate_mean.max():.6f} % of the items that fail the filter ({100 * f1024.fp_rate_mean.max():.4f} % with 3 hashes; PubMed's MeSH filter), and none on Goodreads and arXiv. "
        f"At the timed settings, at most {D['fp']['share_max']} % of the items it passes are false, and recall moves by at most {D['fp']['cost_max']} against the exact filter. Narrower filters let more through: {100 * fw[(fw.dataset == 'pubmed') & (fw.m_bits == 64)].fp_rate_mean.max():.1f} % on PubMed at 64 bits, where recall at a fixed probe setting falls to {pm.min():.2f} from {pm.max():.2f}.",
    ]
)
c5n_all = pd.read_csv(OUT + "c5_meta.csv")
c5n_all = c5n_all[c5n_all.suite == "nsweep-codesign"]
c5n = c5n_all.groupby("n_items").full_over_partial
if len(c5n.size()):
    claims.append(
        [
            "C5",
            "part",
            "wins at 10–30 M only",
            f"<b>SilverTorch co-design: 1.79–2.15× faster.</b> On Meta's code as published, filtering inside the probed clusters is {c5n.median()[30_000_000]:.2f}× faster (median) than filtering the whole index first at 30 M items, "
            f"{c5n.median()[10_000_000]:.2f}× at 10 M, and slower below that ({c5n.median()[1_000_000]:.2f}× at 1 M).",
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
ce = pd.read_csv(OUT + "c6_ceiling.csv")
if len(ce):
    ce = ce.assign(
        lift=ce.fp16 - ce.shipped,
        scale=ce[["global", "per_row"]].max(axis=1) - ce.shipped,
    )
    ce = ce[ce.dataset.isin(REAL)]
    big = ce[ce.lift >= 0.02]
    small = ce[ce.lift < 0.02]
    names = lambda t: sorted(  # noqa: E731
        {
            lab[d] if d not in set(big.dataset) else f"{lab[d]} {w}"
            for d, w in zip(t.dataset, t.sweep)
        }
    )
    claims[-1][3] += (
        f" At a quarter of the clusters, fp16 scores instead of int8 would lift {', '.join(f'{lab[d]} {w}' for d, w in zip(big.dataset, big.sweep))} by {rng(big.lift)}; "
        f"elsewhere ({', '.join(names(small))}) by under 0.02, so the clusters probed, not int8, set the limit. A finer int8 scale (per row or global) adds at most {ce.scale.max():.3f}."
    )
claims.append(
    [
        "C7",
        "ok",
        "holds",
        "<b>Our adaptation keeps Meta's results.</b> Identical ids and scores to Meta's code as published in all 1,536 gate cells (earlier gate at the adapted code's review head; C7 was not rerun in the final rerun, by the user's call).",
    ]
)
D["claims"] = claims
med5 = c5n.median()
D["verdicts"] = [
    [
        "C1",
        "part",
        "shape holds, location does not",
        "LiNR V2 (gather, then score) beats V1 (score, then mask) below about 11 % passing.",
        f"For one query V2 wins below {rng(b1)} passing; at a batch of 16 only below {rng(b16, 3)}; at 64 and 256 never.",
    ],
    [
        "C2",
        "part",
        "not comparable",
        "LiNR V3's 1-bit prefilter is about 10 % faster at near-equal recall.",
        f"At the smallest pool reaching 0.90, V3 takes {rng(r1)} of V2's time for one query and {rng(r16)} at batches of 16 and 64, where V1 beats both.",
    ],
    [
        "C3",
        "no",
        "does not hold as stated",
        "Native PyTorch masking is about 100× slower than a custom kernel.",
        f"{rng(pt[pt.perf_bs == 1].torch_over_triton, 1)}× at one query and {rng(pt[pt.perf_bs == 16].torch_over_triton, 1)}× at a batch of 16 against plain PyTorch; about even with torch.compile at one query.",
    ],
    [
        "C4",
        "ok",
        "holds on Meta's bloom as adapted",
        "SilverTorch's 1024-bit bloom filter lets through about 0.067 % false positives.",
        f"At most {100 * f1024[f1024.k_hash == 5].fp_rate_mean.max():.6f} % of failing items pass; recall moves by at most {D['fp']['cost_max']} against the exact filter.",
    ],
    [
        "C5",
        "part",
        "wins at 10–30 M only",
        "Filtering inside the probed clusters is 1.79–2.15× faster than filtering the whole index.",
        f"{med5[30_000_000]:.2f}× at 30 M, {med5[10_000_000]:.2f}× at 10 M, {med5[1_000_000]:.2f}× at 1 M (median, Meta's code as published).",
    ],
    [
        "C6",
        "part",
        "partially",
        "int8 SilverTorch cannot reach 0.95 recall; selective filters need more probes.",
        f"0.95 is reached within a quarter of the clusters on {len(reach95)} of {len(reach95) + len(miss95)} real filters; the clusters probed, not int8, set the limit.",
    ],
    [
        "C7",
        "ok",
        "holds (earlier gate)",
        "Our adaptation returns what Meta's published code returns.",
        "Identical ids and scores in all 1,536 gate cells, at the adapted code's review head; not rerun in the final rerun.",
    ],
]
pr = json.load(open(OUT + "product_recall.json"))
allf = [
    v["pearson"] for k, v in pr.items() if k.startswith("all k") and "pooled" not in k
]
D["bridge"] = dict(
    bloom100=num(pr["bloom k100 pooled"]["pearson"], 2),
    bloom1000=num(pr["bloom k1000 pooled"]["pearson"], 2),
    n=pr["bloom k100 pooled"]["n"],
    all_lo=num(min(allf), 2),
    all_hi=num(max(allf), 2),
)
# SilverTorch's own settings: recall against the share of clusters probed (curves), speed against recall (timed points)
cv = pd.read_csv(OUT + "c6_curves.csv")
cv = cv[cv.filter_kind == "bloom"]
mem = pd.read_csv(OUT + "memory.csv")
mem = mem[mem.arm == "SilverTorch"]
knobs = []
for (ds, sw), g in cv.groupby(["dataset", "sweep"]):
    if ds not in lab:
        continue
    t = h0[(h0.dataset == ds) & (h0.sweep == sw) & (h0.perf_k == 100)]
    knobs.append(
        dict(
            key=f"{ds}|{sw}",
            label=f"{lab[ds]} · {sw}",
            n=int(g.n_items.iloc[0]),
            pass_rate=num(g.pass_rate.iloc[0], 4),
            curves={
                int(nl): [
                    [num(r.n_probe / nl, 5), num(r["oracle_recall@100"])]
                    for _, r in c.sort_values("n_probe").iterrows()
                ]
                for nl, c in g.groupby("n_lists")
            },
            timed={
                int(bs): [
                    [int(r.n_lists), int(r.n_probe), num(r.st_ms, 4), num(r.recall)]
                    for r in tb_.itertuples()
                ]
                for bs, tb_ in t.groupby("perf_bs")
            },
            linr={
                int(bs): [num(tb_.linr_ms.iloc[0], 4), tb_.linr_best.iloc[0]]
                for bs, tb_ in t.groupby("perf_bs")
            },
            mem={
                int(r.n_lists): [num(r.index_mib, 0), num(r.build_s, 1)]
                for r in mem[mem.dataset == ds].itertuples()
            },
        )
    )
D["knobs"] = sorted(knobs, key=lambda r: (r["n"], r["key"]))

# Thesis-style tables: one per question, algorithms as rows
rp = pd.read_parquet(OUT + "results.parquet")
exact = (
    rp[
        rp.leg.str.contains("timed-bloom")
        & (rp.seed == 0)
        & rp.algo.isin(["linr_v1_filter_mask", "linr_v2"])
        & (rp.perf_k == 100)
    ]
    .groupby(["dataset", "sweep", "algo"])["oracle_recall@100"]
    .first()
)
memall = pd.read_csv(OUT + "memory.csv")


def mib(ds, arm, nl=None):
    m = memall[(memall.dataset == ds) & (memall.arm == arm)]
    if nl is not None:
        m = m[m.n_lists == nl]
    return num(m.index_mib.iloc[0], 0) if len(m) else None


tab = []
k100 = h0[(h0.perf_k == 100) & h0.dataset.isin(REAL)]
for (ds, sw, bs), g in k100.groupby(["dataset", "sweep", "perf_bs"]):
    v1, v2 = g.V1.iloc[0], g.V2.iloc[0]
    tt = s0[
        (s0.dataset == ds) & (s0.sweep == sw) & (s0.perf_bs == bs) & (s0.perf_k == 100)
    ].set_index("target")
    a24 = g[(g.n_probe == 24) & (g.n_lists == g.n_lists.min())]
    rows = [
        [
            "LiNR V1",
            num(exact.get((ds, sw, "linr_v1_filter_mask"))),
            num(v1, 3),
            "exact",
            mib(ds, "V1"),
        ],
        [
            "LiNR V2",
            num(exact.get((ds, sw, "linr_v2"))),
            num(v2, 3),
            "exact",
            mib(ds, "V2"),
        ],
    ]
    for t in TG:
        if t in tt.index and not pd.isna(tt.v3_ms[t]):
            rows.append(
                [
                    f"LiNR V3, pool for {t:.2f}",
                    num(tt.v3_recall[t]),
                    num(tt.v3_ms[t], 3),
                    f"pool {100 * tt.v3_pool[t]:g} %",
                    mib(ds, "V3"),
                ]
            )
    if len(a24):
        a = a24.iloc[0]
        rows.append(
            [
                "SilverTorch, 24 probes",
                num(a.recall),
                num(a.st_ms, 3),
                f"{int(a.n_lists)} lists / 24",
                mib(ds, "SilverTorch", a.n_lists),
            ]
        )
    for t in TG:
        if t in tt.index and not pd.isna(tt.st_ms[t]):
            r = tt.loc[t]
            rows.append(
                [
                    f"SilverTorch, tuned for {t:.2f}",
                    num(r.recall),
                    num(r.st_ms, 3),
                    f"{int(r.n_lists)} / {int(r.n_probe)}"
                    + (" ◐" if r.probe_frac > 0.25 else ""),
                    mib(ds, "SilverTorch", r.n_lists),
                ]
            )
        elif t in tt.index:
            rows.append(
                [
                    f"SilverTorch, tuned for {t:.2f}",
                    None,
                    None,
                    f"not reached (top {num(tt.max_recall[t])})",
                    None,
                ]
            )
    for r in rows:
        r.insert(3, None if r[2] is None else num(v1 / r[2], 2))
    tab.append(
        dict(
            ds=lab[ds],
            key=f"{ds}|{sw}",
            n=int(g.n_items.iloc[0]),
            sweep=sw,
            bs=int(bs),
            pass_rate=num(g.pass_rate.iloc[0], 4),
            dim=int(g.dim.iloc[0]),
            rows=rows,
        )
    )
D["tab"] = sorted(tab, key=lambda r: (r["n"], r["key"], r["bs"]))
# matched-recall summary: one row per real filter, SilverTorch at 0.90 against the best LiNR at that recall
msum = []
for (ds, sw), g in s0[(s0.perf_k == 100) & s0.dataset.isin(REAL)].groupby(
    ["dataset", "sweep"]
):
    row = dict(
        ds=lab[ds],
        sweep=sw,
        n=int(g.n_items.iloc[0]),
        pass_rate=num(g.pass_rate.iloc[0], 4),
    )
    for t in TG:
        x = g[g.target == t].set_index("perf_bs")
        b = x.dropna(subset=["n_probe"])
        row[f"frac{t}"] = None if b.empty else num(b.probe_frac.iloc[0], 3)
        row[f"sp{t}"] = {
            int(bs): None
            if pd.isna(x.st_over_matched[bs])
            else num(1 / x.st_over_matched[bs], 2)
            for bs in x.index
        }
        row[f"top{t}"] = num(x.max_recall.iloc[0])
    msum.append(row)
D["msum"] = sorted(msum, key=lambda r: (r["n"], r["ds"], r["sweep"]))
D["band"] = dict(
    beyond=D["counts"]["beyond"],
    beyond_wins=D["counts"]["beyond_wins"],
    quarter=D["counts"]["quarter"],
    quarter_wins=D["counts"]["quarter_wins"],
    unreached=int(len(s0) - len(reached)),
)
# batch size: ms per query at bs 1 / 16 / 64, tuned for 0.90
bsc = []
for key in ["arxiv|c0_maincat", "pubmed|c0_mesh", "laion30m|tags4"]:
    ds, sw = key.split("|")
    rows = {}
    for t_ in [r for r in tab if r["key"] == key]:
        for name, rec, ms, sp, cfg, mb in t_["rows"]:
            rows.setdefault(name, {})[t_["bs"]] = (
                None if ms is None else num(ms / t_["bs"], 4)
            )
    bsc.append(
        dict(
            ds=lab[ds],
            sweep=sw,
            rows=[[k, v.get(1), v.get(16), v.get(64)] for k, v in rows.items()],
        )
    )
D["batch"] = bsc
# scale: LAION's two filters at 1-30 M, SilverTorch tuned for 0.90 against the best LiNR, and recall at 24 probes
D["scale"] = [
    dict(
        ds=r["ds"],
        sweep=r["sweep"],
        n=r["n"],
        sp=r["sp0.9"],
        frac=r["frac0.9"],
        top=r["top0.9"],
        r24=next(
            (
                p["recall"]
                for p in D["prod"]
                if p["ds"] == r["ds"] and p["sweep"] == r["sweep"]
            ),
            None,
        ),
        sp24=next(
            (
                p["sp16"]
                for p in D["prod"]
                if p["ds"] == r["ds"] and p["sweep"] == r["sweep"]
            ),
            None,
        ),
    )
    for r in D["msum"]
    if r["ds"].startswith("LAION")
]
D["c5tab"] = [
    dict(
        n=int(n),
        bs=int(bs),
        med=num(g.full_over_partial.median(), 2),
        lo=num(g.full_over_partial.min(), 2),
        hi=num(g.full_over_partial.max(), 2),
    )
    for (n, bs), g in c5n_all.groupby(["n_items", "perf_bs"])
]
# selectivity: synthetic pass rates, bs 16
sel = []
for v_ in D["curves"]:
    if v_["bs"] != 16:
        continue
    ser = {s_["name"]: {p[0]: (p[1], p[2]) for p in s_["pts"]} for s_ in v_["series"]}
    prs = sorted(ser["LiNR V1"])
    sel.append(
        dict(
            label=v_["label"],
            rows=[
                [
                    pct_(p),
                    num(ser["LiNR V1"][p][0], 2),
                    num(ser["LiNR V2"][p][0], 2),
                    ser["SilverTorch, 24 probes"].get(p, (None, None))[1],
                    num(ser["SilverTorch, 24 probes"].get(p, (None,))[0], 2),
                    num(ser["SilverTorch, tuned for 0.90"].get(p, (None,))[0], 2),
                ]
                for p in prs
            ],
        )
    )
D["sel"] = sel
# memory
mt = []
for ds in REAL:
    m = memall[memall.dataset == ds]
    st_ = m[m.arm == "SilverTorch"].sort_values("n_lists")
    mt.append(
        dict(
            ds=lab[ds],
            dim=int(m.dim.iloc[0]),
            v1=num(m[m.arm == "V1"].index_mib.iloc[0], 0),
            v3=num(m[m.arm == "V3"].index_mib.iloc[0], 0)
            if (m.arm == "V3").any()
            else None,
            st=num(st_.index_mib.iloc[0], 0),
            st_nl=int(st_.n_lists.iloc[0]),
            st_max=num(st_.index_mib.iloc[-1], 0),
            nl_max=int(st_.n_lists.iloc[-1]),
            b_min=num(st_.build_s.iloc[0], 1),
            b_max=num(st_.build_s.iloc[-1], 1),
        )
    )
D["mem"] = mt
# seed variance summary per dataset (cells with all three seeds)
vp = pd.read_csv(OUT + "variance_points.csv")
vp = vp[vp.seeds == 3].assign(
    t_spread=lambda d: d.st_ms_max / d.st_ms_min - 1,
    r_spread=lambda d: d.recall_max - d.recall_min,
    w_spread=lambda d: d.ratio_max / d.ratio_min - 1,
)
vt = pd.read_csv(OUT + "variance_tuned.csv")
vt = vt[(vt.seeds == 3) & (vt.reached == 3)]
vsum = []
for ds, g in vp.groupby("dataset"):
    w = vt[vt.dataset == ds]
    vsum.append(
        dict(
            ds=lab.get(ds, ds),
            points=int(len(g)),
            t_med=num(100 * g.t_spread.median(), 1),
            t_max=num(100 * g.t_spread.max(), 1),
            r_max=num(g.r_spread.max(), 3),
            w_med=num(100 * g.w_spread.median(), 1),
            cells=int(len(w)),
            flips=int(((w.matched_min < 1) & (w.matched_max > 1)).sum()),
        )
    )
D["vsum"] = vsum
D["vflip"] = [
    f"{lab.get(r.dataset, r.dataset)} {r.sweep}, bs {int(r.perf_bs)}, k {int(r.perf_k)}, {r.target:.2f}: {r.matched_min:.2f}–{r.matched_max:.2f}"
    for r in vt[(vt.matched_min < 1) & (vt.matched_max > 1)].itertuples()
]
# bloom against the exact filter, one row per filter, pooled over lists and batch sizes
D["cbsum"] = [
    dict(
        ds=lab.get(ds, ds),
        sweep=sw,
        n=len(g),
        med=num(g.bloom_over_clause.median(), 2),
        lo=num(g.bloom_over_clause.min(), 2),
        hi=num(g.bloom_over_clause.max(), 2),
        lists=f"{int(g.n_lists.min())}–{int(g.n_lists.max())}",
        cost=num(g.recall_cost.abs().max(), 4),
    )
    for (ds, sw), g in cb.groupby(["dataset", "sweep"])
]
D["cbtab"] = [
    dict(
        ds=lab.get(ds, ds),
        sweep=sw,
        nl=int(nl),
        n=len(g),
        probes=f"{int(g.n_probe.min())}–{int(g.n_probe.max())}",
        med=num(g.bloom_over_clause.median(), 2),
        lo=num(g.bloom_over_clause.min(), 2),
        hi=num(g.bloom_over_clause.max(), 2),
    )
    for (ds, sw, nl), g in cb.groupby(["dataset", "sweep", "n_lists"])
]
# parameter sensitivity: clusters on LAION 30M, and LiNR V3's candidate pool
nlt = []
for sw in ["tags4", "c0_domain"]:
    kn = next(k for k in D["knobs"] if k["key"] == f"laion30m|{sw}")
    for nl, pts in kn["curves"].items():
        at = dict(pts)
        tm = [p for p in kn["timed"].get(16, []) if p[0] == int(nl) and p[3] >= 0.9]
        best = min(tm, key=lambda p: p[2]) if tm else None
        nlt.append(
            dict(
                sweep=sw,
                nl=int(nl),
                mib=kn["mem"][int(nl)][0],
                build=kn["mem"][int(nl)][1],
                r8=at.get(0.125),
                r4=at.get(0.25),
                probes=None if best is None else best[1],
                ms=None if best is None else best[2],
            )
        )
D["nltab"] = nlt
pools = pd.read_csv(OUT + "c2_pools.csv")
D["pools"] = [
    dict(
        ds=lab[ds],
        sweep=sw,
        pass_rate=num(g.pass_rate.iloc[0], 4),
        pts=[[num(100 * p, 1), num(r)] for p, r in zip(g.pool, g.recall)],
    )
    for (ds, sw), g in pools.groupby(["dataset", "sweep"])
    if ds in lab
]
summ = json.load(open(OUT + "summary.json"))
D["prov"] = dict(
    legs=len(summ["runs"]),
    records=summ["records"],
    flagged=summ["h2h_flagged"],
    points=summ["h2h_rows"],
)
json.dump(D, open(sys.argv[2], "w"))
print(len(json.dumps(D)), {k: len(v) for k, v in D.items()})
print(json.dumps(D["counts"], indent=1))
print("\n".join(c[3] for c in claims))
