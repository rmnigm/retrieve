"""research/paper/results.md from the page data: results_md.py DATA_JSON TEMPLATE OUT; {T1}... in TEMPLATE are tables."""

import json
import sys
from pathlib import Path

D = json.load(open(sys.argv[1]))


def f(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def fx(v):
    return "–" if v is None else (f"{v:.1f}×" if v >= 10 else f"{v:.2f}×")


def pct(p):
    return (
        f"{100 * p:.2f} %"
        if p < 0.01
        else f"{100 * p:.1f} %"
        if p < 0.1
        else f"{100 * p:.0f} %"
    )


def md(head, rows):
    out = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["--:"] * (len(head) - 1)) + "|",
    ]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def sig(v):
    return "<0.001" if v < 0.0005 else f"{float(f'{v:.1g}'):g}"


def pv(ds):
    return D["pv"].get(ds, D["pv"]["*"])


def pm_t(ds):
    return f"±{50 * pv(ds)[0]:.1f} %"


def pm_r(ds):
    return "±" + sig(pv(ds)[1] / 2)


def seedcap(ds):
    return f" SilverTorch's seed spread on this dataset: time {pm_t(ds)}, recall {pm_r(ds)}."


def bold(s, cond):
    return f"**{s}**" if cond else s


def t1(key, bs):
    r = next(x for x in D["tab"] if x["key"] == key and x["bs"] == bs)
    best = min(x[2] for x in r["rows"] if x[2] is not None)
    cap = f"{r['ds']}, filter `{r['sweep']}` (pass {pct(r['pass_rate'])}), d = {r['dim']}, k = 100, batch {bs}. ±: SilverTorch's seed spread.\n\n"
    t_, r_ = pv(r["ds"])
    return cap + md(
        [
            "Algorithm",
            "Recall@100",
            "t_med, ms",
            "Speed-up vs V1",
            "Setting",
            "Index, MiB",
        ],
        [
            [
                n,
                f(rec, 3)
                + (
                    " ±" + sig(r_ / 2)
                    if n.startswith("Silver") and rec is not None
                    else ""
                ),
                bold(f(ms, 3), ms == best)
                + (
                    " ±" + sig(ms * t_ / 2)
                    if n.startswith("Silver") and ms is not None
                    else ""
                ),
                bold(fx(sp), ms == best),
                cfg,
                "–" if mb is None else f"{mb:,.0f}",
            ]
            for n, rec, ms, sp, cfg, mb in r["rows"]
        ],
    )


def share(r, t):
    v = r[f"frac{t}"]
    return (
        f"top {f(r[f'top{t}'])}" if v is None else pct(v) + (" ◐" if v > 0.25 else "")
    )


def sp(v):
    return "–" if v is None else bold(fx(v), v >= 1)


T = {}
T["T1a"] = t1("pubmed|c0_mesh", 16)
T["T1b"] = t1("laion30m|tags4", 16)
T["T1c"] = t1("laion1m|c0_domain", 16)


def t2(syn):
    return md(
        [
            "Dataset · filter",
            "Pass",
            "Seed ±",
            "0.90: share",
            "bs 1",
            "bs 16",
            "bs 64",
            "0.95: share",
            "bs 1",
            "bs 16",
            "bs 64",
        ],
        [
            [
                f"{r['ds']} · {r['sweep'] if r['syn'] else '`' + r['sweep'] + '`'}",
                pct(r["pass_rate"]),
                pm_t(r["ds"]),
                share(r, 0.9),
            ]
            + [sp(r["sp0.9"].get(str(b))) for b in (1, 16, 64)]
            + [share(r, 0.95)]
            + [sp(r["sp0.95"].get(str(b))) for b in (1, 16, 64)]
            for r in D["msum"]
            if r["syn"] == syn
        ],
    )


T["T2"] = t2(False)
T["T2s"] = t2(True)


def us(v):
    return "–" if v is None else f"{1000 * v:.1f} µs" if v < 0.1 else f"{v:.3f} ms"


b = D["batch"][0]
best = [min(r[c] for r in b["rows"] if r[c] is not None) for c in (1, 2, 3)]
T["T3"] = f"{b['ds']}, filter `{b['sweep']}`, k = 100.{seedcap(b['ds'])}\n\n" + md(
    ["Algorithm", "B = 1", "B = 16", "B = 64"],
    [
        [r[0]] + [bold(us(r[c]), r[c] == best[c - 1]) for c in (1, 2, 3)]
        for r in b["rows"]
    ],
)
T["T4"] = md(
    [
        "Filter · size",
        "Seed ±",
        "0.90: share",
        "bs 1",
        "bs 16",
        "bs 64",
        "24 probes: recall",
        "24 probes: bs 16",
    ],
    [
        [
            f"`{r['sweep']}` · {r['ds'].replace('LAION ', '')}",
            pm_t(r["ds"]),
            "top " + f(r["top"])
            if r["frac"] is None
            else pct(r["frac"]) + (" ◐" if r["frac"] > 0.25 else ""),
        ]
        + [sp(r["sp"].get(str(b_))) for b_ in (1, 16, 64)]
        + [f(r["r24"], 3), fx(r["sp24"])]
        for r in sorted(D["scale"], key=lambda r: (r["sweep"], r["n"]))
    ],
)
ns = sorted({r["n"] for r in D["c5tab"]})
c5 = {(r["n"], r["bs"]): r for r in D["c5tab"]}
T["T5"] = md(
    ["Catalogue", "Batch 16", "Batch 64"],
    [
        [f"{n / 1e6:g} M items"]
        + [
            f"{bold(fx(c5[n, b_]['med']), c5[n, b_]['med'] >= 1)} ({f(c5[n, b_]['lo'])}–{f(c5[n, b_]['hi'])})"
            for b_ in (16, 64)
        ]
        for n in ns
    ],
)
s = next(x for x in D["sel"] if x["label"].startswith("LAION-synth 30M"))
T["T6"] = f"{s['label']}, batch 16, k = 100.{seedcap(s['label'])}\n\n" + md(
    [
        "Pass rate",
        "V1, ms",
        "V2, ms",
        "SilverTorch 24: recall",
        "SilverTorch 24: ms",
        "SilverTorch at 0.90: ms",
    ],
    [[r[0], f(r[1]), f(r[2]), f(r[3], 3), f(r[4]), f(r[5])] for r in s["rows"]],
)
T["T7"] = md(
    [
        "Dataset",
        "d",
        "LiNR V1 / V2",
        "LiNR V3",
        "SilverTorch",
        "SilverTorch / V1",
        "Build, s (clusters)",
    ],
    [
        [
            r["ds"],
            r["dim"],
            f"{r['v1']:,.0f}",
            f"{r['v3']:,.0f}",
            f"**{r['st']:,.0f}**"
            + ("" if r["st_max"] == r["st"] else f"–{r['st_max']:,.0f}"),
            f(r["st"] / r["v1"]),
            (
                f(r["b_min"], 1)
                if r["b_min"] == r["b_max"]
                else f"{f(r['b_min'], 1)}–{r['b_max']:,.0f}"
            )
            + f" ({r['st_nl']:,}"
            + ("" if r["st_nl"] == r["nl_max"] else f"–{r['nl_max']:,}")
            + ")",
        ]
        for r in D["mem"]
    ],
)
n_of = {x["ds"]: x["n"] for x in D["tab"]}
T["T8"] = md(
    [
        "Dataset · filter",
        "Clusters",
        "Items per cluster",
        "Probes",
        "Bloom ÷ exact (range)",
        "Seed ±",
    ],
    [
        [
            f"{r['ds']} · `{r['sweep']}`",
            f"{r['nl']:,}",
            f"{n_of[r['ds']] / r['nl']:,.0f}",
            r["probes"].split("–")[0]
            if len(set(r["probes"].split("–"))) == 1
            else r["probes"],
            f"{bold(f(r['med']), r['med'] < 1)} ({f(r['lo'])}–{f(r['hi'])})",
            pm_t(r["ds"]),
        ]
        for r in sorted(
            D["cbtab"], key=lambda r: (n_of[r["ds"]], r["ds"], r["sweep"], r["nl"])
        )
    ],
)
T["T9"] = (
    seedcap("LAION 30M").strip()
    + "\n\n"
    + md(
        [
            "Filter · clusters",
            "Index, MiB",
            "Build, s",
            "Recall at 1/8",
            "Recall at 1/4",
            "Probes for 0.90",
            "t_med, ms",
        ],
        [
            [
                f"`{r['sweep']}` · {r['nl']:,}",
                f"{r['mib']:,.0f}",
                f(r["build"], 0),
                f(r["r8"], 3),
                f(r["r4"], 3),
                "–" if r["probes"] is None else f"{r['probes']:,}",
                f(r["ms"]),
            ]
            for r in D["nltab"]
        ],
    )
)
pick = [
    "Goodreads 0.8M|c0_genre",
    "arXiv 3M|c0_maincat",
    "LAION 30M|tags4",
    "PubMed 10M|c0_mesh",
]
pools = [p for p in D["pools"] if p["ds"] + "|" + p["sweep"] in pick]
T["T10"] = md(
    ["Dataset · filter", "Pass"] + [f"{p[0]:g} %" for p in pools[0]["pts"]],
    [
        [f"{p['ds']} · `{p['sweep']}`", pct(p["pass_rate"])]
        + [bold(f(x[1], 3), x[1] >= 0.9) for x in p["pts"]]
        for p in pools
    ],
)
T["T11"] = md(
    [
        "Dataset",
        "Seeds",
        "Settings",
        "Time range",
        "Recall range",
        "Tuned cells",
        "Winner changes",
    ],
    [
        [
            r["ds"],
            r["seeds"],
            r["points"],
            f"{f(r['t'], 1)} %",
            f(r["r"], 3),
            r["cells"],
            r["flips"],
        ]
        for r in D["pvtab"]
    ],
)
ps = D["psweep"]


def at(f_, nl, np_):
    return next((p for p in f_["pts"].get(str(nl), []) if p[0] == np_), None)


def first90(f_, nl):
    return next((p for p in f_["pts"].get(str(nl), []) if p[1] >= 0.9), None)


def cell(p):
    return "–" if p is None else f"{f(p[1], 3)} · {f(p[2])} ms"


T["Tps"] = md(
    [
        "Filter",
        "24 probes",
        "512 probes",
        "1,024 probes",
        "Smallest for 0.90",
        "Tuned for 0.90",
        "LiNR V1, ms",
    ],
    [
        [f_["label"]]
        + [cell(at(f_, 4096, n_)) for n_ in (24, 512, 1024)]
        + [
            "–"
            if first90(f_, 4096) is None
            else f"{first90(f_, 4096)[0]:,} probes · {f(first90(f_, 4096)[2])} ms",
            next(
                (
                    f"{k[1]:,} / {k[2]:,} · {f(k[4])} ms"
                    for k in f_["picks"]
                    if k[0] == 0.9
                ),
                "–",
            ),
            f(f_["v1"]),
        ]
        for f_ in ps["filters"]
    ],
)
vt_, vb = D["vtext"], D["vbelow"]
T.update(
    v_t=D["pvtext"]["t"],
    v_r=D["pvtext"]["r"],
    pv_all=D["pvtext"]["all_t"],
    v_flips="none" if vt_["flips"] == 0 else vt_["flips"],
    v_cells=vt_["cells"],
    v_below=f"At that setting {vb['n']} cells ({', '.join(vb['cells'])}) dip below the target at a later seed, by at most {vb['gap']}."
    if vb["n"]
    else "",
)
c_ = D["counts"]
T.update(
    c_cells=c_["cells"],
    c_q=c_["quarter"],
    c_qw=c_["quarter_wins"],
    c_qmed=f"{c_['quarter_med']:.2f}",
    c_qx=f"{1 / c_['quarter_med']:.1f}",
    c_b=c_["beyond"],
    c_bw=c_["beyond_wins"],
    c_u=D["band"]["unreached"],
)
T["T12"] = md(
    ["", "Verdict", "The paper", "We measured"],
    [[f"**{i}**", v, c, g] for i, _, v, c, g in D["verdicts"]],
)
T["bridge"] = (
    f"Pearson {D['bridge']['bloom100']} at k 100 and {D['bridge']['bloom1000']} at k 1000 over SilverTorch's bloom settings, {D['bridge']['all_lo']}–{D['bridge']['all_hi']} per filter over every method"
)
T["fp"] = (
    f"at the timed settings at most {D['fp']['share_max']} % of the items the bloom passes actually fail the filter (none at {D['fp']['zero']} of {D['fp']['points']} graded settings), there are no false negatives, and recall moves by at most {D['fp']['cost_max']}"
)
T["vflip"] = "; ".join(D["vflip"]).replace(" p1,", " with no filter,") or "none"
T["prov"] = (
    f"{D['prov']['legs']} subtrees and {D['prov']['records']:,} records; {D['prov']['flagged']} of {D['prov']['points']} timed points stay unstable in every run"
)
Path(sys.argv[3]).write_text(Path(sys.argv[2]).read_text().format(**T))
