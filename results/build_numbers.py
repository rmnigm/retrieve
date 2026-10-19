"""Write NUMBERS.md: each fact the paper cites with its value read from a CSV in tables/, and the CSV line(s) it
comes from (line 1 is the header).

usage: python build_numbers.py [RESULTS]   (default: this directory)
"""

import sys
from pathlib import Path

import polars as pl

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
T = R / "tables"
rows = []
c = pl.col


def tab(name):
    return pl.read_csv(T / f"{name}.csv", infer_schema_length=None).with_row_index(
        "line", offset=2
    )


def lines(x):
    ls = x["line"].to_list()
    if len(ls) > 12:
        return f"{len(ls)} rows"
    if len(ls) == 1:
        return f"line {ls[0]}"
    if ls == list(range(ls[0], ls[0] + len(ls))):
        return f"lines {ls[0]}-{ls[-1]}"
    return "lines " + ", ".join(map(str, ls))


def fact(section, text, value, name, x, how=""):
    assert x.height > 0, text
    src = f"`{name}.csv` {lines(x)}" + (f" ({how})" if how else "")
    rows.append((section, text, value, src))


def rng(s, d=3):
    return (
        f"{s.min():.{d}f}-{s.max():.{d}f}" if s.min() != s.max() else f"{s.min():.{d}f}"
    )


# head-to-head counts
k = tab("head_to_head_counts")


def cnt(budget, kk="all", ft="all", tg="all"):
    return k.filter(
        (c("budget") == budget)
        & (c("k") == kk)
        & (c("filter_type") == ft)
        & (c("recall_target") == tg)
    )


S = "Head-to-head"
x = cnt("quarter")
r = x.row(0, named=True)
fact(
    S,
    "settings with a timed SilverTorch point (k 100 / 1000 / 2048 pooled); of them with one within a quarter of the lists",
    f"{r['settings']}; {r['timed_within_quarter']}",
    "head_to_head_counts",
    x,
    "settings, timed_within_quarter",
)
fact(
    S,
    "within a quarter: reached; SilverTorch faster (strict); wins / ties / losses under the tie band; median SilverTorch / LiNR",
    f"{r['reached']}; {r['silvertorch_faster']}; {r['wins']} / {r['ties']} / {r['losses']}; {r['median_silvertorch_over_linr']}",
    "head_to_head_counts",
    x,
)
fact(
    S,
    "within a quarter, settings with no unstable flag: reached; SilverTorch faster; median",
    f"{r['stable_reached']}; {r['stable_silvertorch_faster']}; {r['stable_median']}",
    "head_to_head_counts",
    x,
)
fact(
    S,
    "unstable flag (any timed arm the setting combines) among the settings reached within a quarter",
    f"{r['unstable']} of {r['reached']}",
    "head_to_head_counts",
    x,
)
sel = tab("head_to_head_selections")
fact(
    S,
    "unstable flag over all within-quarter selections, reached or not",
    f"{sel['timing_unstable'].sum()} of {sel.height}",
    "head_to_head_selections",
    sel,
    "sum of timing_unstable",
)
for kk in ("100", "1000", "2048"):
    x = cnt("quarter", kk)
    r = x.row(0, named=True)
    fact(
        S,
        f"k {kk}: settings; timed within a quarter; reached; SilverTorch faster; W / T / L; median",
        f"{r['settings']}; {r['timed_within_quarter']}; {r['reached']}; {r['silvertorch_faster']}; {r['wins']} / {r['ties']} / {r['losses']}; {r['median_silvertorch_over_linr']}",
        "head_to_head_counts",
        x,
    )
x = cnt("deeper")
r = x.row(0, named=True)
fact(
    S,
    "reached only beyond a quarter (one half or three quarters of the lists): settings; SilverTorch faster; W / T / L",
    f"{r['reached']}; {r['silvertorch_faster']}; {r['wins']} / {r['ties']} / {r['losses']}",
    "head_to_head_counts",
    x,
)
x = cnt("any")
r = x.row(0, named=True)
fact(
    S,
    "any depth: reached; SilverTorch faster; W / T / L; not reached at any depth",
    f"{r['reached']}; {r['silvertorch_faster']}; {r['wins']} / {r['ties']} / {r['losses']}; {r['settings'] - r['reached']}",
    "head_to_head_counts",
    x,
)
for kk in ("100", "all"):
    for ft in ("real", "synthetic"):
        for tg in ("0.90", "0.95"):
            x = cnt("quarter", kk, ft, tg)
            r = x.row(0, named=True)
            fact(
                S,
                f"{ft} filters, k {kk}, target {tg}: reached within a quarter of settings; SilverTorch faster; median",
                f"{r['reached']} of {r['settings']}; {r['silvertorch_faster']}; {r['median_silvertorch_over_linr']}",
                "head_to_head_counts",
                x,
            )
pb = tab("head_to_head_probe_budgets")
x = pb.filter((c("outcome") == "within_quarter") & (c("silvertorch_over_linr") >= 1))
fact(
    S,
    "the strict losses within a quarter (dataset filter batch k target: ratio)",
    "; ".join(
        f"{d} {f} b{b} k{kk} {t}: {v}"
        for d, f, b, kk, t, v in x.select(
            "dataset", "filter", "batch", "k", "recall_target", "silvertorch_over_linr"
        ).iter_rows()
    ),
    "head_to_head_probe_budgets",
    x,
)
x = pb.filter(c("verdict") == "tie")
fact(
    S,
    "ties under the dataset's tie band (dataset filter batch k target outcome: ratio, band)",
    "; ".join(
        f"{d} {f} b{b} k{kk} {t} {o}: {v}, {bd}"
        for d, f, b, kk, t, o, v, bd in x.select(
            "dataset",
            "filter",
            "batch",
            "k",
            "recall_target",
            "outcome",
            "silvertorch_over_linr",
            "tie_band",
        ).iter_rows()
    ),
    "head_to_head_probe_budgets",
    x,
)
x = pb.filter(
    (c("filter_type") == "synthetic")
    & (c("recall_target") == 0.90)
    & (c("outcome") != "not_reached")
)
fact(
    S,
    "synthetic filters at 0.90, any depth: settings; SilverTorch faster; speed-up range",
    f"{x.height}; {(x['silvertorch_over_linr'] < 1).sum()}; {1 / x['silvertorch_over_linr'].max():.2f}-{1 / x['silvertorch_over_linr'].min():.1f}x",
    "head_to_head_probe_budgets",
    x,
    "filter_type synthetic, target 0.90, reached",
)
x = pb.filter((c("dataset") == "goodreads-synth") & (c("filter") == "p0001"))
fact(
    S,
    "Goodreads synthetic 0.1 %: outcome, probe fraction and ratio at batch 1 / 16 / 64, target 0.90 and 0.95",
    ", ".join(
        f"{o} {pf} b{b} t{t} {v}"
        for o, pf, b, t, v in x
        .select("outcome", "probe_fraction", "batch", "recall_target", "silvertorch_over_linr")
        .iter_rows()
    ),
    "head_to_head_probe_budgets",
    x,
)
sp = tab("timing_spread")
fact(
    S,
    "tie band (half the sum of SilverTorch's and LiNR's seed time spreads) per dataset",
    ", ".join(
        f"{d} {100 * b:.2f} %" for d, b in sp.select("dataset", "tie_band").iter_rows()
    ),
    "timing_spread",
    sp,
)
lat = tab("silvertorch_latency").filter(c("batch") == 1)
fact(
    S,
    "SilverTorch at batch 1, within-quarter picks: latency range per dataset (ms)",
    ", ".join(
        f"{d} {a}-{b}"
        for d, a, b in lat.select("dataset", "min_ms", "max_ms").iter_rows()
    ),
    "silvertorch_latency",
    lat,
)
eq = tab("evaluated_queries").filter(~c("dataset").str.ends_with("-synth"))
fact(
    S,
    "queries with a non-empty exact filtered result per real filter (recall's population, of 10,000)",
    ", ".join(
        f"{d} {f} {n}"
        for d, f, n in eq.select(
            "dataset", "filter", "min_evaluated_queries"
        ).iter_rows()
    ),
    "evaluated_queries",
    eq,
)

S = "LiNR"
v = tab("linr_v2_over_v1")
for b in (1, 16):
    x = v.filter(
        (c("batch") == b)
        & c("pass_rate").is_between(0.09, 0.11)
        & ~c("timing_unstable")
        & ~c("dataset").str.starts_with("arxiv-corr")
    )
    fact(
        S,
        f"V2 / V1 near 10 % pass, batch {b}, uniform sets, no unstable flag",
        rng(x["v2_over_v1"]),
        "linr_v2_over_v1",
        x,
    )
cs = tab("claims_summary")
x = cs.filter(c("quantity") == "pass rate where V2 / V1 reaches 1")
fact(
    S,
    "pass rate where V2 / V1 reaches 1 at batch 1 (uniform sets)",
    f"{x['min'][0]}-{x['max'][0]} (n {x['n'][0]})",
    "claims_summary",
    x,
)
v64 = tab("linr_v2_over_v1_batch64")
fact(
    S,
    "V2 / V1 at batch 64, every synthetic point (V2 slower everywhere)",
    rng(v64["v2_over_v1"]),
    "linr_v2_over_v1_batch64",
    v64,
)
lo = v64.sort("pass_rate").group_by("dataset", maintain_order=True).first().sort("line")
fact(
    S,
    "V2 / V1 at batch 64 at each set's lowest pass rate (about 0.1 % or 20 %)",
    rng(lo["v2_over_v1"]),
    "linr_v2_over_v1_batch64",
    lo,
)
nt = tab("linr_native_pytorch")
for var in ("V1", "V2"):
    for imp in ("native", "compiled"):
        for mode in ("graph", "eager"):
            for b in (1, 16):
                x = nt.filter(
                    (c("variant") == var)
                    & (c("implementation") == imp)
                    & (c("mode") == mode)
                    & (c("batch") == b)
                )
                if x.height:
                    fact(
                        S,
                        f"{imp} PyTorch {var} / Triton {var}, {mode}, batch {b} ({', '.join(sorted(x['dataset'].unique()))})",
                        rng(x["native_over_kernel"]),
                        "linr_native_pytorch",
                        x,
                    )
nv = tab("linr_native_v2_over_v1")
for mode in ("graph", "eager"):
    for b in (1, 16):
        x = nv.filter((c("mode") == mode) & (c("batch") == b))
        fact(
            S,
            f"un-fused: native V2 / native V1, {mode}, batch {b}, Goodreads synthetic 1 / 10 / 100 %",
            ", ".join(map(str, x["native_v2_over_native_v1"])),
            "linr_native_v2_over_v1",
            x,
        )
ni = tab("linr_native_indexing")
for b in (1, 16):
    x = ni.filter(c("batch") == b)
    fact(
        S,
        f"literal indexing with host synchronisation (eager) / faster Triton kernel (graph), batch {b}, Goodreads / arXiv / PubMed",
        rng(x["native_index_over_kernel"]),
        "linr_native_indexing",
        x,
    )
x = ni.filter(c("native_index_over_kernel") == ni["native_index_over_kernel"].max())
fact(
    S,
    "literal indexing: the largest ratio (dataset, pass rate, batch: native ms, kernel ms, ratio)",
    ", ".join(
        f"{d} {p} b{bb}: {n} / {kk} ms, {r}x"
        for d, p, bb, n, kk, r in x.select(
            "dataset",
            "pass_rate",
            "batch",
            "native_index_ms",
            "kernel_ms",
            "native_index_over_kernel",
        ).iter_rows()
    ),
    "linr_native_indexing",
    x,
)
w = tab("linr_v3_wins")
x = w.filter(c("v3_faster") > 0)
fact(
    S,
    "V3 faster than the faster exact variant (filter type, batch, pool: faster / faster with no flag / timed, smallest ratio)",
    "; ".join(
        f"{ft} b{b} {p}: {a}/{s}/{t}, {m}"
        for ft, b, p, t, a, s, m in x.select(
            "filter_type",
            "batch",
            "v3_pool",
            "timed",
            "v3_faster",
            "v3_faster_stable",
            "min_v3_over_exact",
        ).iter_rows()
    ),
    "linr_v3_wins",
    x,
)
ve = tab("linr_v3_vs_exact")
x = ve.filter(
    (c("v3_over_exact") < 1)
    & ~c("timing_unstable")
    & ~c("dataset").str.ends_with("-synth")
)
fact(
    S,
    "V3 wins on real filters with no unstable flag (dataset filter batch pool: ratio, recall)",
    "; ".join(
        f"{d} {f} b{b} {p}: {r_}, {rc}"
        for d, f, b, p, r_, rc in x.select(
            "dataset", "filter", "batch", "v3_pool", "v3_over_exact", "v3_recall"
        ).iter_rows()
    ),
    "linr_v3_vs_exact",
    x,
)
cl = tab("linr_v3_code_length").filter(
    (c("k") == 100) & (c("v3_pool") == 0.01) & ~c("dataset").str.ends_with("-synth")
)
for d in ("goodreads", "pubmed"):
    for bits in sorted(cl.filter(c("dataset") == d)["k_bits"].unique()):
        x = cl.filter((c("dataset") == d) & (c("k_bits") == bits) & (c("batch") == 1))
        fact(
            S,
            f"V3 recall@100 at a 1 % pool, {d} real filters, {bits} bits",
            rng(x["v3_recall"]),
            "linr_v3_code_length",
            x,
            "batch 1",
        )
x = cl.filter(c("ms_over_shortest_code") != 1.0)
fact(
    S,
    "V3 time, longer over shorter code at the same pool (1 % pool, k 100, real filters)",
    rng(x["ms_over_shortest_code"]),
    "linr_v3_code_length",
    x,
)

S = "SilverTorch"
cm = tab("codesign_medians")
for n in ("laion1m", "laion3m", "laion10m", "laion30m"):
    for b in (16, 64):
        x = cm.filter((c("dataset") == n) & (c("batch") == b))
        fact(
            S,
            f"co-design whole index / searched clusters, {n}, batch {b}: median (stable pairs) all / domain / tags",
            " / ".join(
                str(x.filter(c("filter") == f)["median"][0])
                for f in ("all", "c0_domain", "tags4")
            )
            + f" (stable {x.filter(c('filter') == 'all')['stable_pairs'][0]} of {x.filter(c('filter') == 'all')['pairs'][0]})",
            "codesign_medians",
            x,
        )
x = cm.filter(c("dataset").str.starts_with("laion") & (c("filter") == "all"))
fact(
    S,
    "co-design share of the lists searched per catalogue (32 and 128 probes)",
    ", ".join(
        f"{d} {a:.4g}-{b:.4g}"
        for d, a, b in x.unique("dataset", maintain_order=True)
        .select("dataset", "min_share_searched", "max_share_searched")
        .iter_rows()
    ),
    "codesign_medians",
    x,
)
bw = tab("bloom_width")
x = bw.filter((c("m_bits") == 1024) & (c("k_hash") == 5))
fact(
    S,
    "bloom 1,024 bits / 5 hashes: false-positive rate, mean and per-query max (dataset filter)",
    "; ".join(
        f"{d} {f}: {m} / {mx}"
        for d, f, m, mx in x.select(
            "dataset", "filter", "fp_rate_mean", "fp_rate_max"
        ).iter_rows()
    ),
    "bloom_width",
    x,
)
x = bw.filter((c("dataset") == "pubmed") & (c("k_hash") == 5))
fact(
    S,
    "PubMed MeSH, 5 hashes, per width: mean / max false-positive rate, recall@100 (batch 16, one fixed probe setting)",
    "; ".join(
        f"{m}: {a} / {b}, {r_}"
        for m, a, b, r_ in x.select(
            "m_bits", "fp_rate_mean", "fp_rate_max", "recall_at_100"
        ).iter_rows()
    ),
    "bloom_width",
    x,
)
x = bw.filter((c("dataset") == "pubmed") & (c("k_hash") == 5) & (c("m_bits") == 128))
fact(
    S,
    "load-matched point: PubMed at 128 bits holds at most 8 hashed terms per item (4 MeSH + 4 single-valued clauses), at least 16 bits per term, against Meta's about 1,024 / 60 = 17; mean / max FP, recall",
    f"{x['fp_rate_mean'][0]} / {x['fp_rate_max'][0]}, {x['recall_at_100'][0]}",
    "bloom_width",
    x,
    "terms per item: docs/system/datasets.md (PubMed clauses) and the index build over every non-pad (clause, value)",
)
rb = tab("recall_bridge_correlation")
fact(
    S,
    "recall bridge: Pearson / Spearman of oracle vs held-out recall (dataset dim k: n)",
    "; ".join(
        f"{d} d{dm} k{kk}: {p} / {s} (n {n})"
        for d, dm, kk, n, p, s in rb.select(
            "dataset", "dim", "k", "n", "pearson", "spearman"
        ).iter_rows()
    ),
    "recall_bridge_correlation",
    rb,
)
x = rb.filter(c("dataset") == "goodreads")
fact(
    S,
    "recall bridge, Goodreads genre: oracle and held-out recall ranges at k 100",
    f"{x['oracle_min'][0]}-{x['oracle_max'][0]}; {x['heldout_min'][0]}-{x['heldout_max'][0]}",
    "recall_bridge_correlation",
    x.head(1),
)
rs = tab("reverse_clause_selections").filter((c("recall_target") == 0.90))
for b in (1, 16, 64):
    x = rs.filter(c("batch") == b)
    fact(
        S,
        f"reverse-clause filters at 0.90, batch {b}: LiNR / SilverTorch exact clause filter (speed-up) per filter (separate runs)",
        ", ".join(
            f"{d} {f} {1 / v:.2f}x"
            for d, f, v in x.select(
                "dataset", "filter", "silvertorch_over_linr"
            ).iter_rows()
        ),
        "reverse_clause_selections",
        x,
    )
x = tab("reverse_clause_selections").filter(c("silvertorch_ms").is_null())
fact(
    S,
    "reverse-clause filters not reaching 0.95 within a quarter (best recall)",
    ", ".join(
        f"{d} {f} b{b} {m}"
        for d, f, b, m in x.select(
            "dataset", "filter", "batch", "silvertorch_max_recall"
        ).iter_rows()
    ),
    "reverse_clause_selections",
    x,
)

out = [
    "# Numbers",
    "",
    "Facts the paper cites, each read from a CSV in [`tables/`](tables/) by the script that wrote this",
    "file; the source column gives the CSV line(s) (line 1 is the header). Flag definition: a setting",
    "is `timing_unstable` when any timed record it combines (SilverTorch's pick, the exact LiNR of its",
    "run, V3 at its fastest pool reaching the target) carries the `unstable` flag (relative range of",
    "its three window medians above 5 %). Tie band: half the sum of the dataset's SilverTorch and LiNR",
    "seed spreads (`timing_spread.csv`); a ratio within the band of 1 is a tie. Counts are seed 0.",
    "",
]
for sec in dict.fromkeys(s for s, *_ in rows):
    out += [f"## {sec}", "", "| fact | value | source |", "|---|---|---|"]
    out += [f"| {t} | {v} | {src} |" for s, t, v, src in rows if s == sec]
    out.append("")
(R / "NUMBERS.md").write_text("\n".join(out))
print(len(rows), "facts")
