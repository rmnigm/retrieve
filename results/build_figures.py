"""Draw the paper's figures from the CSVs in tables/ and the quoted values in originals.csv.

usage: python build_figures.py [TABLES] [OUT]   (defaults: tables/, figures/)

One figure per function, each written as PDF and PNG. Hollow markers are points the caption sets apart (a pick
deeper than a quarter of the lists, or a timing flagged unstable); every number drawn comes from a CSV.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TABLES = Path(sys.argv[1] if len(sys.argv) > 1 else "tables")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
ORIGINALS = Path(__file__).parent / "originals.csv"

BATCH = {
    1: "#2a78d6",
    16: "#eb6834",
    64: "#1baf7a",
}  # categorical slots 1-3, validated all-pairs
MARK = {1: "o", 16: "s", 64: "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SYNTH_ROWS = ["p0001", "p001", "p01", "p02", "p03", "p05", "p1"]
SWEEP_FILTERS = ["p1", "p01", "p001", "p0001"]  # the paper's four probe-sweep filters
SWEEP_LISTS = 4096
REAL_ORDER = [
    "goodreads",
    "arxiv",
    "pubmed",
    "laion1m",
    "laion3m",
    "laion10m",
    "laion30m",
]
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
    }
)


def read(name):
    return pl.read_csv(TABLES / f"{name}.csv", infer_schema_length=None)


def original(quantity):
    o = pl.read_csv(ORIGINALS)
    return o.filter(pl.col("quantity") == quantity)["value"].item()


def pct(p):
    return f"{100 * p:g} %"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plain(ax, sizes=None, yticks=None):
    """Plain numbers on log axes; catalogue sizes as `1 M`, `3 M`, ... ticks."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(f)
    ax.yaxis.set_minor_formatter(f)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if sizes is not None:
        ax.set_xticks(sizes, [f"{n / 1e6:g} M" for n in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def batch_handles(batches, hollow=None):
    h = [
        Line2D([], [], color=BATCH[b], marker=MARK[b], ls="", label=f"batch {b}")
        for b in batches
    ]
    if hollow:
        h.append(
            Line2D([], [], color=MUTED, marker="o", mfc="none", ls="", label=hollow)
        )
    return h


def linr_v2_over_v1():
    """LiNR V2 / V1 against the pass rate on the uniform synthetic filters, batch 1, 16 and 64 (k 100), one line per
    catalogue and dimension; hollow points are flagged unstable. Stars: LiNR's own un-fused ratios at 11 %."""
    x = pl.concat([read("linr_v2_over_v1"), read("linr_v2_over_v1_batch64")]).filter(
        ~pl.col("dataset").str.starts_with("arxiv-corr")
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (ds, dim, b), g in x.sort("pass_rate").group_by(
        "dataset", "dim", "batch", maintain_order=True
    ):
        ax.plot(g["pass_rate"], g["v2_over_v1"], color=BATCH[b], lw=0.9, alpha=0.8)
        for u in (False, True):
            s = g.filter(pl.col("timing_unstable") == u)
            ax.plot(
                s["pass_rate"],
                s["v2_over_v1"],
                ls="",
                marker=MARK[b],
                ms=3.5,
                color=BATCH[b],
                mfc="none" if u else BATCH[b],
            )
    p = original("pass rate of the Table 3 setting")
    ax.plot(p, original("V2 / V1 latency (PyTorch)"), "*", ms=10, color=INK)
    ax.plot(
        p, original("V2 / V1 latency (TensorFlow)"), "*", ms=10, color=INK, mfc="none"
    )
    n = read("linr_native_v2_over_v1").filter(
        (pl.col("batch") == 1) & (pl.col("mode") == "graph")
    )
    ax.plot(n["pass_rate"], n["native_v2_over_native_v1"], "D", ms=4, color=MUTED)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set(
        xscale="log", yscale="log", xlabel="pass rate", ylabel="LiNR V2 / V1 latency"
    )
    plain(ax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    h = batch_handles(sorted(x["batch"].unique()), hollow="flagged unstable")
    h += [
        Line2D([], [], color=INK, marker="*", ms=8, ls="", label="LiNR paper, PyTorch"),
        Line2D(
            [],
            [],
            color=INK,
            marker="*",
            ms=8,
            mfc="none",
            ls="",
            label="LiNR paper, TensorFlow",
        ),
        Line2D(
            [],
            [],
            color=MUTED,
            marker="D",
            ms=4,
            ls="",
            label="ours, both native (batch 1)",
        ),
    ]
    ax.legend(handles=h, loc="upper left", fontsize=7)
    save(fig, "linr_v2_over_v1")


def head_to_head():
    """SilverTorch / best LiNR latency at each recall target, k 100: one row per real filter and per synthetic pass
    rate (all catalogues pooled). Filled: picked within a quarter of the lists; hollow: only deeper; x at the right
    edge: not reached at any depth."""
    x = read("head_to_head_probe_budgets").filter(pl.col("k") == 100)
    real = (
        x.filter(pl.col("filter_type") == "real")
        .select("dataset", "filter")
        .unique()
        .with_columns(
            order=pl.col("dataset").replace_strict(
                REAL_ORDER, list(range(len(REAL_ORDER)))
            )
        )
        .sort("order", "filter")
        .drop("order")
    )
    rows = [
        (f"{d} {f}", (pl.col("dataset") == d) & (pl.col("filter") == f))
        for d, f in real.iter_rows()
    ]
    synth = x.filter(pl.col("filter_type") == "synthetic")
    for f in SYNTH_ROWS:
        rate = synth.filter(pl.col("filter") == f)["pass_rate"].median()
        rows.append(
            (
                f"synthetic {pct(round(rate, 3))}",
                (pl.col("filter_type") == "synthetic") & (pl.col("filter") == f),
            )
        )
    edge = x["silvertorch_over_linr"].max() * 1.6
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 0.22 * len(rows) + 0.9), sharey=True)
    for ax, target in zip(axes, (0.90, 0.95)):
        t = x.filter(pl.col("recall_target") == target)
        for i, (_, cond) in enumerate(rows):
            g = t.filter(cond)
            for b in BATCH:
                gb = g.filter(pl.col("batch") == b)
                y = i + {1: -0.2, 16: 0.0, 64: 0.2}[b]
                for outcome, face in (("within_quarter", BATCH[b]), ("deeper", "none")):
                    s = gb.filter(pl.col("outcome") == outcome)
                    ax.plot(
                        s["silvertorch_over_linr"],
                        [y] * s.height,
                        MARK[b],
                        ms=3.5,
                        color=BATCH[b],
                        mfc=face,
                    )
                if gb.filter(pl.col("outcome") == "not_reached").height:
                    ax.plot(edge, y, "x", ms=4, color=BATCH[b])
        ax.axvline(1, color=INK, lw=0.8)
        ax.set(
            xscale="log",
            title=f"recall target {target:.2f}",
            xlabel="SilverTorch / best LiNR latency",
        )
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        )
        ax.title.set_size(8)
    axes[0].set_yticks(range(len(rows)), [r for r, _ in rows])
    axes[0].invert_yaxis()
    h = batch_handles(BATCH, hollow="beyond a quarter of the lists")
    h.append(Line2D([], [], color=MUTED, marker="x", ls="", label="not reached"))
    fig.legend(
        handles=h, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02)
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "head_to_head")


def catalogue_size():
    """(a) SilverTorch's speed-up over the best LiNR on LAION's real filters at recall 0.90 against the catalogue size
    (filled within a quarter of the lists, hollow deeper). (b) Meta's published code: whole-index over searched-cluster
    bloom filtering per stable pair (unstable pairs left out), with the SilverTorch paper's reported range at 20 M."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw=dict(wspace=0.3))
    x = read("head_to_head_probe_budgets").filter(
        pl.col("dataset").str.contains(r"^laion\d+m$")
        & (pl.col("recall_target") == 0.90)
        & (pl.col("k") == 100)
        & (pl.col("outcome") != "not_reached")
    )
    size = {f"laion{n}m": n * 1e6 for n in (1, 3, 10, 30)}
    x = x.with_columns(
        n_items=pl.col("dataset").replace_strict(size, return_dtype=pl.Float64)
    )
    shift = {"c0_domain": 0.94, "tags4": 1.06}
    for (f, bs), g in x.sort("n_items").group_by(
        "filter", "batch", maintain_order=True
    ):
        n = g["n_items"] * shift[f]
        a.plot(
            n,
            1 / g["silvertorch_over_linr"],
            color=BATCH[bs],
            lw=0.8,
            ls="-" if f == "tags4" else ":",
        )
        for outcome, face in (("within_quarter", BATCH[bs]), ("deeper", "none")):
            s = g.filter(pl.col("outcome") == outcome)
            a.plot(
                s["n_items"] * shift[f],
                1 / s["silvertorch_over_linr"],
                MARK[bs],
                ms=4,
                color=BATCH[bs],
                mfc=face,
            )
    a.axhline(1, color=INK, lw=0.8)
    a.set(
        xscale="log",
        yscale="log",
        xlabel="catalogue size (items)",
        ylabel="speed-up over best LiNR",
    )
    a.set_title("(a) LAION real filters, recall 0.90", fontsize=8)
    plain(a, sorted(size.values()), yticks=[0.5, 1, 2, 4, 8])
    h = batch_handles(BATCH, hollow="beyond a quarter of the lists")
    h += [
        Line2D([], [], color=MUTED, ls="-", label="tags"),
        Line2D([], [], color=MUTED, ls=":", label="domain"),
    ]
    a.legend(
        handles=h, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3
    )

    c = read("codesign_pairs").filter(
        pl.col("dataset").str.starts_with("laion") & ~pl.col("timing_unstable")
    )
    fmark = {"c0_domain": "o", "tags4": "s"}
    for (f, bs), g in c.group_by("filter", "batch"):
        b.plot(
            g["n_items"] * shift[f],
            g["whole_index_over_searched_clusters"],
            fmark[f],
            ms=4,
            color=BATCH[bs],
            mfc=BATCH[bs] if f == "tags4" else "none",
        )
    lo = original("co-design latency improvement (low)")
    hi = original("co-design latency improvement (high)")
    n20 = original("catalogue size of the co-design setting")
    b.vlines(n20, lo, hi, color=INK, lw=3)
    b.annotate(
        "SilverTorch\npaper",
        (n20, lo),
        xytext=(0, -4),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6.5,
    )
    b.axhline(1, color=INK, lw=0.8)
    b.set(
        xscale="log",
        yscale="log",
        xlabel="catalogue size (items)",
        ylabel="whole index / searched clusters",
    )
    b.set_title("(b) co-design, Meta's published code", fontsize=8)
    plain(b, sorted(c["n_items"].unique()))
    batches = sorted(c["batch"].unique())
    h = [
        Line2D([], [], color=BATCH[k], marker="s", ls="", label=f"batch {k}")
        for k in batches
    ]
    h += [
        Line2D([], [], color=MUTED, marker="s", ls="", label="tags"),
        Line2D([], [], color=MUTED, marker="o", mfc="none", ls="", label="domain"),
    ]
    b.legend(handles=h, fontsize=6.5, loc="upper left")
    save(fig, "catalogue_size")


def probe_sweep():
    """SilverTorch's recall@100 against latency over its n_probe range on LAION 30M synthetic filters (4,096 lists,
    batch 16); hollow points are flagged unstable. Vertical lines: LiNR V1 (exact) per filter, same run."""
    x = read("probe_sweep").filter(
        (pl.col("n_lists") == SWEEP_LISTS) & pl.col("filter").is_in(SWEEP_FILTERS)
    )
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, adjacent lines
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    h = []
    for f, col in zip(SWEEP_FILTERS, colors):
        g = x.filter(pl.col("filter") == f).sort("n_probe")
        label = "no filter" if f == "p1" else pct(round(g["pass_rate"][0], 3))
        ax.plot(g["silvertorch_ms"], g["silvertorch_recall"], color=col, lw=1)
        for u in (False, True):
            s = g.filter(pl.col("timing_unstable") == u)
            ax.plot(
                s["silvertorch_ms"],
                s["silvertorch_recall"],
                "o",
                ms=3,
                color=col,
                mfc="none" if u else col,
            )
        h.append(Line2D([], [], color=col, marker="o", ms=3, label=label))
    v1 = x["v1_ms"]
    ax.axvline(v1.median(), color=INK, lw=0.8, ls="--")
    h.append(
        Line2D(
            [],
            [],
            color=INK,
            ls="--",
            label=f"LiNR V1 (exact), {v1.min():.1f}-{v1.max():.1f} ms",
        )
    )
    h.append(
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            ms=3,
            mfc="none",
            ls="",
            label="flagged unstable",
        )
    )
    ax.set(xscale="log", xlabel="latency per batch of 16 (ms)", ylabel="recall@100")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(handles=h, fontsize=7, loc="lower right")
    save(fig, "probe_sweep")


if __name__ == "__main__":
    for draw in (linr_v2_over_v1, head_to_head, catalogue_size, probe_sweep):
        draw()
        print(draw.__name__)
