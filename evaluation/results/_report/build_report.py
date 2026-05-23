"""Aggregate eval JSONs into a Markdown report + PNG plots.

Reads the 6 aggregated filter-bench bundles in
    evaluation/results/{arxiv,goodreads}/<dim>-filter.json
and writes REPORT.md + PNGs to the script's directory. Per-sweep selectivity
(from selectivity.json) drives a high/mid/low tier breakdown of the tradeoff
tables, so iso-recall winners aren't averaged across very different filter
densities.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SELECTIVITY_PATH = OUT / "selectivity.json"

DATASETS = {
    "arxiv":     {64: ROOT / "arxiv/d64-filter.json",     128: ROOT / "arxiv/d128-filter.json",     256: ROOT / "arxiv/d256-filter.json"},
    "goodreads": {64: ROOT / "goodreads/d64-filter.json", 128: ROOT / "goodreads/d128-filter.json", 256: ROOT / "goodreads/d256-filter.json"},
}
DIMS = [64, 128, 256]
FILTER_KINDS = ["clause", "bloom"]
TIERS = ["high", "mid", "low"]
TIER_LABEL = {"high": "high sel. (strict, <5% kept)", "mid": "mid sel. (5–25% kept)", "low": "low sel. (loose, ≥25% kept)"}

IMPL_NAMES = {
    "linr_v1_filter_mask": "linr_v1",
    "linr_v2": "linr_v2",
    "linr_v3": "linr_v3",
    "linr_v4": "linr_v4",
    "silvertorch": "silvertorch",
}
IMPL_ORDER = ["linr_v1", "linr_v2", "linr_v3", "linr_v4", "silvertorch"]
COLORS = {
    "linr_v1":     "#1f77b4",
    "linr_v2":     "#2ca02c",
    "linr_v3":     "#d62728",
    "linr_v4":     "#8c564b",
    "silvertorch": "#9467bd",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_selectivity() -> dict[str, dict[str, dict]]:
    return json.loads(SELECTIVITY_PATH.read_text())


def load_dataset(name: str, selectivity: dict[str, dict[str, dict]]) -> pl.DataFrame:
    sel_map = selectivity[name]
    rows: list[dict] = []
    for dim, path in DATASETS[name].items():
        for r in json.loads(path.read_text()):
            r["dim"] = dim
            r["algo"] = IMPL_NAMES.get(r["impl"], r["impl"])
            r["dataset"] = name
            r["selectivity"] = sel_map.get(r["sweep"], {}).get("selectivity")
            r["tier"] = sel_map.get(r["sweep"], {}).get("tier")
            rows.append(r)
    df = pl.DataFrame(rows, infer_schema_length=None, strict=False)
    df = df.with_columns(_recall_col(df).alias("recall"), _ndcg_col(df).alias("ndcg"))
    return df


def _recall_col(df: pl.DataFrame) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Float64)
    for k in sorted(df["k"].unique().to_list()):
        col = f"recall@{k}"
        if col in df.columns:
            expr = pl.when(pl.col("k") == k).then(pl.col(col)).otherwise(expr)
    return expr


def _ndcg_col(df: pl.DataFrame) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.Float64)
    for k in sorted(df["k"].unique().to_list()):
        col = f"ndcg@{k}"
        if col in df.columns:
            expr = pl.when(pl.col("k") == k).then(pl.col(col)).otherwise(expr)
    return expr


# ---------------------------------------------------------------------------
# Pareto + tables
# ---------------------------------------------------------------------------

def pareto_mask(x: np.ndarray, y: np.ndarray, *, x_lower_better: bool, y_higher_better: bool) -> np.ndarray:
    n = len(x)
    keep = np.ones(n, dtype=bool)
    xs = x if x_lower_better else -x
    ys = y if y_higher_better else -y
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]):
                keep[i] = False
                break
    return keep


def md_table(df: pl.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = r[c]
            if v is None:
                cells.append("n/a")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def agg_tier(df: pl.DataFrame, dim: int, tier: str) -> pl.DataFrame:
    """Mean per (filter_kind, algo) within one selectivity tier, at bs=8 k=100."""
    sub = df.filter(
        (pl.col("dim") == dim) & (pl.col("batch_size") == 8) & (pl.col("k") == 100) & (pl.col("tier") == tier)
    )
    if sub.height == 0:
        return sub
    agg = (
        sub.group_by(["filter_kind", "algo"])
        .agg(
            pl.col("recall").mean().alias("recall"),
            pl.col("ndcg").mean().alias("ndcg"),
            pl.col("median_ms").mean().alias("p50_ms"),
            pl.col("peak_mem_mib").mean().alias("peak_MiB"),
            pl.col("index_mem_mib").mean().alias("idx_MiB"),
        )
    )
    # Pareto on (p50 ↓, recall ↑) within each filter_kind
    pareto_col: list[str] = []
    for r in agg.iter_rows(named=True):
        pareto_col.append("")
    agg = agg.with_columns(pl.Series("pareto", pareto_col))
    for fk in agg["filter_kind"].unique().to_list():
        sub_fk = agg.filter(pl.col("filter_kind") == fk)
        x = sub_fk["p50_ms"].to_numpy()
        y = sub_fk["recall"].to_numpy()
        keep = pareto_mask(x, y, x_lower_better=True, y_higher_better=True)
        marks = np.where(keep, "★", "")
        idx = agg.with_row_index("__i").filter(pl.col("filter_kind") == fk)["__i"].to_numpy()
        new_pareto = list(agg["pareto"])
        for j, i in enumerate(idx):
            new_pareto[int(i)] = str(marks[j])
        agg = agg.with_columns(pl.Series("pareto", new_pareto))
    fk_ord = {"clause": 0, "bloom": 1}
    agg = (
        agg.with_columns(
            pl.col("filter_kind").map_elements(lambda v: fk_ord.get(v, 99), return_dtype=pl.Int8).alias("__fk_ord"),
        )
        .sort(["__fk_ord", "p50_ms"])
        .drop("__fk_ord")
        .rename({"filter_kind": "filter"})
    )
    return agg


def per_sweep_table(df: pl.DataFrame, dim: int) -> pl.DataFrame:
    """One row per (filter_kind, sweep, algo) at bs=8 k=100, sorted by tier then selectivity."""
    sub = df.filter(
        (pl.col("dim") == dim) & (pl.col("batch_size") == 8) & (pl.col("k") == 100)
    )
    if sub.height == 0:
        return sub
    agg = sub.group_by(["filter_kind", "sweep", "tier", "selectivity", "algo"]).agg(
        pl.col("recall").mean().alias("recall"),
        pl.col("median_ms").mean().alias("p50_ms"),
        pl.col("peak_mem_mib").mean().alias("peak_MiB"),
    )
    tier_ord = {"high": 0, "mid": 1, "low": 2}
    algo_ord = {a: i for i, a in enumerate(IMPL_ORDER)}
    fk_ord = {"clause": 0, "bloom": 1}
    agg = (
        agg.with_columns(
            pl.col("tier").map_elements(lambda v: tier_ord.get(v, 99), return_dtype=pl.Int8).alias("__t"),
            pl.col("filter_kind").map_elements(lambda v: fk_ord.get(v, 99), return_dtype=pl.Int8).alias("__f"),
            pl.col("algo").map_elements(lambda v: algo_ord.get(v, 99), return_dtype=pl.Int8).alias("__a"),
            (pl.col("selectivity") * 100).round(2).alias("sel_pct"),
        )
        .sort(["__f", "__t", "selectivity", "__a"])
        .drop(["__t", "__f", "__a"])
        .rename({"filter_kind": "filter"})
    )
    return agg


def tradeoff_callout(agg: pl.DataFrame) -> str:
    if agg.height == 0:
        return "_no rows_"
    lines = []
    for fk in ["clause", "bloom"]:
        s = agg.filter(pl.col("filter") == fk)
        if s.height == 0:
            continue
        rows = s.to_dicts()
        best_lat = min(rows, key=lambda r: r["p50_ms"])
        best_recall = max(rows, key=lambda r: r["recall"])
        best_mem = min(rows, key=lambda r: r["peak_MiB"])
        pareto = ", ".join(sorted(r["algo"] for r in rows if r["pareto"] == "★"))
        lines.append(
            f"- **{fk}** — fastest: `{best_lat['algo']}` ({best_lat['p50_ms']:.2f} ms); "
            f"best recall: `{best_recall['algo']}` ({best_recall['recall']:.4f}); "
            f"lowest peak mem: `{best_mem['algo']}` ({best_mem['peak_MiB']:.1f} MiB); "
            f"Pareto: {pareto or '—'}."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_panel(ax, sub: pl.DataFrame, x_col: str, y_col: str, *, x_lower_better: bool, y_higher_better: bool) -> None:
    s = sub.filter((pl.col("batch_size") == 8) & (pl.col("k") == 100))
    for algo in IMPL_ORDER:
        ss = s.filter(pl.col("algo") == algo)
        if ss.height == 0:
            continue
        ax.scatter(ss[x_col].to_numpy(), ss[y_col].to_numpy(),
                   s=42, alpha=0.7, color=COLORS[algo],
                   edgecolor="white", linewidth=0.3, label=algo)
    means = s.group_by("algo").agg(pl.col(x_col).mean().alias("x"), pl.col(y_col).mean().alias("y"))
    if means.height > 0:
        xs = means["x"].to_numpy()
        ys = means["y"].to_numpy()
        keep = pareto_mask(xs, ys, x_lower_better=x_lower_better, y_higher_better=y_higher_better)
        for j in np.where(keep)[0]:
            ax.scatter(xs[j], ys[j], s=180, facecolor="none",
                       edgecolor="black", linewidth=1.4, zorder=5)
    ax.grid(True, alpha=0.3)


def plot_recall_vs_latency(df: pl.DataFrame, dataset: str, path: Path) -> None:
    fig, axes = plt.subplots(len(FILTER_KINDS), len(DIMS), figsize=(15, 8), sharey=True)
    for r, fk in enumerate(FILTER_KINDS):
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            sub = df.filter((pl.col("dim") == dim) & (pl.col("filter_kind") == fk))
            _scatter_panel(ax, sub, "median_ms", "recall", x_lower_better=True, y_higher_better=True)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(f"d={dim}")
            if r == len(FILTER_KINDS) - 1:
                ax.set_xlabel("p50 latency, ms (log)")
            if c == 0:
                ax.set_ylabel(f"{fk}\nrecall@100")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — recall vs latency (per sweep; ◯ = Pareto-optimal algo mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_recall_vs_memory(df: pl.DataFrame, dataset: str, path: Path) -> None:
    fig, axes = plt.subplots(len(FILTER_KINDS), len(DIMS), figsize=(15, 8), sharey=True)
    for r, fk in enumerate(FILTER_KINDS):
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            sub = df.filter((pl.col("dim") == dim) & (pl.col("filter_kind") == fk))
            _scatter_panel(ax, sub, "peak_mem_mib", "recall", x_lower_better=True, y_higher_better=True)
            if r == 0:
                ax.set_title(f"d={dim}")
            if r == len(FILTER_KINDS) - 1:
                ax.set_xlabel("peak GPU mem, MiB")
            if c == 0:
                ax.set_ylabel(f"{fk}\nrecall@100")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — recall vs peak memory")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_latency_vs_memory(df: pl.DataFrame, dataset: str, path: Path) -> None:
    fig, axes = plt.subplots(len(FILTER_KINDS), len(DIMS), figsize=(15, 8), sharey=True)
    for r, fk in enumerate(FILTER_KINDS):
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            sub = df.filter(
                (pl.col("dim") == dim) & (pl.col("filter_kind") == fk) &
                (pl.col("batch_size") == 8) & (pl.col("k") == 100)
            )
            for algo in IMPL_ORDER:
                ss = sub.filter(pl.col("algo") == algo)
                if ss.height == 0:
                    continue
                sizes = 20 + 200 * np.clip(ss["recall"].to_numpy(), 0, 1)
                ax.scatter(ss["peak_mem_mib"].to_numpy(), ss["median_ms"].to_numpy(),
                           s=sizes, alpha=0.6, color=COLORS[algo],
                           edgecolor="white", linewidth=0.3, label=algo)
            means = sub.group_by("algo").agg(
                pl.col("peak_mem_mib").mean().alias("x"),
                pl.col("median_ms").mean().alias("y"),
            )
            if means.height > 0:
                xs = means["x"].to_numpy()
                ys = means["y"].to_numpy()
                keep = pareto_mask(xs, ys, x_lower_better=True, y_higher_better=False)
                for j in np.where(keep)[0]:
                    ax.scatter(xs[j], ys[j], s=240, facecolor="none",
                               edgecolor="black", linewidth=1.4, zorder=5)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(f"d={dim}")
            if r == len(FILTER_KINDS) - 1:
                ax.set_xlabel("peak GPU mem, MiB")
            if c == 0:
                ax.set_ylabel(f"{fk}\np50 latency, ms (log)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — latency vs memory (marker size ∝ recall; ◯ = Pareto mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_sweep_latency(df: pl.DataFrame, dataset: str, path: Path, selectivity: dict[str, dict[str, dict]]) -> None:
    """Grouped bar (algo) per sweep, sweeps ordered by selectivity ascending so high-sel sweeps
    appear on the left. Tier boundaries marked with vertical lines."""
    sel_map = selectivity[dataset]
    fig, axes = plt.subplots(len(FILTER_KINDS), len(DIMS), figsize=(16, 8.5))
    for r, fk in enumerate(FILTER_KINDS):
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            sub = df.filter(
                (pl.col("dim") == dim) & (pl.col("filter_kind") == fk) &
                (pl.col("batch_size") == 8) & (pl.col("k") == 100)
            )
            sweeps_in = sub["sweep"].unique().to_list()
            sweeps = sorted(sweeps_in, key=lambda s: sel_map.get(s, {}).get("selectivity", 1.0))
            tiers = [sel_map.get(s, {}).get("tier", "?") for s in sweeps]
            algos = [a for a in IMPL_ORDER if a in set(sub["algo"].unique().to_list())]
            x = np.arange(len(sweeps))
            width = 0.8 / max(len(algos), 1)
            for i, algo in enumerate(algos):
                vals = []
                for sw in sweeps:
                    sl = sub.filter((pl.col("sweep") == sw) & (pl.col("algo") == algo))["median_ms"].mean()
                    vals.append(sl if sl is not None else 0)
                ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, color=COLORS[algo], label=algo)
            ax.set_xticks(x)
            labels = [f"{sw}\n({sel_map.get(sw, {}).get('selectivity', 0)*100:.1f}%·{tiers[k][:1].upper()})" for k, sw in enumerate(sweeps)]
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", alpha=0.3)
            for k in range(1, len(sweeps)):
                if tiers[k] != tiers[k - 1]:
                    ax.axvline(k - 0.5, color="black", alpha=0.25, linewidth=0.8, linestyle="--")
            if r == 0:
                ax.set_title(f"d={dim}")
            if c == 0:
                ax.set_ylabel(f"{fk}\np50 latency, ms (log)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — per-sweep latency, sweeps ordered by selectivity (low→high % kept); dashed = tier boundary")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_perf_by_tier(df: pl.DataFrame, dataset: str, path: Path) -> None:
    """Recall vs latency, faceted as 2 (filter) × 3 (tier) at d=128. Lets you see how
    algo ranking flips as filter density changes."""
    fig, axes = plt.subplots(len(FILTER_KINDS), len(TIERS), figsize=(15, 8), sharey=True)
    for r, fk in enumerate(FILTER_KINDS):
        for c, tier in enumerate(TIERS):
            ax = axes[r, c]
            sub = df.filter(
                (pl.col("dim") == 128) & (pl.col("filter_kind") == fk) & (pl.col("tier") == tier) &
                (pl.col("batch_size") == 8) & (pl.col("k") == 100)
            )
            for algo in IMPL_ORDER:
                ss = sub.filter(pl.col("algo") == algo)
                if ss.height == 0:
                    continue
                ax.scatter(ss["median_ms"].to_numpy(), ss["recall"].to_numpy(),
                           s=55, alpha=0.75, color=COLORS[algo],
                           edgecolor="white", linewidth=0.3, label=algo)
            means = sub.group_by("algo").agg(
                pl.col("median_ms").mean().alias("x"),
                pl.col("recall").mean().alias("y"),
            )
            if means.height > 0:
                xs = means["x"].to_numpy()
                ys = means["y"].to_numpy()
                keep = pareto_mask(xs, ys, x_lower_better=True, y_higher_better=True)
                for j in np.where(keep)[0]:
                    ax.scatter(xs[j], ys[j], s=200, facecolor="none",
                               edgecolor="black", linewidth=1.4, zorder=5)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(TIER_LABEL[tier])
            if r == len(FILTER_KINDS) - 1:
                ax.set_xlabel("p50 latency, ms (log)")
            if c == 0:
                ax.set_ylabel(f"{fk}\nrecall@100")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} d=128 — recall vs latency, faceted by selectivity tier")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dim_scaling(dfs: dict[str, pl.DataFrame], metric: str, ylabel: str, suptitle: str, path: Path, *, log_y: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for r, dataset in enumerate(["arxiv", "goodreads"]):
        for c, fk in enumerate(FILTER_KINDS):
            ax = axes[r, c]
            sub = dfs[dataset].filter(
                (pl.col("filter_kind") == fk) & (pl.col("batch_size") == 8) & (pl.col("k") == 100)
            )
            for algo in IMPL_ORDER:
                ss = sub.filter(pl.col("algo") == algo)
                if ss.height == 0:
                    continue
                means = ss.group_by("dim").agg(pl.col(metric).mean().alias("v")).sort("dim")
                ax.plot(means["dim"].to_numpy(), means["v"].to_numpy(),
                        "-o", color=COLORS[algo], label=algo, linewidth=1.6, markersize=6)
            ax.set_xticks(DIMS)
            ax.grid(True, alpha=0.3)
            if log_y:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{fk}")
            if r == 1:
                ax.set_xlabel("embedding dim")
            if c == 0:
                ax.set_ylabel(f"{dataset}\n{ylabel}")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(IMPL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def selectivity_table(selectivity: dict[str, dict[str, dict]], dataset: str) -> str:
    rows = []
    for sw, info in sorted(selectivity[dataset].items(), key=lambda kv: kv[1]["selectivity"]):
        rows.append({"sweep": sw, "selectivity_%": round(info["selectivity"] * 100, 3),
                     "n_clauses": info["n_clauses"], "tier": info["tier"]})
    return md_table(pl.DataFrame(rows), ["sweep", "selectivity_%", "n_clauses", "tier"])


def build_dataset_section(dataset: str, df: pl.DataFrame, selectivity: dict[str, dict[str, dict]]) -> str:
    parts = [f"## {dataset}", ""]
    parts.append("### Selectivity per sweep")
    parts.append("")
    parts.append(selectivity_table(selectivity, dataset))
    parts.append("")
    parts.append(f"![recall vs latency]({dataset}_recall_vs_latency.png)")
    parts.append(f"![recall vs memory]({dataset}_recall_vs_memory.png)")
    parts.append(f"![latency vs memory]({dataset}_latency_vs_memory.png)")
    parts.append(f"![tier facet (d=128)]({dataset}_perf_by_tier_d128.png)")
    parts.append(f"![per-sweep latency]({dataset}_per_sweep_latency.png)")
    parts.append("")
    for dim in DIMS:
        parts.append(f"### {dataset} · d={dim} — by selectivity tier")
        parts.append("")
        for tier in TIERS:
            agg = agg_tier(df, dim, tier)
            if agg.height == 0:
                continue
            sweeps = sorted(
                df.filter((pl.col("dim") == dim) & (pl.col("tier") == tier))["sweep"].unique().to_list(),
                key=lambda s: selectivity[dataset].get(s, {}).get("selectivity", 1.0),
            )
            sweep_blurb = ", ".join(f"`{s}` ({selectivity[dataset][s]['selectivity']*100:.1f}%)" for s in sweeps)
            parts.append(f"#### {TIER_LABEL[tier]} — sweeps: {sweep_blurb}")
            parts.append("")
            parts.append(tradeoff_callout(agg))
            parts.append("")
            parts.append(md_table(agg, ["filter", "algo", "recall", "ndcg", "p50_ms", "peak_MiB", "idx_MiB", "pareto"]))
            parts.append("")
        # full per-sweep dump (collapsed-friendly)
        ps = per_sweep_table(df, dim)
        if ps.height > 0:
            parts.append("<details><summary>Per-sweep detail (bs=8, k=100)</summary>")
            parts.append("")
            parts.append(md_table(ps, ["filter", "sweep", "tier", "sel_pct", "algo", "recall", "p50_ms", "peak_MiB"]))
            parts.append("")
            parts.append("</details>")
            parts.append("")
    return "\n".join(parts)


def main() -> None:
    selectivity = load_selectivity()
    dfs = {name: load_dataset(name, selectivity) for name in DATASETS}

    for name in DATASETS:
        plot_recall_vs_latency(dfs[name], name, OUT / f"{name}_recall_vs_latency.png")
        plot_recall_vs_memory(dfs[name], name, OUT / f"{name}_recall_vs_memory.png")
        plot_latency_vs_memory(dfs[name], name, OUT / f"{name}_latency_vs_memory.png")
        plot_per_sweep_latency(dfs[name], name, OUT / f"{name}_per_sweep_latency.png", selectivity)
        plot_perf_by_tier(dfs[name], name, OUT / f"{name}_perf_by_tier_d128.png")
    plot_dim_scaling(dfs, "median_ms", "p50 ms (log)", "Dim scaling — p50 latency (bs=8, k=100, mean over sweeps)",
                     OUT / "dim_scaling_latency.png", log_y=True)
    plot_dim_scaling(dfs, "peak_mem_mib", "peak MiB", "Dim scaling — peak GPU memory (bs=8, k=100, mean over sweeps)",
                     OUT / "dim_scaling_memory.png", log_y=False)

    sections = [
        "# Retrieve eval — filter benches (2026-05-23 snapshot)",
        "",
        "## Summary",
        "",
        "- Sources: 6 aggregated bundles in `results/{arxiv,goodreads}/d{64,128,256}-filter.json`.",
        "- Algos: " + ", ".join(f"`{a}`" for a in IMPL_ORDER) + ".",
        "- Filters: `clause`, `bloom`; sweeps cover a wide selectivity range (see per-dataset tables).",
        "- Tables aggregate over filter sweeps **within a selectivity tier** at bs=8, k=100. ★ = Pareto-optimal on (p50 latency ↓, recall ↑) within filter_kind.",
        "- Selectivity tiers (% items kept by predicate): **high** = strict (<5%), **mid** = 5–25%, **low** = loose (≥25%). Computed empirically from `item_attrs_narrow.pt` (see `compute_selectivity.py`).",
        "",
        "## Cross-dim scaling",
        "",
        "![dim scaling latency](dim_scaling_latency.png)",
        "",
        "![dim scaling memory](dim_scaling_memory.png)",
        "",
    ]
    for ds in ["arxiv", "goodreads"]:
        sections.append(build_dataset_section(ds, dfs[ds], selectivity))

    txt = "\n".join(sections)
    (OUT / "REPORT.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
