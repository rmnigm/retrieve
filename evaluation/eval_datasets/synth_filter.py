"""``eval-data synth-filter``: the synthetic selectivity attrs (docs/system/datasets.md
§ Synthetic selectivity attrs). Uniform: one ``u_i ~ U(0, 1)`` per catalog item, in item-row
order; item ``i`` passes the clause of rate ``p`` iff ``u_i < p``, so pass sets nest across
rates. ``--correlated``: per rate a coarse spherical k-means with ``round(1/p)`` centroids; the
item's value is its cluster, the query's its nearest centroid (by inner product, the k-means's
own rule), so the pass set sits around the query.
Written beside the dataset's real attrs, never over them: ``item_attrs_narrow.pt`` keys every
existing oracle and record through its digest."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click
import polars as pl
import torch
import yaml
from loguru import logger

from eval_datasets import layout
from eval_datasets.etl.synth_arxiv import spherical_kmeans
from eval_datasets.layout import atomic_write, drop_legacy_padding_row

RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CORR_RATES = (0.01, 0.03, 0.1)
CORR_ITERS = 25
CORR_ITEM_ATTRS, CORR_QUERY_ATTRS, CORR_SIDECAR = (
    "item_attrs_corr.pt", "query_attrs_corr.pt", "synth_corr.json"
)  # fmt: skip
SEED = 20261008
ITEM_ATTRS, U_FILE, QUERY_ATTRS, SIDECAR = (
    "item_attrs_synth.pt", "synth_u.pt", "query_attrs_synth.pt", "synth_filter.json"
)  # fmt: skip
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def synth_u(n: int, seed: int = SEED) -> torch.Tensor:
    """``[n]`` float64 uniforms from ``torch.Generator().manual_seed(seed)``."""
    return torch.rand(n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def synth_flags(u: torch.Tensor, rates: Sequence[float]) -> torch.Tensor:
    """``[N, len(rates), 1]`` int64: 1 = pass (``u_i < p``), 0 = fail."""
    return (u[:, None] < torch.tensor(rates, dtype=torch.float64)).long().unsqueeze(-1)


def build(data_dir: Path, rates: Sequence[float] = RATES, seed: int = SEED) -> dict:
    """Write the four synth files into ``data_dir``; N is the real attrs' row count with the
    legacy pad row dropped (``layout.load_item_attrs``'s N), U the full ``eval_split`` rows."""
    data_dir = Path(data_dir)
    real = torch.load(str(data_dir / "item_attrs_narrow.pt"), weights_only=True, mmap=True)
    n = int(drop_legacy_padding_row(real, kind="attrs", what="item_attrs_narrow.pt").shape[0])
    del real
    n_queries = int(
        pl.scan_parquet(data_dir / "eval_split.parquet").select(pl.len()).collect().item()
    )
    u = synth_u(n, seed)
    flags = synth_flags(u, rates)
    counts = flags[:, :, 0].sum(dim=0).tolist()
    for name, t in (
        (U_FILE, u),
        (ITEM_ATTRS, flags),
        (QUERY_ATTRS, torch.ones(n_queries, len(rates), dtype=torch.long)),
    ):
        atomic_write(data_dir / name, lambda fh, t=t: torch.save(t, fh))
    meta = {
        "rates": list(rates),
        "seed": seed,
        "n_items": n,
        "n_queries": n_queries,
        "pass_counts": counts,
        "pass_rates": [c / n for c in counts],
    }
    (data_dir / SIDECAR).write_text(json.dumps(meta, indent=1) + "\n")
    logger.info("{}: N={} U={} pass counts {}", data_dir, n, n_queries, counts)
    return meta


def _pass_stats(x: torch.Tensor) -> dict:
    x = x.double()
    lo, med, hi = torch.quantile(x, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)).tolist()
    return {"mean": x.mean().item(), "std": x.std().item(), "min": lo, "median": med, "max": hi}


def build_correlated(
    data_dir: Path, content_dir: Path, rates: Sequence[float] = CORR_RATES, seed: int = SEED
) -> dict:
    """Write the three correlated files into ``data_dir`` from the item and query embeddings
    the harness loads (``layout.load_text_items`` / ``load_text_queries`` of ``content_dir``);
    a query's pass rate is its cluster's size over N."""
    data_dir, content_dir = Path(data_dir), Path(content_dir)
    items = layout.load_text_items(content_dir, torch.device("cpu"))
    real = torch.load(str(data_dir / "item_attrs_narrow.pt"), weights_only=True, mmap=True)
    real = drop_legacy_padding_row(real, kind="attrs", what="item_attrs_narrow.pt")
    layout.check_items_aligned(int(items.shape[0]), int(real.shape[0]), what="item_attrs_narrow.pt")
    del real
    queries, _, _ = layout.load_text_queries(data_dir, content_dir, int(items.shape[1]))
    n = int(items.shape[0])
    item_vals, query_vals, per_rate = [], [], []
    for p in rates:
        k = round(1 / p)
        centroids, _ = spherical_kmeans(
            items, k, n_iter=CORR_ITERS, seed=seed, device=torch.device("cpu")
        )
        # Re-assigned against the final centroids: the k-means's own assignment predates its
        # last update, and items and queries must take the same rule.
        assign = torch.cat([(c @ centroids.T).argmax(dim=1) for c in items.split(1 << 20)])
        qv = (queries @ centroids.T).argmax(dim=1)
        sizes = torch.bincount(assign, minlength=k)
        item_vals.append(assign)
        query_vals.append(qv)
        per_rate.append(
            {
                "rate": p,
                "n_lists": k,
                "query_pass_rate": _pass_stats(sizes[qv].double() / n),
                "cluster_size": _pass_stats(sizes),
                "centroids": centroids.tolist(),
            }
        )
        logger.info("corr p={}: k={} query pass rate {}", p, k, per_rate[-1]["query_pass_rate"])
    for name, t in (
        (CORR_ITEM_ATTRS, torch.stack(item_vals, dim=1).unsqueeze(-1)),
        (CORR_QUERY_ATTRS, torch.stack(query_vals, dim=1)),
    ):
        atomic_write(data_dir / name, lambda fh, t=t: torch.save(t, fh))
    meta = {
        "rates": list(rates),
        "seed": seed,
        "kmeans": {"kind": "spherical", "max_iter": CORR_ITERS},
        "content_dir": content_dir.name,
        "n_items": n,
        "n_queries": int(queries.shape[0]),
        "per_rate": per_rate,
    }
    (data_dir / CORR_SIDECAR).write_text(json.dumps(meta) + "\n")
    return meta


@click.command("synth-filter")
@click.option("--dataset", required=True, help="config/<dataset>.yaml names the data_dir")
@click.option("--rates", help=f"comma-separated; default {RATES}, --correlated {CORR_RATES}")
@click.option("--seed", default=SEED, show_default=True, type=int)
@click.option("--correlated", is_flag=True, help="cluster-correlated attrs (text datasets)")
@click.option("--dim", default=128, show_default=True, help="--correlated: content_dir dim")
@click.option("--config-dir", default=CONFIG_DIR, type=click.Path(path_type=Path))
def synth_filter(
    dataset: str, rates: str | None, seed: int, correlated: bool, dim: int, config_dir: Path
) -> None:
    """Write item_attrs_synth.pt, synth_u.pt, query_attrs_synth.pt and synth_filter.json, or
    with --correlated item_attrs_corr.pt, query_attrs_corr.pt and synth_corr.json."""
    raw = yaml.safe_load((config_dir / f"{dataset}.yaml").read_text())
    data_dir = Path(raw["data_dir"])
    if not data_dir.is_absolute():
        data_dir = config_dir.parent / data_dir
    rs = [float(r) for r in rates.split(",")] if rates else None
    if not correlated:
        click.echo(json.dumps(build(data_dir, rs or RATES, seed)))
        return
    content = raw.get("content_dir")
    if content is None:
        raise click.BadParameter(f"{dataset} has no content_dir", param_hint="--correlated")
    content = content[dim] if isinstance(content, dict) else content
    meta = build_correlated(data_dir, data_dir / content, rs or CORR_RATES, seed)
    click.echo(json.dumps({k: v for k, v in meta.items() if k != "per_rate"}))
    for r in meta["per_rate"]:
        click.echo(json.dumps({k: v for k, v in r.items() if k != "centroids"}))


__all__ = [
    "CORR_RATES",
    "RATES",
    "SEED",
    "build",
    "build_correlated",
    "synth_filter",
    "synth_flags",
    "synth_u",
]
