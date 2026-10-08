"""``eval-data synth-filter``: the synthetic selectivity attrs (docs/system/datasets.md
§ Synthetic selectivity attrs). One ``u_i ~ U(0, 1)`` per catalog item, in item-row order;
item ``i`` passes the clause of rate ``p`` iff ``u_i < p``, so pass sets nest across rates.
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

from eval_datasets.layout import atomic_write, drop_legacy_padding_row

RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
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


@click.command("synth-filter")
@click.option("--dataset", required=True, help="config/<dataset>.yaml names the data_dir")
@click.option("--rates", default=",".join(map(str, RATES)), show_default=True)
@click.option("--seed", default=SEED, show_default=True, type=int)
@click.option("--config-dir", default=CONFIG_DIR, type=click.Path(path_type=Path))
def synth_filter(dataset: str, rates: str, seed: int, config_dir: Path) -> None:
    """Write item_attrs_synth.pt, synth_u.pt, query_attrs_synth.pt and synth_filter.json."""
    data_dir = Path(yaml.safe_load((config_dir / f"{dataset}.yaml").read_text())["data_dir"])
    if not data_dir.is_absolute():
        data_dir = config_dir.parent / data_dir
    meta = build(data_dir, [float(r) for r in rates.split(",")], seed)
    click.echo(json.dumps(meta))


__all__ = ["RATES", "SEED", "build", "synth_filter", "synth_flags", "synth_u"]
