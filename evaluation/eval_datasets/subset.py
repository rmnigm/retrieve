"""``eval-data subset``: a seeded row subset of a staged dataset (docs/system/datasets.md
§ Subsets), for the fixed-d N-sweep (LAION 30 M d256 → 1 / 3 / 10 M).

The rows are the sorted first ``n`` of one ``randperm(N_parent)`` drawn from
``torch.Generator().manual_seed(seed)``, so the subsets of one seed nest (1 M ⊂ 3 M ⊂ 10 M).
Item vectors, real attrs and synth attrs are the parent's rows byte for byte (synth: each item
keeps its ``u_i``, so its pass / fail at every rate). Queries, query attrs, the reverse flags and
the vocab are the parent's. The held-out targets are recomputed over the subset by ingest's rule
(exact unfiltered top-1 by inner product, fp32, ties to the lowest id). ``MANIFEST.sha256``
lists every written file."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import click
import polars as pl
import torch
import yaml

from eval_datasets import layout
from eval_datasets.common import file_hexdigest, merge_prep_log
from eval_datasets.hub import data_root
from eval_datasets.ingest import nearest_items
from eval_datasets.layout import atomic_write, load_sharded
from eval_datasets.synth_filter import ITEM_ATTRS, QUERY_ATTRS, SIDECAR, U_FILE

SEED = 20261013
COPIED = ("clause_is_reverse_narrow.pt", "vocab.json", QUERY_ATTRS)
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def subset_rows(n_parent: int, n: int, seed: int = SEED) -> torch.Tensor:
    """``[n]`` sorted parent rows: the first ``n`` of one seeded permutation."""
    perm = torch.randperm(n_parent, generator=torch.Generator().manual_seed(seed))
    return perm[:n].sort().values


def _save(path: Path, t: torch.Tensor) -> None:
    atomic_write(path, lambda fh: torch.save(t.contiguous().clone(), fh))


def _load(path: Path) -> torch.Tensor:
    return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)


def build(parent: Path, out: Path, n: int, seed: int = SEED, shard_rows: int = 1_000_000) -> dict:
    parent, out = Path(parent), Path(out)
    (content_p,) = [p for p in parent.iterdir() if (p / "shard_index.json").exists()]
    idx = json.loads((content_p / "shard_index.json").read_text())
    n_parent, d = int(idx["n_items"]), int(idx["dim"])
    if not 0 < n < n_parent:
        raise ValueError(f"n {n} must be in (0, {n_parent})")
    rows = subset_rows(n_parent, n, seed)
    content = out / content_p.name
    content.mkdir(parents=True, exist_ok=True)

    # vectors: the parent's fp16 rows, gathered shard by shard, re-sharded at shard_rows
    parts = []
    for s in idx["shards"]:
        start, cnt = int(s["start_id"]), int(s["n_rows"])
        lo, hi = torch.searchsorted(rows, torch.tensor([start, start + cnt])).tolist()
        if hi > lo:
            parts.append(_load(content_p / s["filename"])[rows[lo:hi] - start])
    emb = torch.cat(parts)
    del parts
    entries = []
    for k, start in enumerate(range(0, n, shard_rows)):
        fname = f"text_emb_shard_{k:03d}.pt"
        _save(content / fname, emb[start : start + shard_rows])
        entries.append({"filename": fname, "start_id": start, "n_rows": min(shard_rows, n - start)})
    del emb
    index = {"n_items": n, "dim": d, "dtype": idx["dtype"], "n_shards": len(entries),
             "shards": entries}  # fmt: skip
    (content / "shard_index.json").write_text(json.dumps(index, indent=2))
    meta = json.loads((content_p / "text_emb.meta.json").read_text())
    meta |= {"n_items": n, "subset_of": parent.name, "subset_seed": seed}
    (content / "text_emb.meta.json").write_text(json.dumps(meta))
    for f in ("query_emb.pt", "query_emb.meta.json"):
        shutil.copyfile(content_p / f, content / f)

    # attrs: real and synth item rows; everything query-side is the parent's
    real = _load(parent / "item_attrs_narrow.pt")
    if real.shape[0] != n_parent:
        raise ValueError(f"item_attrs_narrow.pt has {real.shape[0]} rows, not {n_parent}")
    _save(out / "item_attrs_narrow.pt", real[rows])
    del real
    for f in COPIED:
        if (parent / f).exists():
            shutil.copyfile(parent / f, out / f)
    synth = (parent / ITEM_ATTRS).exists()
    if synth:
        _save(out / ITEM_ATTRS, _load(parent / ITEM_ATTRS)[rows])
        u = _load(parent / U_FILE)[rows]
        _save(out / U_FILE, u)
        side = json.loads((parent / SIDECAR).read_text())
        counts = [int((u < r).sum()) for r in side["rates"]]
        side |= {"n_items": n, "pass_counts": counts, "pass_rates": [c / n for c in counts],
                 "subset_of": parent.name, "subset_seed": seed}  # fmt: skip
        (out / SIDECAR).write_text(json.dumps(side, indent=1) + "\n")

    # targets over the subset (ingest's rule); the parent's query attrs
    items = load_sharded(content / "shard_index.json", torch.device("cpu"), normalize=True)
    tgt, _ = nearest_items(items, _load(content / "query_emb.pt").float())
    del items
    split = pl.read_parquet(parent / "eval_split.parquet")
    ids = (tgt + 1).numpy()
    pl.DataFrame({"item_id": ids}, schema={"item_id": pl.Int64}).write_parquet(
        out / "heldout.parquet"
    )
    pl.DataFrame(
        {"target_id": ids, "query_attrs_narrow": split["query_attrs_narrow"]},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(out / "eval_split.parquet")

    rows_sha = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    problems = layout.validate_layout(out, content)
    merge_prep_log(out, "subset", {
        "parent": parent.name, "n_parent": n_parent, "n_items": n, "seed": seed,
        "rows_sha256": rows_sha, "synth": synth, "problems": problems,
        "targets": "exact unfiltered top-1 by inner product over the subset, fp32, ties to the "
                   "lowest id",
    })  # fmt: skip
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{file_hexdigest(p, 'sha256')}  {p.relative_to(out)}\n" for p in files)
    )
    return {"n_items": n, "seed": seed, "rows_sha256": rows_sha, "problems": problems}


def write_configs(parent: str, name: str, config_dir: Path = CONFIG_DIR) -> None:
    """``<name>.yaml`` / ``<name>-synth.yaml``: the parent's configs on the subset's data dir."""
    for suffix in ("", "-synth"):
        src = Path(config_dir) / f"{parent}{suffix}.yaml"
        if not src.exists():
            continue
        cfg = yaml.safe_load(src.read_text())
        cfg["data_dir"] = f"data/{name}"
        header = (
            f"# Written by `eval-data subset {parent} {name}` (eval_datasets/subset.py): "
            f"{parent}{suffix}'s config on the seeded row subset data/{name} "
            "(docs/system/datasets.md § Subsets).\n"
        )
        dumped = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)
        (Path(config_dir) / f"{name}{suffix}.yaml").write_text(header + dumped)


@click.command("subset")
@click.argument("parent")
@click.argument("name")
@click.option("--n-items", required=True, type=int)
@click.option("--seed", default=SEED, show_default=True, type=int)
@click.option("--config-dir", default=CONFIG_DIR, type=click.Path(path_type=Path))
def subset_cmd(parent: str, name: str, n_items: int, seed: int, config_dir: Path) -> None:
    """Build data/NAME as a seeded N_ITEMS-row subset of data/PARENT, and its configs."""
    click.echo(json.dumps(build(data_root() / parent, data_root() / name, n_items, seed)))
    write_configs(parent, name, config_dir)
