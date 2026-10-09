"""``eval-data ingest``: a staged text-shape dataset, and its ``config/<name>.yaml``, from
vectors and attribute tables, with no per-dataset ETL (docs/system/datasets.md § Adding a
dataset; retrieve/docs/adding-a-dataset.md).

Inputs:

- **item vectors**: one or more ``.npy`` files ``[n_i, D]`` (fp16 or fp32), concatenated in
  the order given. Each is L2-normalised in fp32 and stored fp16 as
  ``content_d<D>/text_emb_shard_NNN.pt``, the sharded layout.
- **item attributes**: a parquet file, row ``i`` = item ``i``. There is one column per
  attribute, of any scalar type or a list of one. Each column is dictionary-coded (codes by
  count, descending) into ``item_attrs_narrow.pt [N, C, A]``, where ``A`` is the longest list.
  A null is ``-1``.
- **query vectors**: one ``.npy`` ``[U, D]``. **Query attributes**: a parquet file with the
  same columns (scalars only), coded with the items' vocabulary; a value no item has is
  ``-1``, which turns that clause off for that query.
- **targets** (optional): ``.npy`` ``[U]`` 0-indexed item rows. Without them, each query's
  target is its exact unfiltered top-1 item (yfcc10m's convention).

A clause is ``NAME[=COLUMN][!]``: ``COLUMN`` defaults to ``NAME``, and ``!`` makes the clause
reverse (passes items *without* the query's value). A sweep is ``NAME=CLAUSE,CLAUSE``; the
yaml's bloom block is every sweep with no reverse clause.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml

from eval_datasets.common import merge_prep_log
from eval_datasets.hub import data_root
from eval_datasets.layout import atomic_write, load_sharded, validate_layout

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def parse_clause(spec: str) -> tuple[str, str, bool]:
    """``NAME[=COLUMN][!]`` → ``(name, column, reverse)``."""
    reverse = spec.endswith("!")
    name, _, column = spec.rstrip("!").partition("=")
    return name, column or name, reverse


def nearest_items(items, queries, *, item_chunk: int = 2_000_000, query_batch: int = 256):
    """``(ids [U] 0-indexed, scores [U])``: each query's exact top-1 item by inner product in
    fp32, equal scores to the lowest id."""
    ids = torch.empty(queries.shape[0], dtype=torch.long)
    best = torch.empty(queries.shape[0], dtype=torch.float32)
    for qs in range(0, queries.shape[0], query_batch):
        q = queries[qs : qs + query_batch].to(items.device)
        b_s = torch.full((q.shape[0],), float("-inf"), device=items.device)
        b_i = torch.zeros(q.shape[0], dtype=torch.long, device=items.device)
        for start in range(0, items.shape[0], item_chunk):
            s = q @ items[start : start + item_chunk].t()
            m = s.max(dim=1).values
            pos = torch.arange(s.shape[1], device=items.device).expand_as(s)
            first = torch.where(s == m[:, None], pos, s.shape[1]).min(dim=1).values + start
            better = m > b_s  # strictly: an equal score in a later chunk has a higher id
            b_s, b_i = torch.where(better, m, b_s), torch.where(better, first, b_i)
        ids[qs : qs + q.shape[0]], best[qs : qs + q.shape[0]] = b_i.cpu(), b_s.cpu()
    return ids, best


def _vocab(values: pl.Series) -> pl.DataFrame:
    """``value → code``, codes by count descending, ties by value."""
    if isinstance(values.dtype, pl.List):
        values = values.explode()
    return (
        values.drop_nulls()
        .rename("value")
        .value_counts(name="count")
        .sort(["count", "value"], descending=[True, False])
        .with_row_index("code")
        .with_columns(pl.col("code").cast(pl.Int64))
    )


def _codes(values: pl.Series, vocab: pl.DataFrame, a_max: int) -> np.ndarray:
    """``[len(values), a_max]`` int64 codes, ``-1`` padded (unknown and null → ``-1``)."""
    out = np.full((len(values), a_max), -1, dtype=np.int64)
    df = pl.DataFrame({"row": np.arange(len(values)), "value": values})
    if isinstance(values.dtype, pl.List):
        df = df.explode("value").with_columns(pos=pl.int_range(pl.len()).over("row"))
    else:
        df = df.with_columns(pos=pl.lit(0, dtype=pl.Int64))
    df = df.join(vocab.select("value", "code"), on="value", how="inner")
    out[df["row"].to_numpy(), df["pos"].to_numpy()] = df["code"].to_numpy()
    return out


def _normalized_fp16(a: np.ndarray) -> torch.Tensor:
    return F.normalize(torch.from_numpy(np.array(a, dtype=np.float32)), dim=-1).half()


def _meta(prefix: str | None, n: int, d: int, sources: list[str]) -> dict:
    return {"prefix": prefix, "dim": d, "n_rows": n, "shape": [n, d], "dtype": "float16",
            "normalization": "l2", "sources": sources}  # fmt: skip


def ingest(
    name: str,
    *,
    items: Sequence[Path],
    item_attrs: Path,
    queries: Path,
    query_attrs: Path,
    clauses: Sequence[str],
    sweeps: Sequence[str],
    targets: Path | None = None,
    doc_prefix: str | None = None,
    query_prefix: str | None = None,
    output_dir: Path | None = None,
    config_dir: Path = CONFIG_DIR,
    shard_rows: int = 1_000_000,
    users_limit: int = 10_000,
    device: str = "cpu",
) -> list[str]:
    """Stage ``name`` and write ``config_dir/<name>.yaml``; returns ``validate_layout``'s
    problems (empty = ok)."""
    t0 = time.monotonic()
    out = Path(output_dir) if output_dir else data_root() / name
    specs = [parse_clause(c) for c in clauses]
    sweep_map = {}
    for s in sweeps:
        sname, _, members = s.partition("=")
        names = [c[0] for c in specs]
        sweep_map[sname] = [names.index(m) for m in members.split(",")]

    # Vectors: one fp16 shard per shard_rows of each input file.
    n, d, entries = 0, None, []
    for path in items:
        a = np.load(path, mmap_mode="r")
        if d is not None and a.shape[1] != d:
            raise ValueError(f"{path}: width {a.shape[1]} != {d}")
        d = int(a.shape[1])
        content = out / f"content_d{d}"
        content.mkdir(parents=True, exist_ok=True)
        for start in range(0, a.shape[0], shard_rows):
            shard = _normalized_fp16(a[start : start + shard_rows])
            fname = f"text_emb_shard_{len(entries):03d}.pt"
            atomic_write(content / fname, partial(torch.save, shard))
            entries.append({"filename": fname, "start_id": n, "n_rows": int(shard.shape[0])})
            n += int(shard.shape[0])
    q = _normalized_fp16(np.load(queries))
    if q.shape[1] != d:
        raise ValueError(f"{queries}: width {q.shape[1]} != items' {d}")
    atomic_write(content / "query_emb.pt", partial(torch.save, q))
    (content / "query_emb.meta.json").write_text(
        json.dumps(_meta(query_prefix, q.shape[0], d, [str(queries)]), indent=2)
    )
    (content / "text_emb.meta.json").write_text(
        json.dumps(_meta(doc_prefix, n, d, [str(p) for p in items]) | {"layout": "sharded"})
    )
    (content / "shard_index.json").write_text(
        json.dumps(
            {
                "n_items": n,
                "dim": d,
                "dtype": "float16",
                "n_shards": len(entries),
                "shards": entries,
            },
            indent=2,
        )  # fmt: skip
    )

    # Attributes: one vocabulary per source column, shared by the clauses that read it.
    columns = list(dict.fromkeys(c[1] for c in specs))
    it = pl.read_parquet(item_attrs, columns=columns)
    qt = pl.read_parquet(query_attrs, columns=columns)
    if it.height != n or qt.height != q.shape[0]:
        raise ValueError(f"attrs rows {it.height} / {qt.height} != vectors {n} / {q.shape[0]}")
    lists = [c for c in columns if isinstance(it[c].dtype, pl.List)]
    if any(isinstance(qt[c].dtype, pl.List) for c in columns):
        raise ValueError("query attrs must be scalar: a query sends one value per clause")
    a_max = max([1, *(int(it[c].list.len().max() or 0) for c in lists)])
    vocabs = {c: _vocab(it[c]) for c in columns}
    item_codes = {c: _codes(it[c], vocabs[c], a_max) for c in columns}
    attrs = np.stack([item_codes[c] for _, c, _ in specs], axis=1)
    qa = np.stack([_codes(qt[c], vocabs[c], 1)[:, 0] for _, c, _ in specs], axis=1)
    reverse = [r for _, _, r in specs]
    torch.save(torch.from_numpy(attrs), out / "item_attrs_narrow.pt")
    torch.save(torch.tensor(reverse), out / "clause_is_reverse_narrow.pt")
    (out / "vocab.json").write_text(
        json.dumps({c: {"values": v["value"].cast(pl.Utf8).to_list(),
                        "counts": v["count"].to_list()} for c, v in vocabs.items()})
    )  # fmt: skip

    if targets is not None:
        tgt = torch.from_numpy(np.load(targets).astype(np.int64))
        rule = f"given: {targets}"
    else:
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        emb = load_sharded(content / "shard_index.json", torch.device(device), normalize=True)
        tgt, _ = nearest_items(emb, q.float())
        del emb
        rule = "exact unfiltered top-1 by inner product, fp32, ties to the lowest id"
    if tgt.shape[0] != q.shape[0] or not ((tgt >= 0) & (tgt < n)).all():
        raise ValueError("targets must be [U] item rows in [0, N)")
    pl.DataFrame({"item_id": (tgt + 1).numpy()}, schema={"item_id": pl.Int64}).write_parquet(
        out / "heldout.parquet"
    )
    pl.DataFrame(
        {"target_id": (tgt + 1).numpy(), "query_attrs_narrow": qa.tolist()},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(out / "eval_split.parquet")

    rel = out.relative_to(data_root()) if out.is_relative_to(data_root()) else None
    bloom = {k: v for k, v in sweep_map.items() if not any(reverse[c] for c in v)}
    config = {
        "data_dir": f"data/{rel}" if rel else str(out),
        "content_dir": f"content_d{d}",
        "dims": [d],
        "users_limit": users_limit,
        "filters": {
            "attrs": "item_attrs_narrow.pt",
            "reverse": "clause_is_reverse_narrow.pt",
            "clause": sweep_map,
            **({"bloom": bloom} if bloom else {}),
        },
    }
    header = (
        f"# Written by `eval-data ingest {name}` (eval_datasets/ingest.py). Clauses, in order: "
        + ", ".join(f"C{i} {s[0]}{' (reverse)' if s[2] else ''}" for i, s in enumerate(specs))
        + ".\n"
    )
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    (Path(config_dir) / f"{name}.yaml").write_text(
        header + yaml.safe_dump(config, sort_keys=False, default_flow_style=None)
    )
    problems = validate_layout(out, content)
    pass_rate = {
        s[0]: round(float(np.mean([
            (attrs[:, i, :] == qa[j, i]).any(axis=1).mean() if not s[2]
            else 1 - (attrs[:, i, :] == qa[j, i]).any(axis=1).mean()
            for j in range(min(qa.shape[0], 100)) if qa[j, i] >= 0
        ] or [np.nan])), 6)
        for i, s in enumerate(specs)
    }  # fmt: skip
    merge_prep_log(out, "ingest", {
        "n_items": n, "n_queries": int(q.shape[0]), "dim": d, "a_max": a_max,
        "clauses": [list(s) for s in specs], "sweeps": sweep_map, "targets": rule,
        "vocab_sizes": {c: v.height for c, v in vocabs.items()},
        "query_clause_active": {s[0]: round(float((qa[:, i] >= 0).mean()), 4)
                                for i, s in enumerate(specs)},
        "pass_rate_first_100_queries": pass_rate,
        "problems": problems, "wall_clock_sec": round(time.monotonic() - t0, 1),
    })  # fmt: skip
    return problems


@click.command("ingest")
@click.argument("name")
@click.option("--items", "items", multiple=True, required=True, type=click.Path(path_type=Path),
              help="item vectors, .npy [n_i, D]; repeat, in item order")  # fmt: skip
@click.option("--item-attrs", required=True, type=click.Path(path_type=Path), help="parquet")
@click.option("--queries", required=True, type=click.Path(path_type=Path), help=".npy [U, D]")
@click.option("--query-attrs", required=True, type=click.Path(path_type=Path), help="parquet")
@click.option("--clause", "clauses", multiple=True, required=True, help="NAME[=COLUMN][!]")
@click.option("--sweep", "sweeps", multiple=True, required=True, help="NAME=CLAUSE,CLAUSE")
@click.option("--targets", type=click.Path(path_type=Path), help=".npy [U] 0-indexed items")
@click.option("--doc-prefix", default=None, help="the encoder's item prefix (default: none)")
@click.option("--query-prefix", default=None, help="the encoder's query prefix")
@click.option("--output-dir", type=click.Path(path_type=Path), help="default: data root/NAME")
@click.option("--config-dir", default=CONFIG_DIR, type=click.Path(path_type=Path))
@click.option("--shard-rows", default=1_000_000, show_default=True)
@click.option("--users-limit", default=10_000, show_default=True)
@click.option("--device", default="cpu", show_default=True, help="for the computed targets")
def ingest_cmd(name: str, **kw) -> None:
    """Stage a text-shape dataset from vectors + attribute tables, write its yaml, check it."""
    problems = ingest(name, **kw)
    for p in problems:
        click.echo(f"PROBLEM {p}")
    click.echo(f"{name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    raise SystemExit(1 if problems else 0)


__all__ = ["ingest", "ingest_cmd", "nearest_items", "parse_clause"]
