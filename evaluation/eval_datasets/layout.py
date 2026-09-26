"""The on-disk dataset layout as code (docs/system/datasets.md § On-disk layout): the readers
the harness uses, the checks they carry, ``apply_users_limit`` (one prefix over every
query-side tensor) and ``validate_layout`` (``bench check``). ``eval_datasets/etl/*`` write
this layout; nothing here imports the harness or the trainer.

Two item layouts are accepted, not assumed (roadmap A1): the modern ``[N, …]`` 0-indexed one
and the pre-``3b1b5b3`` 1-indexed ``[N+1, …]`` one with a training-side padding row at index
0 — what the copies published on the Hub still are. ``drop_legacy_padding_row`` recognises
that row by its content (all-zero for an embedding matrix, all ``-1`` for an attribute tensor)
and drops it; ``check_items_aligned`` then requires the attrs to describe exactly the items.
Held-out target ids are 1-indexed item ids on disk in both layouts, so the ``-1`` shift is the
same in both; what the drop fixes is the *rows* those ids index. On the text path attrs and
embeddings agree with each other either way and only the targets would be off by one —
``cos(query, target)`` 0.99 → 0.62, no error — which is why the drop is loud.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger

EXPECTED_DOC_PREFIX = "search_document: "
EXPECTED_QUERY_PREFIX = "search_query: "


def atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """``write(fh)`` into ``<path>.tmp``, fsync, ``os.replace``: a crash mid-save leaves the old
    file or nothing, never a torn one. The tmp file is removed on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ----- items --------------------------------------------------------------------------------


def drop_legacy_padding_row(t: torch.Tensor, *, kind: str, what: str) -> torch.Tensor:
    """Drop row 0 of a pre-``3b1b5b3`` ``[N+1, …]`` tensor when its content says it is the
    padding row (``kind="emb"``: all-zero; ``kind="attrs"``: all ``-1``); anything else is
    returned untouched, so a modern tensor whose row 0 is a real item is never shortened."""
    if kind not in ("emb", "attrs"):
        raise ValueError(f"kind must be 'emb' or 'attrs', got {kind!r}")
    if t.ndim == 0 or t.shape[0] == 0:
        return t
    is_pad = bool((t[0] == 0).all()) if kind == "emb" else bool((t[0] == -1).all())
    if not is_pad:
        return t
    logger.warning(
        "{}: row 0 is a padding row — legacy 1-indexed [N+1, …] layout (pre-3b1b5b3, as "
        "published on the Hub). Dropping it: {} → {} rows.",
        what,
        t.shape[0],
        t.shape[0] - 1,
    )
    return t[1:].contiguous()


def check_items_aligned(n_items: int, n_attrs: int, *, what: str) -> None:
    """Row ``i`` of the attrs and of the embeddings is item id ``i+1``; a mismatch is a filter
    mask scoring every query against the wrong items, so it fails here naming both counts."""
    if n_attrs != n_items:
        raise RuntimeError(
            f"{what} has {n_attrs} rows but there are {n_items} items: the filter mask would "
            f"be misaligned. Expected {n_items} (0-indexed dense) or {n_items + 1} (legacy "
            "[N+1] with a padding row at index 0) — see docs/system/datasets.md and "
            "eval_datasets.layout.drop_legacy_padding_row"
        )


def prefix_problem(meta: Path, expected: str) -> str | None:
    """The prefix policy, in one place: a ``*.meta.json`` sidecar must carry a ``prefix`` key
    equal to the nomic prefix the harness expects **or explicitly ``null``** — the encoder has
    no prefix concept (YFCC's CLIP descriptors, MedCPT's precomputed vectors), a declared fact
    rather than an omission. A sidecar *without* the key is a problem: it cannot be told apart
    from a forgotten prefix. Returns the message, or ``None`` when the sidecar is fine."""
    with open(meta) as f:
        payload = json.load(f)
    if "prefix" not in payload:
        return f"{meta}: no 'prefix' key (write null to declare a prefix-free encoder)"
    prefix = payload["prefix"]
    if prefix is None or prefix == expected:
        return None
    return f"{meta}: prefix={prefix!r} != {expected!r}"


def assert_prefixes(content_dir: Path) -> None:
    """Catch silent nomic prefix swaps (they cost 5–15 % arxiv recall with no error); a
    ``prefix: null`` sidecar passes (``prefix_problem``)."""
    for name, expected in (("text_emb", EXPECTED_DOC_PREFIX), ("query_emb", EXPECTED_QUERY_PREFIX)):
        meta = content_dir / f"{name}.meta.json"
        if not meta.exists():
            logger.warning("missing {} — skipping the prefix assertion", meta)
            continue
        problem = prefix_problem(meta, expected)
        if problem is not None:
            raise RuntimeError(f"{problem} — re-encode")


def load_sharded(
    shard_index: Path, device: torch.device, *, normalize: bool = False
) -> torch.Tensor:
    """``content/text_emb_shard_*.pt`` → one ``[N, D]`` tensor on device, in the shards' dtype,
    or with ``normalize`` fp32 L2-normalised shard by shard — the peak is then the fp32 matrix
    plus one shard, not the three whole-matrix copies of ``.float()`` + ``F.normalize``."""
    with open(shard_index) as f:
        idx = json.load(f)
    n, d = int(idx["n_items"]), int(idx["dim"])
    dtype = torch.float32 if normalize else getattr(torch, idx["dtype"])
    out = torch.empty((n, d), dtype=dtype, device=device)
    for s in idx["shards"]:
        shard = torch.load(str(shard_index.parent / s["filename"]), map_location=device)
        start, rows = int(s["start_id"]), int(s["n_rows"])
        if shard.shape != (rows, d):
            raise RuntimeError(f"{s['filename']}: shape {tuple(shard.shape)} != ({rows}, {d})")
        out[start : start + rows] = F.normalize(shard.float(), dim=-1) if normalize else shard
        del shard
    return out


def load_text_items(content_dir: Path, device: torch.device) -> torch.Tensor:
    """``[N, D]`` fp32 L2-normalised item embeddings of a pre-encoded text layout (the pad row
    of a legacy file dropped *before* normalisation); the prefix sidecars asserted first."""
    assert_prefixes(content_dir)
    shard_index = content_dir / "shard_index.json"
    what = f"{content_dir.name}/text_emb"
    if shard_index.exists():
        # Row-wise, so normalising before the pad-row drop is the same; an all-zero row stays zero.
        return drop_legacy_padding_row(
            load_sharded(shard_index, device, normalize=True), kind="emb", what=what
        )
    item_embs = torch.load(str(content_dir / "text_emb.pt"), map_location=device)
    item_embs = drop_legacy_padding_row(item_embs, kind="emb", what=what)
    return F.normalize(item_embs.float().contiguous(), dim=-1)


def load_text_queries(
    data_dir: Path, content_dir: Path, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(queries [U, D] fp32 normalised, targets [U, 1] 0-indexed, n_targets [U])`` from
    ``query_emb.pt`` + ``heldout.parquet`` (whose 1-indexed ``item_id`` is shifted ``-1``)."""
    heldout = pl.read_parquet(data_dir / "heldout.parquet")
    targets = torch.tensor(heldout["item_id"].to_list(), dtype=torch.long).unsqueeze(-1) - 1
    queries = torch.load(str(content_dir / "query_emb.pt"), map_location="cpu")
    if queries.shape[0] != heldout.height:
        raise RuntimeError(f"query_emb rows={queries.shape[0]} != heldout rows={heldout.height}")
    if queries.shape[1] != dim:
        raise RuntimeError(f"query dim={queries.shape[1]} != item dim={dim}")
    queries = F.normalize(queries.float(), dim=-1)
    return queries, targets, torch.ones(heldout.height, dtype=torch.long)


def load_query_attrs(eval_split: Path, n_queries: int) -> torch.Tensor:
    """``[U, C]`` int64 ``query_attrs_narrow`` from ``eval_split.parquet``, aligned 1:1 with the
    *full* test split — checked here, before any trim."""
    df = pl.read_parquet(eval_split)
    if df.height != n_queries:
        raise RuntimeError(f"{eval_split}: rows={df.height} != queries={n_queries}; regen `attrs`")
    return torch.tensor(df["query_attrs_narrow"].to_list(), dtype=torch.long)


def load_item_attrs(
    attrs: Path, reverse: Path | None, n_items: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(item_attrs [N, C, A], clause_is_reverse [C] | None)`` on device, the legacy pad row
    dropped and the row count checked against ``n_items``."""
    item_attrs = torch.load(str(attrs), map_location=device, weights_only=True)
    item_attrs = drop_legacy_padding_row(item_attrs, kind="attrs", what=attrs.name)
    check_items_aligned(n_items, int(item_attrs.shape[0]), what=attrs.name)
    rev = torch.load(str(reverse), map_location=device, weights_only=True) if reverse else None
    return item_attrs, rev


def apply_users_limit(n: int | None, *tensors: torch.Tensor | None) -> tuple:
    """The one ``users_limit`` site: the first ``n`` rows of every query-side tensor (prefix,
    not sample — H §2.2, §7). ``None`` entries pass through; a limit at or above the split is
    a no-op."""
    if n is None or not tensors or tensors[0] is None or n >= tensors[0].shape[0]:
        return tensors
    logger.info("users_limit={}: keeping the first {} of {} queries", n, n, tensors[0].shape[0])
    return tuple(t[:n] if t is not None else None for t in tensors)


# ----- validate_layout ----------------------------------------------------------------------


def _rows(path: Path, kind: str) -> int:
    t = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    return int(drop_legacy_padding_row(t, kind=kind, what=path.name).shape[0])


def validate_layout(data_dir: Path, content_dir: Path | None = None) -> list[str]:
    """Every way a dataset directory can be wrong for the harness, as messages (empty = ok):
    the files each shape needs, the nomic prefix sidecars, and the row alignments the readers
    above enforce — items vs attrs, queries vs held-out vs ``eval_split``. ``content_dir`` set
    = the text shape; unset = the sequential shape (``test.parquet`` + a checkpoint)."""
    data_dir = Path(data_dir)
    problems: list[str] = []
    n_items = n_queries = None
    if content_dir is not None:
        content_dir = Path(content_dir)
        query_emb, text_emb, heldout = (
            content_dir / "query_emb.pt", content_dir / "text_emb.pt", data_dir / "heldout.parquet"
        )  # fmt: skip
        if not query_emb.exists():
            problems.append(f"missing {query_emb}")
        if (content_dir / "shard_index.json").exists():
            n_items = int(json.loads((content_dir / "shard_index.json").read_text())["n_items"])
        elif text_emb.exists():
            n_items = _rows(text_emb, "emb")
        else:
            problems.append(f"missing {text_emb}")
        prefixes = (("text_emb", EXPECTED_DOC_PREFIX), ("query_emb", EXPECTED_QUERY_PREFIX))
        for name, expected in prefixes:
            meta = content_dir / f"{name}.meta.json"
            if not meta.exists():
                problems.append(f"missing {meta} (no prefix assertion possible)")
            elif (problem := prefix_problem(meta, expected)) is not None:
                problems.append(problem)
        if heldout.exists():
            n_queries = pl.read_parquet(heldout).height
            if query_emb.exists() and (q := _rows(query_emb, "emb")) != n_queries:
                problems.append(f"query_emb rows {q} != heldout rows {n_queries}")
        else:
            problems.append(f"missing {heldout}")
    else:  # the sequential shape: what the harness reads (eval-data fetch ships no train/val)
        for name in ("item_id_map.json", "test.parquet"):
            if not (data_dir / name).exists():
                problems.append(f"missing {data_dir / name}")
        if (data_dir / "item_id_map.json").exists():
            n_items = len(json.loads((data_dir / "item_id_map.json").read_text()))
        if (data_dir / "test.parquet").exists():
            n_queries = pl.read_parquet(data_dir / "test.parquet").height
    attrs = data_dir / "item_attrs_narrow.pt"
    if attrs.exists():
        n_attrs = _rows(attrs, "attrs")
        if n_items is not None and n_attrs != n_items:
            problems.append(f"{attrs.name} rows {n_attrs} != items {n_items}")
        if not (data_dir / "clause_is_reverse_narrow.pt").exists():
            problems.append(f"missing {data_dir / 'clause_is_reverse_narrow.pt'}")
        split = data_dir / "eval_split.parquet"
        if not split.exists():
            problems.append(f"missing {split}")
        elif n_queries is not None:
            rows = pl.read_parquet(split).height
            if rows != n_queries:
                problems.append(f"eval_split rows {rows} != queries {n_queries}")
    return problems


__all__ = [
    "EXPECTED_DOC_PREFIX",
    "EXPECTED_QUERY_PREFIX",
    "apply_users_limit",
    "assert_prefixes",
    "atomic_write",
    "check_items_aligned",
    "drop_legacy_padding_row",
    "load_item_attrs",
    "load_query_attrs",
    "load_sharded",
    "load_text_items",
    "load_text_queries",
    "prefix_problem",
    "validate_layout",
]
