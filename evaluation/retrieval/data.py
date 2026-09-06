"""Inputs for harness v2 (H §2.2, §3.1 ``data.py``): item embeddings, queries, held-out
targets, query attrs, item attrs — loaded once per ``(dataset, dim)`` — plus the per-sweep
query attrs, the filter modules keyed by *filter* backend and the fixed-seed perf pool.

``load_inputs`` dispatches on the ``Dataset``: ``checkpoint`` → SASRec encode of the test
split (cached next to the checkpoint, keyed on ckpt mtime + ``max_seq_length``, the *full*
split so the trim below is the only ``users_limit`` site); ``content_dir`` → the pre-encoded
text layout (``text_emb.pt`` / ``query_emb.pt`` + meta sidecars whose nomic prefixes are
asserted, ``heldout.parquet``; sharded synth catalogs reassembled). ``users_limit`` is a
prefix, applied in exactly one place, to queries / targets / query attrs together (H §2.2,
§7: prefix, not sample, for golden comparability). The numerics are the old ``loaders.py``
ones (pad row dropped, targets shifted −1, fp16 → fp32 + L2-normalise on the text path);
``encode.py`` stays the one file that imports ``training.*``.

**Two on-disk layouts are accepted, not assumed** (roadmap A1, ``df6db40``): the modern
``[N, …]`` 0-indexed dense one ``docs/system/datasets.md`` documents, and the pre-``3b1b5b3``
1-indexed ``[N+1, …]`` one with a training-side padding row at index 0 — which is what the
copies published on the Hub (``eval-fetch``) still are. ``drop_legacy_padding_row`` recognises
the padding row by its *content* (all-zero for an embedding matrix, all ``-1`` for an
attribute tensor) and drops it from every per-item tensor — the pre-encoded ``text_emb`` and
``item_attrs_narrow`` (the SASRec path's ``nn.Embedding`` always carries the pad row and
loses it by construction) — and ``load_inputs`` then requires ``item_attrs`` and ``item_embs``
to have the same row count, raising with both counts on anything else. Held-out target ids
are 1-indexed item ids on disk in both layouts, so the ``-1`` shift below is the same in
both; what the pad-row drop fixes is the *rows* those ids index. Getting this wrong is not
always loud: on the arxiv path attrs and embeddings are both 1-indexed and agree with each
other, so only the targets are off by one — ``cos(query, target)`` 0.99 → 0.62, no error.

Returned dict (``inputs``): ``item_embs [N, D]`` fp32 on device; ``queries [U, D]`` fp32,
``targets [U, T]`` int64 ``-1``-padded 0-indexed, ``n_targets [U]`` — all CPU; ``qa
[U, C]`` int64 CPU or ``None``; ``item_attrs [N, C, A]`` int64 and ``clause_is_reverse
[C]`` bool on device or ``None``; ``attrs_digest`` (``oracle.attrs_digest`` of those two,
the item side of the oracle fingerprint, hashed once here); ``n_items``, ``n_queries``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger

from retrieval.algos import FILTER_BACKEND, build_filter
from retrieval.bench import atomic_write
from retrieval.config import Dataset
from retrieval.encode import encode_queries, load_model_for_eval
from retrieval.oracle import attrs_digest
from retrieve.interfaces import FilterModule

EXPECTED_DOC_PREFIX = "search_document: "
EXPECTED_QUERY_PREFIX = "search_query: "
ENCODE_CACHE = "encoded_queries_v2.pt"
_CACHE_FREE_FRACTION, _CACHE_RESERVE_BYTES = 0.7, 4 * 2**30


# ----- legacy [N+1, …] layout ------------------------------------------------------------


def drop_legacy_padding_row(t: torch.Tensor, *, kind: str, what: str) -> torch.Tensor:
    """Drop row 0 of a pre-``3b1b5b3`` ``[N+1, …]`` retrieval tensor (module docstring).

    A padding row is recognised by its content, never by its position alone: all-zero for
    an embedding matrix (``kind="emb"``; impossible for an L2-normalised row) and all ``-1``
    for an attribute tensor (``kind="attrs"``; the ``-1`` fill is "no value"). Anything else
    is returned untouched, so a modern ``[N, …]`` tensor whose row 0 is a real item is never
    shortened. Dropping is loud (a warning naming ``what`` and both row counts)."""
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
    """``item_attrs`` must describe exactly the ``n_items`` rows of ``item_embs`` — row ``i``
    of both is item id ``i+1``. A mismatch is a misaligned filter mask that would score
    every query against the wrong items three layers down (in the oracle, or silently in
    the algos), so it fails here, naming both counts and the two accepted layouts."""
    if n_attrs != n_items:
        raise RuntimeError(
            f"{what} has {n_attrs} rows but there are {n_items} items: the filter mask would "
            f"be misaligned. Expected {n_items} (0-indexed dense) or {n_items + 1} (legacy "
            "[N+1] with a padding row at index 0) — see docs/system/datasets.md and "
            "retrieval.data.drop_legacy_padding_row"
        )


# ----- item embeddings + queries ---------------------------------------------------------


def _assert_prefixes(content_dir: Path) -> None:
    """Catch silent nomic prefix swaps (they cost 5–15 % arxiv recall with no error)."""
    for name, expected in (("text_emb", EXPECTED_DOC_PREFIX), ("query_emb", EXPECTED_QUERY_PREFIX)):
        meta = content_dir / f"{name}.meta.json"
        if not meta.exists():
            logger.warning("missing {} — skipping the prefix assertion", meta)
            continue
        with open(meta) as f:
            prefix = json.load(f).get("prefix")
        if prefix != expected:
            raise RuntimeError(f"{meta}: prefix={prefix!r} != {expected!r} — re-encode")


def _load_sharded(shard_index: Path, device: torch.device) -> torch.Tensor:
    """``content/text_emb_shard_*.pt`` (synth catalogs) → one ``[N, D]`` tensor on device."""
    with open(shard_index) as f:
        idx = json.load(f)
    n, d, dtype = int(idx["n_items"]), int(idx["dim"]), getattr(torch, idx["dtype"])
    out = torch.empty((n, d), dtype=dtype, device=device)
    for s in idx["shards"]:
        shard = torch.load(str(shard_index.parent / s["filename"]), map_location=device)
        start, rows = int(s["start_id"]), int(s["n_rows"])
        if shard.shape != (rows, d):
            raise RuntimeError(f"{s['filename']}: shape {tuple(shard.shape)} != ({rows}, {d})")
        out[start : start + rows].copy_(shard)
        del shard
    return out


def _pre_encoded(ds: Dataset, device: torch.device) -> tuple[torch.Tensor, ...]:
    content = ds.content_dir
    assert content is not None
    _assert_prefixes(content)
    shard_index = content / "shard_index.json"
    if shard_index.exists():
        item_embs = _load_sharded(shard_index, device)
    else:
        item_embs = torch.load(str(content / "text_emb.pt"), map_location=device)
    # A legacy [N+1, D] file loses its all-zero padding row here, *before* normalisation.
    # This is the path where the old layout is silent: attrs and embeddings agree with each
    # other and only the target shift below would be wrong (module docstring).
    item_embs = drop_legacy_padding_row(item_embs, kind="emb", what=f"{content.name}/text_emb")
    item_embs = F.normalize(item_embs.float().contiguous(), dim=-1)  # fp16 on disk → fp32
    heldout = pl.read_parquet(ds.data_dir / "heldout.parquet")
    # heldout stores 1-indexed item ids (the encoder's pad-aware space) → 0-indexed. The
    # shift is the same in both layouts; row i of the pad-row-dropped item_embs is id i+1.
    targets = torch.tensor(heldout["item_id"].to_list(), dtype=torch.long).unsqueeze(-1) - 1
    queries = torch.load(str(content / "query_emb.pt"), map_location="cpu")
    if queries.shape[0] != heldout.height:
        raise RuntimeError(f"query_emb rows={queries.shape[0]} != heldout rows={heldout.height}")
    if queries.shape[1] != item_embs.shape[1]:
        raise RuntimeError(f"query dim={queries.shape[1]} != item dim={item_embs.shape[1]}")
    queries = F.normalize(queries.float(), dim=-1)
    return item_embs, queries, targets, torch.ones(heldout.height, dtype=torch.long)


def _sasrec(ds: Dataset, device: torch.device) -> tuple[torch.Tensor, ...]:
    ckpt = ds.checkpoint
    assert ckpt is not None
    cache = ckpt.parent / ENCODE_CACHE
    key = {"ckpt_mtime": ckpt.stat().st_mtime, "max_seq_length": ds.encode["max_seq_length"]}
    if cache.exists():
        blob = torch.load(str(cache), map_location="cpu", weights_only=True)
        if all(blob.get(k) == v for k, v in key.items()):
            logger.info("loaded encoded queries from {}", cache)
            item_embs = blob["item_embs"].to(device).contiguous()
            return item_embs, blob["queries"], blob["targets"], blob["n_targets"]
        logger.info("{}: stale (ckpt mtime / max_seq_length changed); re-encoding", cache)
    with open(ds.data_dir / "item_id_map.json") as f:
        num_items = len(json.load(f))
    model = load_model_for_eval(ckpt, num_items=num_items, device=device)
    # Drop the training-side padding row (item id 0): retrieval is 0-indexed over real items.
    item_embs = model.get_output_embeddings().weight.detach()[1:].to(device).contiguous()
    queries, targets, n_targets = encode_queries(
        model,
        ds.data_dir / "test.parquet",
        max_length=ds.encode["max_seq_length"],
        encode_batch_size=ds.encode["batch_size"],
        num_workers=ds.encode["num_workers"],
        device=device,
    )
    targets = torch.where(targets >= 0, targets - 1, targets)  # 1-indexed ids → 0-indexed
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    blob = {
        "item_embs": item_embs.cpu(),
        "queries": queries,
        "targets": targets,
        "n_targets": n_targets,
        **key,
    }
    est = sum(t.numel() * t.element_size() for t in blob.values() if torch.is_tensor(t))
    budget = (
        int(shutil.disk_usage(str(ckpt.parent)).free * _CACHE_FREE_FRACTION) - _CACHE_RESERVE_BYTES
    )
    if est > budget:
        logger.warning(
            "not caching encoded queries: {:.1f} GB > budget {:.1f} GB", est / 2**30, budget / 2**30
        )
    else:
        atomic_write(cache, lambda fh: torch.save(blob, fh))
        logger.info("wrote {} ({:.2f} GB)", cache, est / 2**30)
    return item_embs, queries, targets, n_targets


def load_query_attrs(eval_split: Path, n_queries: int) -> torch.Tensor:
    """``[U, C]`` int64 ``query_attrs_narrow`` from ``eval_split.parquet`` (aligned 1:1 with
    the *full* test split — checked here, before any trim)."""
    df = pl.read_parquet(eval_split)
    if df.height != n_queries:
        raise RuntimeError(f"{eval_split}: rows={df.height} != queries={n_queries}; regen `attrs`")
    return torch.tensor(df["query_attrs_narrow"].to_list(), dtype=torch.long)


def load_inputs(ds: Dataset, device: torch.device, *, with_filters: bool = True) -> dict[str, Any]:
    """Everything a ``(dataset, dim)`` process needs (module docstring). ``with_filters``
    loads query / item attrs when the dataset has a ``filters:`` block."""
    item_embs, queries, targets, n_targets = (_sasrec if ds.checkpoint else _pre_encoded)(
        ds, device
    )
    qa = item_attrs = reverse = None
    if with_filters and ds.attrs is not None:
        qa = load_query_attrs(ds.data_dir / "eval_split.parquet", queries.shape[0])
        item_attrs = torch.load(str(ds.attrs), map_location=device, weights_only=True)
        # A legacy [N+1, C, A] file loses its all-(-1) padding row; either way the attrs
        # must then index exactly the items item_embs does (loud on goodreads, where the
        # SASRec path drops its pad row by construction and a legacy attrs file is one row
        # too long; checked rather than assumed everywhere else).
        item_attrs = drop_legacy_padding_row(item_attrs, kind="attrs", what=ds.attrs.name)
        check_items_aligned(int(item_embs.shape[0]), int(item_attrs.shape[0]), what=ds.attrs.name)
        if ds.reverse is not None:
            reverse = torch.load(str(ds.reverse), map_location=device, weights_only=True)
    n = ds.users_limit
    if n is not None and n < queries.shape[0]:  # the one users_limit site: a prefix
        logger.info("users_limit={}: keeping the first {} of {} queries", n, n, queries.shape[0])
        queries, targets, n_targets = queries[:n], targets[:n], n_targets[:n]
        qa = qa[:n] if qa is not None else None
    logger.info("inputs: items {} queries {} dim {}", item_embs.shape[0], queries.shape[0], ds.dim)
    return {
        "item_embs": item_embs,
        "queries": queries.contiguous(),
        "targets": targets.contiguous(),
        "n_targets": n_targets.contiguous(),
        "qa": qa,
        "item_attrs": item_attrs,
        "clause_is_reverse": reverse,
        "attrs_digest": attrs_digest(item_attrs, reverse),
        "n_items": int(item_embs.shape[0]),
        "n_queries": int(queries.shape[0]),
    }


# ----- per sweep ------------------------------------------------------------------------


def sweep_qa(
    qa: torch.Tensor | None, clauses: tuple[int, ...] | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(qa_sweep, skip_mask)`` for one sweep: inactive clauses coded ``-1`` (exact: match
    anything; bloom: no bits queried), rows with no live clause skip-masked. ``clauses``
    ``None`` (a ``none`` cell) → ``(None, None)``."""
    if clauses is None:
        return None, None
    if qa is None:
        raise ValueError(f"sweep over clauses {clauses} needs query attrs (eval_split.parquet)")
    n_clauses = qa.shape[1]
    if any(not 0 <= c < n_clauses for c in clauses):
        raise ValueError(f"clauses {clauses} out of range for {n_clauses} clauses")
    out = qa.clone()
    inactive = torch.ones(n_clauses, dtype=torch.bool)
    inactive[list(clauses)] = False
    out[:, inactive] = -1
    return out, (out == -1).all(dim=1)


def build_filters(
    filter_kind: str, inputs: dict[str, Any], backends: list[str], *, bloom: dict[str, int]
) -> dict[str, FilterModule]:
    """One standalone ``FilterModule`` per *filter* backend the cell backends map to
    (``FILTER_BACKEND``: official → triton), so ``[triton, official]`` builds one module.
    Empty for ``none``."""
    mods: dict[str, FilterModule] = {}
    if filter_kind == "none":
        return mods
    for fb in dict.fromkeys(FILTER_BACKEND[b] for b in backends):
        mods[fb] = build_filter(
            filter_kind,
            inputs["item_attrs"],
            clause_is_reverse=inputs["clause_is_reverse"],
            backend=fb,
            m_bits=bloom["m_bits"],
            k_hash=bloom["k_hash"],
        )
    logger.info("filter modules ({}): {}", filter_kind, sorted(mods))
    return mods


def exact_filter(
    filter_kind: str, filters: dict[str, FilterModule], inputs: dict[str, Any], backend: str
) -> FilterModule | None:
    """The oracle's exact mask source for a cell: the clause module itself on ``clause``
    cells, a fresh ``ExactAttributeFilter`` on ``bloom`` cells, ``None`` on ``none``."""
    fb = FILTER_BACKEND[backend]
    if filter_kind == "clause":
        return filters[fb]
    if filter_kind == "bloom":
        return build_filter(
            "clause",
            inputs["item_attrs"],
            clause_is_reverse=inputs["clause_is_reverse"],
            backend=fb,
        )
    return None


def query_pool(
    inputs: dict[str, Any],
    qa_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    *,
    bs: int,
    seed: int,
    n_pool: int = 4096,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The fixed-seed perf pool (H §2.5): ``[n_pool, bs, D]`` query batches drawn with
    ``torch.Generator().manual_seed(seed)`` from the kept rows (skip-masked rows excluded),
    plus the matching ``[n_pool, bs, C]`` attr batches on filter cells. Identical across
    backends and modes, and to the old harness's pool at the same seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    queries = inputs["queries"]
    if skip_mask is not None:
        keep_idx = (~skip_mask.bool()).nonzero(as_tuple=False).reshape(-1)
        if keep_idx.numel() == 0:
            raise ValueError("every query is skip-masked; nothing to time")
        rows = keep_idx[torch.randint(0, keep_idx.numel(), (n_pool, bs), generator=g).reshape(-1)]
    else:
        rows = torch.randint(0, queries.shape[0], (n_pool, bs), generator=g).reshape(-1)
    pool = queries[rows].reshape(n_pool, bs, -1).to(device).contiguous()
    qa_pool = (
        qa_sweep[rows].reshape(n_pool, bs, -1).to(device).contiguous()
        if qa_sweep is not None
        else None
    )
    return pool, qa_pool


__all__ = [
    "ENCODE_CACHE",
    "build_filters",
    "check_items_aligned",
    "drop_legacy_padding_row",
    "exact_filter",
    "load_inputs",
    "load_query_attrs",
    "query_pool",
    "sweep_qa",
]
