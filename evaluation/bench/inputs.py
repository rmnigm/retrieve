"""Inputs of a ``(dataset, dim)`` process (H §2.2): item embeddings, queries, held-out targets,
query attrs, item attrs — loaded once — plus the per-sweep query attrs, the standalone
filter modules keyed by *filter* backend, the oracle's exact mask source and the fixed-seed
perf pool.

``load_inputs`` dispatches on the ``Dataset``: ``checkpoint`` → ``training.encode.encode_split``
(SASRec over the full test split, cached next to the checkpoint); ``content_dir`` → the
pre-encoded text layout through ``eval_datasets.layout``. ``users_limit`` is applied once,
here, as a prefix over queries / targets / query attrs together.

Returned dict: ``item_embs [N, D]`` fp32 on device; ``queries [U, D]`` fp32, ``targets [U, T]``
int64 ``-1``-padded 0-indexed, ``n_targets [U]`` — all CPU; ``qa [U, C]`` int64 CPU or ``None``;
``item_attrs [N, C, A]`` int64 and ``clause_is_reverse [C]`` bool on device or ``None``;
``attrs_digest`` (``oracle.attrs_digest`` of those two, hashed once); ``n_items``, ``n_queries``.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from bench import algos
from bench.config import Dataset
from bench.oracle import attrs_digest
from eval_datasets import layout
from retrieve.interfaces import FilterModule
from training.encode import encode_split


def load_inputs(ds: Dataset, device: torch.device, *, with_filters: bool = True) -> dict[str, Any]:
    if ds.checkpoint:
        item_embs, queries, targets, n_targets = encode_split(
            ds.checkpoint,
            ds.data_dir,
            max_seq_length=ds.encode["max_seq_length"],
            batch_size=ds.encode["batch_size"],
            num_workers=ds.encode["num_workers"],
            device=device,
        )
    else:
        assert ds.content_dir is not None
        item_embs = layout.load_text_items(ds.content_dir, device)
        queries, targets, n_targets = layout.load_text_queries(
            ds.data_dir, ds.content_dir, int(item_embs.shape[1])
        )
    qa = item_attrs = reverse = None
    if with_filters and ds.attrs is not None:
        qa = layout.load_query_attrs(ds.data_dir / "eval_split.parquet", queries.shape[0])
        item_attrs, reverse = layout.load_item_attrs(
            ds.attrs, ds.reverse, int(item_embs.shape[0]), device
        )
    queries, targets, n_targets, qa = layout.apply_users_limit(
        ds.users_limit, queries, targets, n_targets, qa
    )
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
    (``official`` → ``triton``, so ``[triton, official]`` builds one module). Empty for ``none``."""
    mods: dict[str, FilterModule] = {}
    if filter_kind == "none":
        return mods
    for fb in dict.fromkeys(algos.filter_backend(b) for b in backends):
        mods[fb] = algos.build_filter(
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
    fb = algos.filter_backend(backend)
    if filter_kind == "clause":
        return filters[fb]
    if filter_kind == "bloom":
        return algos.build_filter(
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
    ``torch.Generator().manual_seed(seed)`` from the kept rows, plus the matching
    ``[n_pool, bs, C]`` attr batches on filter cells. Identical across backends and modes."""
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


__all__ = ["build_filters", "exact_filter", "load_inputs", "query_pool", "sweep_qa"]
