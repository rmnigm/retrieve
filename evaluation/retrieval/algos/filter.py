"""Filter construction + per-batch mask building.

The harness has exactly one filter module per (filter_kind, attrs) pair —
``ExactAttributeFilter`` for ``clause``, ``BloomFilter`` for ``bloom``,
``None`` for ``none``. ``build_filter`` builds it; ``make_mask`` is the
one-line helper algos call to turn per-batch query attrs into a ``[B, N]``
bool mask with the padding-row sentinel zeroed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import Backend, FilterModule
from retrieve.layers.filters import BloomFilter, ExactAttributeFilter


def build_filter(
    filter_kind: str,
    *,
    item_attrs_narrow: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    bloom_m_bits: int = 1024,
    bloom_k_hash: int = 5,
    device: torch.device,
    backend: Backend = "triton",
) -> FilterModule | None:
    """Build the filter module for one ``filter_kind``.

    Returns ``None`` for ``filter_kind="none"`` (no filtering). Bloom is
    paper-strict forward-only: the YAML's bloom block lists only
    forward-clause sweeps so reverse columns in the shared
    ``clause_is_reverse`` never get bloom-queried, and we pass
    ``clause_is_reverse=None`` to bypass the lib-level guard. ``backend``
    routes through to ``ExactAttributeFilter`` / ``BloomFilter`` so the
    filter kernel matches the retrieval-module backend in each cell.
    """
    if filter_kind == "none":
        return None
    if item_attrs_narrow is None:
        raise ValueError(f"filter_kind={filter_kind} requires item_attrs_narrow")
    if filter_kind == "clause":
        f = ExactAttributeFilter(backend=backend).to(device)
        f.register_index(item_attrs_narrow, clause_is_reverse=clause_is_reverse)
        return f
    if filter_kind == "bloom":
        bf = BloomFilter(
            m_bits=bloom_m_bits, k_hash=bloom_k_hash, backend=backend
        ).to(device)
        bf.register_index(item_attrs_narrow)
        return bf
    raise ValueError(f"unknown filter_kind: {filter_kind!r}")


def make_mask(
    filter_mod: FilterModule | None,
    qa_narrow: Tensor | None,
) -> Tensor | None:
    """Per-batch ``[B, N]`` bool mask for algos that consume one.

    Returns ``None`` if either input is ``None`` (unfiltered cell or
    pure-IVF batch). Item 0 is the padding row (``item_embs[0] = 0``)
    and the filtered-FullScan oracle sets ``scores[:, 0] = -inf``;
    forcing the sentinel False here keeps every algo's mask in agreement
    with the oracle even when reverse clauses would otherwise admit it.
    """
    if filter_mod is None or qa_narrow is None:
        return None
    m = filter_mod.evaluate_mask(qa_narrow)
    m[:, 0] = False
    return m
