"""Global test fixtures and CUDA gate.

`retrieve` is a GPU-only library, so the entire suite is skipped without CUDA.
Tests build tensors explicitly with ``device="cuda"`` (via the helpers below
or directly) — using ``torch.set_default_device("cuda")`` would clash with
library code that intentionally creates CPU-side generators.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="retrieve is GPU-only; CUDA required")
        for it in items:
            it.add_marker(skip)


def make_index(
    n: int,
    d: int,
    *,
    normalized: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random unit-norm item embeddings, ``[N, D]``, on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    embs = torch.randn(n, d, generator=g, device="cuda", dtype=dtype)
    if normalized:
        embs = embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return embs


def make_query(
    b: int,
    d: int,
    *,
    normalized: bool = True,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random unit-norm query embeddings, ``[B, D]``, on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, d, generator=g, device="cuda", dtype=dtype)
    if normalized:
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return q


def make_mask(b: int, n: int, *, pass_rate: float, seed: int = 2) -> torch.Tensor:
    """Random boolean mask ``[B, N]`` with the given expected pass rate."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(b, n, generator=g, device="cuda") < pass_rate


def make_attrs(
    n: int,
    c: int,
    a_max: int,
    *,
    n_vocab: int = 50,
    pad_rate: float = 0.3,
    seed: int = 3,
) -> torch.Tensor:
    """Random per-item clause attributes ``[N, C, A_max]`` with -1 padding."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    attrs = torch.randint(0, n_vocab, (n, c, a_max), generator=g, dtype=torch.long, device="cuda")
    pad = torch.rand(n, c, a_max, generator=g, device="cuda") < pad_rate
    attrs[pad] = -1
    return attrs


def make_query_attrs(
    b: int,
    c: int,
    *,
    n_vocab: int = 50,
    inactive_rate: float = 0.2,
    seed: int = 4,
) -> torch.Tensor:
    """Random query clause attributes ``[B, C]`` with some clauses inactive (-1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, n_vocab, (b, c), generator=g, dtype=torch.long, device="cuda")
    inactive = torch.rand(b, c, generator=g, device="cuda") < inactive_rate
    q[inactive] = -1
    return q


def valid_id_set(ids: torch.Tensor, scores: torch.Tensor, b: int) -> set[int]:
    """Set of ids in row ``b`` with a finite score and a non-padded id."""
    valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
    return set(ids[b][valid].tolist())


def recall_at_k(approx_ids: torch.Tensor, exact_ids: torch.Tensor) -> float:
    """Mean per-row set overlap of ``approx`` vs ``exact``, divided by K."""
    b, k = approx_ids.shape
    overlap = sum(len(set(approx_ids[i].tolist()) & set(exact_ids[i].tolist())) for i in range(b))
    return overlap / (b * k)


def assert_recall_monotone(recalls: list[float], *, slack: float = 0.02) -> None:
    """Assert ``recalls`` is non-decreasing within ``slack`` between adjacent entries."""
    for prev, nxt in zip(recalls, recalls[1:]):
        assert nxt >= prev - slack, f"recall regressed: {recalls}"
