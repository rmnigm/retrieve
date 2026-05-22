"""Triton ``clause_mask`` vs the pure-torch broadcast baseline.

``ExactAttributeFilter.evaluate_mask`` now routes to ``clause_mask`` on CUDA, so we
compute the broadcast reference inline (intentionally materializing the
``[B, N, C, A_max]`` intermediate this kernel exists to avoid) and assert
byte-exact equality. No compaction → ordering is unambiguous.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.filters.clause_mask import (
    ClauseMaskConfig,
    _clause_mask_impl,
    clause_mask,
)
from tests.conftest import make_attrs, make_query_attrs


def _ref_mask(
    item_clause_attrs: torch.Tensor,
    clause_is_reverse: torch.Tensor,
    query_clause_attrs: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch broadcast — the path ``clause_mask`` replaces."""
    q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
    ic = item_clause_attrs.unsqueeze(0)  # [1, N, C, A_max]
    match = q == ic
    clause_pass = match.any(dim=-1)
    rev = clause_is_reverse.unsqueeze(0).unsqueeze(0)
    clause_pass = torch.where(rev, ~clause_pass, clause_pass)
    inactive = (query_clause_attrs == -1).unsqueeze(1)
    clause_pass = clause_pass | inactive
    return clause_pass.all(dim=-1)


@pytest.mark.parametrize("n", [256, 4096])
@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("a_max", [1, 4])
@pytest.mark.parametrize("pad_rate", [0.0, 0.3])
def test_clause_mask_matches_pure_torch(n, c, a_max, pad_rate):
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=40, pad_rate=pad_rate, seed=n + c)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=8, c=c, n_vocab=40, inactive_rate=0.2, seed=a_max + 1)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


def test_clause_mask_with_reverse_clauses():
    n, c = 1024, 3
    attrs = make_attrs(n, c=c, a_max=3, n_vocab=20, pad_rate=0.2, seed=42)
    is_reverse = torch.tensor([True, False, True], device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=20, inactive_rate=0.0, seed=43)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


def test_clause_mask_all_reverse():
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=15, pad_rate=0.1, seed=51)
    is_reverse = torch.ones(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=15, inactive_rate=0.0, seed=52)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


def test_clause_mask_inactive_query_passes_all():
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=10, pad_rate=0.0, seed=7)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = torch.full((4, c), -1, dtype=torch.long, device="cuda")

    out = clause_mask(attrs, is_reverse, q)
    expected = torch.ones(4, n, dtype=torch.bool, device="cuda")
    assert torch.equal(out, expected)


def test_clause_mask_a_max_one():
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=1, n_vocab=20, pad_rate=0.0, seed=61)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=20, inactive_rate=0.1, seed=62)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


def test_clause_mask_b_one():
    """Single-query batch — degenerate grid axis 0."""
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=20, pad_rate=0.1, seed=99)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=1, c=c, n_vocab=20, inactive_rate=0.2, seed=100)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


def test_clause_mask_n_smaller_than_block():
    """N below the fixed BLOCK_N=256 — single-tile path."""
    n, c = 64, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=15, pad_rate=0.1, seed=71)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=15, inactive_rate=0.0, seed=72)

    out = clause_mask(attrs, is_reverse, q)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("block_n, num_warps", [(128, 2), (512, 8), (1024, 4)])
def test_clause_mask_config_override(block_n, num_warps):
    """Non-default ``ClauseMaskConfig`` produces the same logical mask —
    proves the ``config=`` kwarg plumbs through ``_clause_mask_impl`` to
    the kernel launch."""
    n, c = 4096, 3
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=30, pad_rate=0.2, seed=81)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=8, c=c, n_vocab=30, inactive_rate=0.2, seed=82)

    cfg = ClauseMaskConfig(block_n=block_n, num_warps=num_warps)
    out = _clause_mask_impl(attrs, is_reverse, q, config=cfg)
    ref = _ref_mask(attrs, is_reverse, q)
    assert torch.equal(out, ref)
