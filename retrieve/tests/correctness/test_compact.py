"""``compact_mask`` — direct correctness for the [B, N] bool → (ids, counts) helper.

Currently exercised only transitively through filter and LiNR tests; pinning
the contract directly keeps the helper safe under refactor.
"""

from __future__ import annotations

import torch

from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_mask


def _per_row_set_match(ids: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> None:
    b = mask.shape[0]
    for r in range(b):
        c = int(counts[r].item())
        got = set(ids[r, :c].tolist())
        expected = set(mask[r].nonzero(as_tuple=True)[0].tolist())
        assert got == expected, f"row {r}: got {got} vs {expected}"


def test_shape_and_counts_match_mask_sum():
    b, n = 8, 512
    mask = make_mask(b=b, n=n, pass_rate=0.3, seed=0)
    ids, counts = compact_mask(mask)

    assert counts.shape == (b,)
    assert counts.dtype == torch.int64
    assert torch.equal(counts, mask.sum(dim=1))

    # Full-width return: matches the bloom_compact / clause_compact contract.
    assert ids.shape == (b, n)
    assert ids.dtype == torch.int64


def test_valid_ids_match_mask_per_row():
    mask = make_mask(b=8, n=512, pass_rate=0.4, seed=1)
    ids, counts = compact_mask(mask)
    _per_row_set_match(ids, counts, mask)


def test_all_true_returns_full():
    b, n = 4, 256
    mask = torch.ones(b, n, dtype=torch.bool, device="cuda")
    ids, counts = compact_mask(mask)
    assert ids.shape == (b, n)
    assert torch.equal(counts, torch.full((b,), n, dtype=torch.int64, device="cuda"))
    for r in range(b):
        assert set(ids[r].tolist()) == set(range(n))


def test_all_false_returns_empty():
    b, n = 4, 256
    mask = torch.zeros(b, n, dtype=torch.bool, device="cuda")
    ids, counts = compact_mask(mask)
    # Full-width return: shape is [b, n] but counts is all-zero so the row
    # contents are unspecified.
    assert ids.shape == (b, n)
    assert torch.equal(counts, torch.zeros(b, dtype=torch.int64, device="cuda"))


def test_single_row_batch():
    mask = make_mask(b=1, n=256, pass_rate=0.5, seed=4)
    ids, counts = compact_mask(mask)
    assert ids.shape[0] == 1
    _per_row_set_match(ids, counts, mask)


def test_mixed_rows_padded_to_max():
    """Rows with fewer passing items are right-padded; counts bounds valid reads."""
    mask = torch.zeros(3, 16, dtype=torch.bool, device="cuda")
    mask[0, [1, 5]] = True  # 2 passing
    mask[1, [0, 2, 4, 6, 8, 10]] = True  # 6 passing
    mask[2, []] = True  # 0 passing

    ids, counts = compact_mask(mask)
    assert torch.equal(counts, torch.tensor([2, 6, 0], dtype=torch.int64, device="cuda"))
    # Full-width return: 16, not the row-max of 6.
    assert ids.shape == (3, 16)
    # Valid id sets per row.
    assert set(ids[0, :2].tolist()) == {1, 5}
    assert set(ids[1, :6].tolist()) == {0, 2, 4, 6, 8, 10}
    # Row 2 has nothing valid: counts[2]==0; positions past counts are unspecified
    # — we only check counts.
