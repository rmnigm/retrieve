"""Addressing past 2³¹ elements (kernels.md § Addressing): one large case per overflow class,
each with a planted, fully predictable answer so a wrapped int32 offset shows up as a wrong
value or an illegal address rather than as noise.

- output axis, ``B·N ≥ 2³¹``: ``B = 144``, ``N = 16M`` (row 143 starts past 2³¹);
- item axis, ``N·row_stride ≥ 2³¹``: one ``[140M, 16]`` int64 table (``N·W = 2.24e9``), read as
  bloom signatures, as ``[N, 2, 8]`` clause attrs and as OPORP bits.

Skipped when the device lacks the free memory."""

from __future__ import annotations

import pytest
import torch

from retrieve.ops import triton as T

_GIB = 2**30


def _need(gib: float) -> None:
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < gib * _GIB:
        pytest.skip(f"needs {gib} GiB free, have {free / _GIB:.1f}")


@pytest.fixture(scope="module")
def output_axis():
    _need(48)
    b, n = 144, 16_000_000
    assert (b - 1) * n >= 2**31
    attrs = (torch.arange(n, device="cuda") % 7).view(n, 1, 1)
    qa = (torch.arange(b, device="cuda") % 7).view(b, 1)
    rev = torch.zeros(1, dtype=torch.bool, device="cuda")
    yield b, n, attrs, rev, qa


def test_output_axis_clause_mask(output_axis):
    b, n, attrs, rev, qa = output_axis
    mask = T.clause_mask(attrs, rev, qa)
    cls = torch.arange(n, device="cuda") % 7
    for row in (0, 3, b - 2, b - 1):
        assert torch.equal(mask[row], cls == row % 7), row
    del mask


def test_output_axis_compact_and_scorers(output_axis):
    """clause_compact's row base (scratch and output), then the two indirect scorers reading
    that ``[144, 16M]`` candidate buffer and writing ``[144, P]`` scores."""
    b, n, attrs, rev, qa = output_axis
    pos, counts = T.clause_compact(attrs, rev, qa)
    last = b - 1
    expect = torch.arange(last % 7, n, 7, device="cuda")
    assert counts[last] == expect.numel()
    assert torch.equal(pos[last, : expect.numel()], expect)
    assert (pos[last, expect.numel() :] == -1).all()

    # Planted winners among row `last`'s candidates: the four largest ids of its class.
    winners = expect[-4:].flip(0)
    embs = torch.zeros(n, 16, dtype=torch.float16, device="cuda")
    embs[winners, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda", dtype=torch.float16)
    q = torch.zeros(b, 16, dtype=torch.float16, device="cuda")
    q[:, 0] = 1
    ids, scores = T.fused_masked_knn_topk(q, embs, pos, counts, 4)
    assert torch.equal(ids[last], winners)
    assert torch.equal(scores[last], torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda"))
    del embs, ids, scores

    bits = torch.full((n, 1), -1, dtype=torch.int64, device="cuda")  # popcount 64
    bits[winners, 0] = torch.tensor([0, 1, 3, 7], device="cuda")  # popcount 0, 1, 2, 3
    qbits = torch.zeros(b, 1, dtype=torch.int64, device="cuda")
    ids, scores = T.oporp_1bit_match_topk_indirect(qbits, bits, 4, pos, counts)
    assert torch.equal(ids[last], winners)
    assert torch.equal(scores[last], torch.tensor([64.0, 62.0, 60.0, 58.0], device="cuda"))


@pytest.fixture(scope="module")
def item_axis():
    _need(24)
    n, w = 140_000_000, 16
    assert (n - 1) * w >= 2**31
    table = torch.zeros(n, w, dtype=torch.int64, device="cuda")
    marked = torch.tensor(
        [5, 2**31 // w + 3, n - 70, n - 2, n - 1], dtype=torch.int64, device="cuda"
    )
    table[marked, 0] = 1
    yield n, table, marked


def test_item_axis_bloom(item_axis):
    n, table, marked = item_axis
    qb = torch.zeros(2, 16, dtype=torch.int64, device="cuda")
    qb[0, 0] = 1  # row 1 has no bits: every item passes
    mask = T.bloom_match(qb, table)
    assert torch.equal(mask[0].nonzero().squeeze(1), marked)
    assert mask[1].all()
    del mask
    pos, counts = T.bloom_compact(qb[:1], table)
    assert counts.tolist() == [marked.numel()]
    assert torch.equal(pos[0, : marked.numel()], marked)
    assert (pos[0, marked.numel() :] == -1).all()


def test_item_axis_clause(item_axis):
    n, table, marked = item_axis
    attrs = table.view(n, 2, 8)  # clause 0, slot 0 holds the mark; everything else is 0
    rev = torch.zeros(2, dtype=torch.bool, device="cuda")
    qa = torch.tensor([[1, -1]], device="cuda")
    assert torch.equal(T.clause_mask(attrs, rev, qa)[0].nonzero().squeeze(1), marked)
    pos, counts = T.clause_compact(attrs, rev, qa)
    assert counts.tolist() == [marked.numel()]
    assert torch.equal(pos[0, : marked.numel()], marked)


def test_item_axis_oporp_full(item_axis):
    n, table, marked = item_axis
    qbits = torch.zeros(1, 16, dtype=torch.int64, device="cuda")
    qbits[0, 0] = 1  # Hamming 0 on the marked rows, 1 everywhere else
    ids, scores = T.oporp_1bit_match_topk_full(qbits, table, marked.numel())
    assert torch.equal(ids[0].sort().values, marked)
    assert (scores == 1024.0).all()
