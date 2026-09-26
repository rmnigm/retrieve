"""Triton ``codesigned_probe_score_exact`` vs the same op in ``retrieve.ops.reference`` (the
pure-torch phase 2+3 with the exact predicate) on the compact CSR probe layout."""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing.quantize import quantize_int8_global
from retrieve.ops import reference
from retrieve.ops.triton.codesigned_probe_score import codesigned_probe_score
from retrieve.ops.triton.codesigned_probe_score_exact import (
    DEFAULT_CONFIG,
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
    codesigned_probe_score_exact,
)
from tests.conftest import make_index, make_query
from tests.parity.conftest import (
    POISON,
    ProbeLayout,
    assert_topk_equal,
    make_exact,
    make_probe_family,
    poison_empty,
)


def _exact_args(b=4, n_lists=32, max_size=100, n_probe=6, d=64, c=2, a_max=2, reverse="mixed"):
    """``(query, probe_ids, cluster_offsets, codes, sort_perm, attrs, rev, q_attrs, gs)`` and the
    layout; everything but ``k`` and ``width`` of the op."""
    lay = make_probe_family(b, n_lists, max_size, n_probe)
    codes, gs = quantize_int8_global(make_index(lay.n, d))
    attrs, rev, q_attrs = make_exact(lay.n, b, c=c, a_max=a_max, reverse=reverse)
    args = [make_query(b, d), lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm]
    return [*args, attrs, rev, q_attrs, gs], lay


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k,c,a_max",
    [
        (32, 40, 4, 64, 8, 1, 1),
        (64, 120, 8, 64, 16, 2, 2),
        (128, 700, 16, 128, 32, 3, 4),
        (128, 700, 16, 192, 32, 3, 4),
        (64, 300, 8, 768, 32, 2, 2),
    ],
)
@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("reverse", ["none", "mixed"])
def test_codesigned_exact_matches_ref(n_lists, max_size, n_probe, d, k, c, a_max, b, reverse):
    args, lay = _exact_args(b, n_lists, max_size, n_probe, d, c, a_max, reverse)
    out = codesigned_probe_score_exact(*args, k, lay.width)
    assert_topk_equal(*out, *reference.codesigned_probe_score_exact(*args, k, lay.width))


def test_codesigned_exact_inactive_query():
    """All clauses inactive (``-1``): the predicate is identically true, so the op equals the
    unfiltered ``codesigned_probe_score`` bit for bit."""
    args, lay = _exact_args()
    args[7] = torch.full_like(args[7], -1)
    k = 16
    out = codesigned_probe_score_exact(*args, k, lay.width)
    assert_topk_equal(*out, *codesigned_probe_score(*args[:5], args[8], k, lay.width))


def test_config_override_matches_default():
    """Plumbing: a different ``config`` (32- vs 128-lane cluster-aligned tiles) reaches the
    launch and produces identical ids / scores."""
    (query, probe, off, codes, perm, attrs, rev, q_attrs, gs), lay = _exact_args()
    res = [
        _codesigned_probe_score_exact_impl(
            query,
            probe,
            off,
            codes,
            perm,
            gs,
            8,
            lay.width,
            item_clause_attrs=attrs,
            clause_is_reverse=rev,
            query_clause_attrs=q_attrs,
            config=cfg,
        )  # fmt: skip
        for cfg in (
            CodesignedProbeScoreExactConfig(block_p=32, num_warps=4),
            CodesignedProbeScoreExactConfig(block_p=128, num_warps=8),
        )
    ]
    assert_topk_equal(*res[0], *res[1])


def test_empty_score_buffer_does_not_leak(monkeypatch):
    """The ``[B, width]`` score buffer is ``torch.empty``; the kernel must write every slot (a
    dot, or ``-inf`` for a predicate reject and the tail past the row's items). Poisoned
    allocator, hit count, exact parity."""
    args, lay = _exact_args()
    k, b = 8, args[0].shape[0]
    ref = reference.codesigned_probe_score_exact(*args, k, lay.width)
    hits = poison_empty(monkeypatch, (b, lay.width))
    out = codesigned_probe_score_exact(*args, k, lay.width)
    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out[1] == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    assert_topk_equal(*out, *ref)


def test_row_alone_equals_row_in_batch():
    """Row-local reduction: each row run alone (its query, probes and clause attrs) equals its
    row in the batch."""
    (query, probe, off, codes, perm, attrs, rev, q_attrs, gs), lay = _exact_args()
    k = 16
    ids, scores = codesigned_probe_score_exact(
        query, probe, off, codes, perm, attrs, rev, q_attrs, gs, k, lay.width
    )
    for bi in range(query.shape[0]):
        one = codesigned_probe_score_exact(
            query[bi : bi + 1], probe[bi : bi + 1], off, codes, perm, attrs, rev,
            q_attrs[bi : bi + 1], gs, k, lay.width,
        )  # fmt: skip
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])


@pytest.mark.parametrize("r", [0, 1])
def test_across_tile_cutoff(r):
    """A probed cluster of ``2·block_p + r`` items (a full and a one-lane last tile) beside a
    one-item cluster; ``block_p`` read from the kernel's shipped config."""
    bp = DEFAULT_CONFIG.block_p
    sizes = torch.tensor([2 * bp + r, 1, 5], device="cuda")
    assert sizes[0] % bp == r
    off = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), sizes.cumsum(0)])
    n = int(off[-1])
    lay = ProbeLayout(torch.tensor([[0, 1], [1, 0]], device="cuda"), off,
                      torch.randperm(n, device="cuda"), int(sizes[0] + sizes[2]), n)  # fmt: skip
    codes, gs = quantize_int8_global(make_index(n, 64))
    attrs, rev, q_attrs = make_exact(n, 2, reverse="mixed")
    args = [make_query(2, 64), lay.probe_ids, off, codes, lay.sort_perm, attrs, rev, q_attrs, gs]
    out = codesigned_probe_score_exact(*args, 32, lay.width)
    assert_topk_equal(*out, *reference.codesigned_probe_score_exact(*args, 32, lay.width))


def test_degenerate_rows_give_exact_sentinels():
    """A row probing only empty clusters is ``(-1, -inf)`` in every slot; a row whose only item
    passes (all clauses inactive) has it in slot 0 and ``(-1, -inf)`` after it."""
    sizes = torch.tensor([0, 0, 1, 9], device="cuda")
    off = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), sizes.cumsum(0)])
    n = int(off[-1])
    perm = torch.randperm(n, device="cuda")
    codes, gs = quantize_int8_global(make_index(n, 64))
    attrs, rev, q_attrs = make_exact(n, 2)
    q_attrs = torch.full_like(q_attrs, -1)
    probe = torch.tensor([[0, 1], [1, 2]], device="cuda")
    k = 8
    ids, scores = codesigned_probe_score_exact(
        make_query(2, 64), probe, off, codes, perm, attrs, rev, q_attrs, gs, k, 10
    )
    assert torch.equal(ids[0], torch.full((k,), -1, device="cuda"))
    assert torch.equal(ids[1, 1:], torch.full((k - 1,), -1, device="cuda"))
    assert ids[1, 0].item() == perm[0].item() and torch.isfinite(scores[1, 0])
    tail = torch.cat([scores[0], scores[1, 1:]])
    assert torch.equal(tail, torch.full_like(tail, float("-inf")))
