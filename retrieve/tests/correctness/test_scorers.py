"""``DotProductScorer`` — candidate-set rerank correctness."""

from __future__ import annotations

import torch

from retrieve.layers.utils.scorers import DotProductScorer
from tests.conftest import make_index, make_query

N, D, B, P = 1024, 128, 8, 32


def test_register_index_buffer_shape_dtype_device():
    embs = make_index(N, D)
    scorer = DotProductScorer().to("cuda")
    scorer.register_index(embs)
    assert scorer.item_embs.shape == (N, D)
    assert scorer.item_embs.dtype == embs.dtype
    assert scorer.item_embs.device == embs.device


def test_forward_matches_gather_bmm():
    embs = make_index(N, D)
    query = make_query(B, D)
    g = torch.Generator(device="cuda").manual_seed(0)
    cand = torch.randint(0, N, (B, P), generator=g, dtype=torch.long, device="cuda")

    scorer = DotProductScorer().to("cuda")
    scorer.register_index(embs)
    got = scorer(query, cand)

    expected = torch.einsum("bd,bpd->bp", query, embs[cand])
    assert got.shape == (B, P)
    assert torch.allclose(got, expected, atol=1e-5)


def test_forward_b_one():
    """Single-query path — bmm broadcasts cleanly."""
    embs = make_index(N, D)
    query = make_query(1, D)
    g = torch.Generator(device="cuda").manual_seed(1)
    cand = torch.randint(0, N, (1, P), generator=g, dtype=torch.long, device="cuda")

    scorer = DotProductScorer().to("cuda")
    scorer.register_index(embs)
    got = scorer(query, cand)
    assert got.shape == (1, P)
    expected = (query.squeeze(0) * embs[cand.squeeze(0)]).sum(dim=-1)
    assert torch.allclose(got.squeeze(0), expected, atol=1e-5)


def test_forward_p_zero():
    """Empty candidate set returns shape ``[B, 0]``."""
    embs = make_index(N, D)
    query = make_query(B, D)
    cand = torch.empty(B, 0, dtype=torch.long, device="cuda")

    scorer = DotProductScorer().to("cuda")
    scorer.register_index(embs)
    got = scorer(query, cand)
    assert got.shape == (B, 0)


def test_forward_negative_id_reads_last_row():
    """``candidate_ids = -1`` reads the last item via Python-style indexing.

    The scorer does not validate ``candidate_ids`` — callers that produce -1
    padding must mask out those positions post-hoc. This test pins the
    current contract so a stricter implementation is a deliberate change.
    """
    embs = make_index(N, D)
    query = make_query(B, D)
    cand = torch.full((B, 1), -1, dtype=torch.long, device="cuda")

    scorer = DotProductScorer().to("cuda")
    scorer.register_index(embs)
    got = scorer(query, cand)
    expected = (query * embs[-1].unsqueeze(0)).sum(dim=-1, keepdim=True)
    assert torch.allclose(got, expected, atol=1e-5)
