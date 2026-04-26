"""Triton ``fused_matmul_topk`` vs ``(q @ x.T).masked_fill(...).topk(k)``."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.linr.fused_matmul_topk import fused_matmul_topk
from tests.conftest import make_index, make_mask, make_query
from tests.parity.conftest import assert_topk_matches


def _ref(query, item_embs, k, mask=None):
    scores = query @ item_embs.t()
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    topk_scores, topk_ids = torch.topk(scores, k, dim=1)
    return topk_ids, topk_scores


@pytest.mark.parametrize("b,n,d,k", [(1, 1024, 64, 16), (16, 16_384, 128, 200)])
def test_no_mask_matches_pure_torch(b, n, d, k):
    embs = make_index(n, d)
    query = make_query(b, d)

    out_ids, out_scores = fused_matmul_topk(query, embs, k)
    ref_ids, ref_scores = _ref(query, embs, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


@pytest.mark.parametrize("b,n,d,k", [(16, 16_384, 128, 200)])
@pytest.mark.parametrize("pass_rate", [0.05, 0.5])
def test_with_mask_matches_pure_torch(b, n, d, k, pass_rate):
    embs = make_index(n, d)
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pass_rate)

    out_ids, out_scores = fused_matmul_topk(query, embs, k, mask=mask)
    ref_ids, ref_scores = _ref(query, embs, k, mask=mask)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)
