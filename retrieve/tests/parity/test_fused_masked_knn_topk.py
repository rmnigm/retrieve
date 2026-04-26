"""Triton ``fused_masked_knn_topk`` vs gather + bmm + topk reference.

The reference impl mirrors ``LiNR_V2._forward_prefilter``: gather the passing
rows into ``[B, P, D]``, score with bmm, top-K locally. Both V2 and this test
take ``(positive_indices, counts)`` directly — kernel never sees a bool mask.
The local helper here ``compact_mask``-s a random mask only to construct test
inputs.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.linr.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_index, make_mask, make_query
from tests.parity.conftest import assert_topk_matches


def _ref(query, item_embs, mask, k):
    counts = mask.sum(dim=1)
    p = int(counts.max().item())
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    reduced_ids = sorted_idx[:, :p]
    reduced_embs = item_embs[reduced_ids]
    scores = torch.bmm(query.unsqueeze(1), reduced_embs.transpose(1, 2)).squeeze(1)
    valid = torch.arange(p, device=query.device).unsqueeze(0) < counts.unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    topk_scores, topk_local = torch.topk(scores, k, dim=1)
    topk_ids = reduced_ids.gather(1, topk_local)
    return topk_ids, topk_scores


@pytest.mark.parametrize("b,n,d,k", [(1, 1024, 64, 16), (16, 16_384, 128, 200)])
@pytest.mark.parametrize("pass_rate", [0.05, 0.5])
def test_matches_pure_torch(b, n, d, k, pass_rate):
    embs = make_index(n, d)
    query = make_query(b, d)
    mask = make_mask(b, n, pass_rate=pass_rate)

    pos, counts = compact_mask(mask)
    out_ids, out_scores = fused_masked_knn_topk(query, embs, pos, counts, k)
    ref_ids, ref_scores = _ref(query, embs, mask, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)
