"""Triton ``int8_ann_fused`` vs gather + dequant + bmm + topk reference."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.silvertorch.int8_ann_fused import int8_ann_fused
from retrieve.layers.utils.quantize import quantize_int8
from tests.conftest import make_index, make_query
from tests.parity.conftest import assert_topk_matches


def _ref(query, codes, scales, positive_indices, k):
    cand_codes = codes[positive_indices].float()
    cand_scales = scales[positive_indices]
    scores = torch.einsum("bd,bpd->bp", query, cand_codes) * cand_scales
    topk_scores, topk_local = torch.topk(scores, k, dim=1)
    topk_ids = positive_indices.gather(1, topk_local)
    return topk_ids, topk_scores


@pytest.mark.parametrize("n,d,p,k", [(1024, 64, 128, 8), (8192, 128, 512, 32)])
@pytest.mark.parametrize("b", [1, 16])
def test_int8_ann_fused_matches_pure_torch(n, d, p, k, b):
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)

    g = torch.Generator(device="cuda").manual_seed(n + d + p + k + b)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.full((b,), p, dtype=torch.long, device="cuda")

    out_ids, out_scores = int8_ann_fused(query, codes, scales, pos, counts, k)
    ref_ids, ref_scores = _ref(query, codes, scales, pos, k)
    assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores)


def test_partial_counts_handled():
    """When per-row count < P, the kernel must respect the count."""
    n, d, p, k, b = 1024, 64, 128, 8, 4
    embs = make_index(n, d)
    codes, scales = quantize_int8(embs)
    query = make_query(b, d)

    g = torch.Generator(device="cuda").manual_seed(0)
    pos = torch.randint(0, n, (b, p), generator=g, device="cuda", dtype=torch.long)
    counts = torch.tensor([p, p // 2, 4, 1], dtype=torch.long, device="cuda")

    out_ids, out_scores = int8_ann_fused(query, codes, scales, pos, counts, k)
    # For each row, all returned ids (where finite) must come from the first
    # `count` entries of pos[row].
    for bi in range(b):
        valid_pool = set(pos[bi, : counts[bi].item()].tolist())
        for j in range(k):
            if torch.isfinite(out_scores[bi, j]):
                assert out_ids[bi, j].item() in valid_pool
