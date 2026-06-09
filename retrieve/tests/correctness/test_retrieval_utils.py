"""``FullScanKNN`` and ``post_filter_topk`` — direct correctness tests.

``FullScanKNN`` is the brute-force oracle the rest of the suite leans on, so
its own correctness needs a non-circular check. ``post_filter_topk`` is a
small mask-and-count helper used downstream of the full-scan path; its
contract is straightforward but currently untested.
"""

from __future__ import annotations

import torch

from retrieve.layers.utils.retrieval import FullScanKNN, post_filter_topk
from tests.conftest import make_index, make_mask, make_query

N, D, B, K = 1024, 128, 8, 32


def _topk_reference(query: torch.Tensor, embs: torch.Tensor, k: int):
    scores = query @ embs.t()
    sc, ids = torch.topk(scores, k, dim=1)
    return ids, sc


def test_fullscan_matches_matmul_topk():
    embs = make_index(N, D)
    query = make_query(B, D)
    ref_ids, ref_scores = _topk_reference(query, embs, K)

    knn = FullScanKNN(k=K)
    knn.register_index(embs)
    ids, scores = knn(query)

    assert ids.shape == (B, K)
    assert scores.shape == (B, K)
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(scores, ref_scores, atol=1e-5)


def test_fullscan_with_mask_post_filters():
    """Mask path returns a *subset* of the unmasked top-K (post-filter, not in-loop)."""
    embs = make_index(N, D)
    query = make_query(B, D)
    mask = make_mask(B, N, pass_rate=0.3)

    knn = FullScanKNN(k=K)
    knn.register_index(embs)
    unmasked_ids, _ = knn(query)
    ids, _ = knn(query, mask=mask)

    assert ids.shape == (B, K)
    for b in range(B):
        for j in range(K):
            i = int(ids[b, j].item())
            if i == -1:
                continue
            assert int(unmasked_ids[b, j].item()) == i, (
                "mask is post-filter: surviving ids keep their original positions"
            )
            assert bool(mask[b, i].item()), f"row {b} pos {j}: id {i} fails mask"


def test_fullscan_with_candidate_ids_matches_gather_bmm():
    embs = make_index(N, D)
    query = make_query(B, D)
    g = torch.Generator(device="cuda").manual_seed(0)
    p = 64
    cand = torch.randint(0, N, (B, p), generator=g, dtype=torch.long, device="cuda")

    knn = FullScanKNN(k=K)
    knn.register_index(embs)
    ids, scores = knn(query, candidate_ids=cand)

    cand_embs = embs[cand]
    ref_scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)
    ref_topk_scores, ref_local = torch.topk(ref_scores, K, dim=1)
    ref_ids = cand.gather(1, ref_local)

    assert ids.shape == (B, K)
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(scores, ref_topk_scores, atol=1e-5)


def test_fullscan_candidate_ids_p_less_than_k():
    """When p < k the kernel returns only ``actual_k = p`` columns (no padding)."""
    embs = make_index(N, D)
    query = make_query(B, D)
    p = K // 2
    g = torch.Generator(device="cuda").manual_seed(2)
    cand = torch.randint(0, N, (B, p), generator=g, dtype=torch.long, device="cuda")

    knn = FullScanKNN(k=K)
    knn.register_index(embs)
    ids, scores = knn(query, candidate_ids=cand)
    # Current contract: forward returns actual_k columns when p < k.
    assert ids.shape == (B, p)
    assert scores.shape == (B, p)


def test_post_filter_topk_drops_failed():
    """Failed positions get id=-1; counts equals the per-row pass count."""
    g = torch.Generator(device="cuda").manual_seed(7)
    topk_ids = torch.randint(0, N, (B, K), generator=g, dtype=torch.long, device="cuda")
    mask = make_mask(B, N, pass_rate=0.5, seed=8)

    out_ids, counts = post_filter_topk(topk_ids, mask)

    assert out_ids.shape == (B, K)
    assert counts.shape == (B,)
    for b in range(B):
        keep = mask[b, topk_ids[b]]
        assert int(counts[b].item()) == int(keep.sum().item())
        # Surviving positions retain their id; failed positions become -1.
        for j in range(K):
            if keep[j]:
                assert out_ids[b, j].item() == topk_ids[b, j].item()
            else:
                assert out_ids[b, j].item() == -1


def test_post_filter_topk_all_pass():
    g = torch.Generator(device="cuda").manual_seed(9)
    topk_ids = torch.randint(0, N, (B, K), generator=g, dtype=torch.long, device="cuda")
    mask = torch.ones(B, N, dtype=torch.bool, device="cuda")

    out_ids, counts = post_filter_topk(topk_ids, mask)
    assert torch.equal(out_ids, topk_ids)
    assert torch.equal(counts, torch.full((B,), K, dtype=torch.int64, device="cuda"))


def test_post_filter_topk_all_fail():
    g = torch.Generator(device="cuda").manual_seed(11)
    topk_ids = torch.randint(0, N, (B, K), generator=g, dtype=torch.long, device="cuda")
    mask = torch.zeros(B, N, dtype=torch.bool, device="cuda")

    out_ids, counts = post_filter_topk(topk_ids, mask)
    assert (out_ids == -1).all()
    assert torch.equal(counts, torch.zeros(B, dtype=torch.int64, device="cuda"))
