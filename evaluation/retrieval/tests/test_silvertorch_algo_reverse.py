"""SilvertorchAlgo wrapper: reverse-clause propagation.

Regression test for the bug where SilvertorchAlgo dropped clause_is_reverse
on the floor, causing Recall ≈ 0 on reverse-clause sweeps. See
docs/plans/silvertorch-reverse-clause-wrapper-fix.md.
"""

import torch
import pytest
from retrieval.algos import build_algorithm


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_silvertorch_clause_reverse_matches_full_scan(backend):
    """Build a tiny exact-clause silvertorch with one reverse clause and
    verify its top-K matches a brute-force filtered FullScan oracle."""
    torch.manual_seed(0)
    N, D, B, K = 256, 16, 4, 8
    n_lists, n_probe = 16, 16  # probe all → IVF is exact
    embs = torch.randn(N + 1, D, device="cuda")
    embs[0] = 0
    # 1 clause, 1 attribute per item, 4 possible values.
    item_attrs = torch.randint(0, 4, (N + 1, 1, 1), device="cuda")
    clause_is_reverse = torch.tensor([True], device="cuda")
    qa = torch.randint(0, 4, (B, 1), device="cuda")
    queries = torch.randn(B, D, device="cuda")

    algo = build_algorithm(
        "silvertorch",
        embs,
        k=K,
        filter_kind="clause",
        item_attrs_narrow=item_attrs,
        clause_is_reverse=clause_is_reverse,
        params={"n_lists": n_lists, "n_probe": n_probe, "n_iter": 3, "seed": 0},
        backend=backend,
    )
    ids, _ = algo(queries.contiguous(), qa.contiguous())

    # Brute-force reference: items where attr != query (reverse), top-K by dot.
    scores = queries @ embs.t()
    mask = item_attrs[:, 0, 0].unsqueeze(0) != qa[:, :1]
    scores = scores.masked_fill(~mask, float("-inf"))
    scores[:, 0] = float("-inf")
    ref_ids = scores.topk(K, dim=1).indices

    # Per-row set-equality (kernel and reference may disagree on tie order).
    assert set(ids[0].tolist()) == set(ref_ids[0].tolist())
