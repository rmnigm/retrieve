"""CPU-only unit tests for retrieval quality metrics.

Two halves. The first locks the three semantic contracts downstream rows depend on:

- ``-1`` padding (on either candidate or target side) never counts as a hit;
- NDCG's IDCG clamps the ideal-hit count to ``min(num_targets, k)``;
- recall's denominator is ``num_targets``, not ``k`` (the per-row ``nt_k`` of tight
  filters).

The second is harness-v2's WP-1 gate (H §6): the device-accumulated running sums of
``metrics.accumulate`` / ``finalize`` equal the OLD harness's per-row means to 1e-9 on
random data. The reference below is the pre-v2 ``metrics.py`` (commit c75b481) copied
verbatim so the comparison survives C3's deletion of the old API.

Hand-built or seeded random tensors only; no GPU, no data files.
"""

from __future__ import annotations

import math

import pytest
import torch

from retrieval.metrics import (
    accumulate,
    accumulator,
    finalize,
    jaccard_at_k,
    ndcg_at_k,
    recall_at_k,
)

# ----- old contracts -------------------------------------------------------------


def test_hits_mask_ignores_padding():
    # Both sides carry -1 padding; -1 == -1 must NOT register as a hit.
    cand = torch.tensor([[5, -1, 3]])
    tgt = torch.tensor([[-1, 3]])
    nt = torch.tensor([1])

    # k=3 reaches the real hit (3); the -1s on both sides contribute nothing.
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(1.0)
    # k=2 sees only [5, -1]; candidate -1 vs target -1 is padding, not a hit.
    assert recall_at_k(cand, tgt, nt, k=2).item() == pytest.approx(0.0)


def test_ndcg_idcg_clamps_to_k():
    # 5 relevant targets but k=2: IDCG must be built from min(num_targets, k)=2
    # ideal hits, so a perfect top-2 scores exactly 1.0 (not < 1).
    cand = torch.tensor([[1, 2]])
    tgt = torch.tensor([[1, 2, 3, 4, 5]])
    nt = torch.tensor([5])
    assert ndcg_at_k(cand, tgt, nt, k=2).item() == pytest.approx(1.0)

    # Partial case: single hit at rank 2, one relevant target, k=2.
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1 (one ideal hit at rank 1).
    cand = torch.tensor([[9, 1]])
    tgt = torch.tensor([[1]])
    nt = torch.tensor([1])
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(cand, tgt, nt, k=2).item() == pytest.approx(expected)


def test_recall_denominator_is_num_targets_not_k():
    # One relevant target retrieved within k=3 → recall is 1/1, not 1/3.
    cand = torch.tensor([[7, 8, 9]])
    tgt = torch.tensor([[7, -1]])
    nt = torch.tensor([1])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(1.0)

    # Two relevant targets, one retrieved → 1/2, again independent of k.
    tgt = torch.tensor([[7, 5]])
    nt = torch.tensor([2])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(0.5)

    # Zero-target row: clamp(min=1) keeps the denominator finite → 0.0, not NaN.
    tgt = torch.tensor([[-1, -1]])
    nt = torch.tensor([0])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(0.0)


# ----- old harness reference (metrics.py @ c75b481, verbatim) ------------------------


def _old_hits_mask(candidate_ids, targets, k):
    topk = candidate_ids[:, :k]
    eq = topk.unsqueeze(2) == targets.unsqueeze(1)
    valid = (topk.unsqueeze(2) != -1) & (targets.unsqueeze(1) != -1)
    return (eq & valid).any(dim=2)


def _old_recall(candidate_ids, targets, num_targets, k):
    hits = _old_hits_mask(candidate_ids, targets, k)
    return hits.sum(dim=1).float() / num_targets.float().clamp(min=1)


def _old_precision(candidate_ids, targets, num_targets, k):
    return _old_hits_mask(candidate_ids, targets, k).sum(dim=1).float() / k


def _old_mrr(candidate_ids, targets, num_targets, k):
    hits = _old_hits_mask(candidate_ids, targets, k)
    found = hits.any(dim=1)
    rank = hits.float().argmax(dim=1) + 1
    return (1.0 / rank.float()) * found.float()


def _old_ndcg(candidate_ids, targets, num_targets, k):
    hits = _old_hits_mask(candidate_ids, targets, k)
    positions = torch.arange(1, k + 1, device=candidate_ids.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg = (hits.float() * discounts.unsqueeze(0)).sum(dim=1)
    ideal_hits = positions.unsqueeze(0) <= num_targets.unsqueeze(1).float().clamp(max=k)
    idcg = (ideal_hits.float() * discounts.unsqueeze(0)).sum(dim=1)
    return dcg / idcg.clamp(min=1e-8)


_OLD = {"recall": _old_recall, "precision": _old_precision, "mrr": _old_mrr, "ndcg": _old_ndcg}


def _old_means(chunks, ks):
    """Old ``accumulate_metrics`` + ``finalize_metrics``: per-row floats, then a plain mean."""
    accum: dict[str, list[float]] = {}
    for ids, tgt, nt in chunks:
        for k in ks:
            for name, fn in _OLD.items():
                accum.setdefault(f"{name}@{k}", []).extend(fn(ids, tgt, nt, k).cpu().tolist())
    return {key: sum(v) / len(v) for key, v in accum.items()}


def _random_chunks(g, *, n_chunks=7, b=16, k_max=40, t=12, n_items=300):
    """Ragged chunks of (ids [B, K_max] with -1 tails, targets [B, T] -1-padded, nt [B])."""
    chunks = []
    for i in range(n_chunks):
        bb = b if i % 3 else 5  # ragged last-chunk shapes as the quality stream produces
        ids = torch.stack([torch.randperm(n_items, generator=g)[:k_max] for _ in range(bb)])
        ids[:, k_max - 3 :][torch.rand(bb, 3, generator=g) < 0.3] = -1
        tgt = torch.randint(0, n_items // 4, (bb, t), generator=g)  # dense hits: small id range
        nt = torch.randint(0, t + 1, (bb,), generator=g)
        tgt[torch.arange(t).unsqueeze(0) >= nt.unsqueeze(1)] = -1
        chunks.append((ids, tgt, nt))
    return chunks


def test_running_sums_equal_old_per_row_means_fixed_targets():
    g = torch.Generator().manual_seed(0)
    ks = [5, 10, 40]
    chunks = _random_chunks(g)
    acc = accumulator(ks, "cpu")
    for ids, tgt, nt in chunks:
        accumulate(acc, ids, tgt, nt)
    new = finalize(acc)
    old = _old_means(chunks, ks)
    assert new["n"] == sum(ids.shape[0] for ids, _, _ in chunks)
    assert set(old) == {key for key in new if key != "n"}
    for key, v in old.items():
        assert abs(new[key] - v) <= 1e-9, (key, new[key], v)
    # The hit-dense data must actually exercise the metrics (not all-zero).
    assert 0.0 < old["recall@40"] < 1.0 and old["mrr@5"] > 0.0


def test_running_sums_equal_old_ranked_oracle_targets():
    # Oracle targets: the target set at k is the oracle's own top-k prefix and nt is its
    # non-(-1) count — the old harness's ``ot_k`` / ``nt_k`` per k (sweep.py::_run_quality).
    g = torch.Generator().manual_seed(1)
    ks = [4, 16, 40]
    chunks = _random_chunks(g, t=40, n_items=60)
    acc = accumulator(ks, "cpu")
    for ids, tgt, _ in chunks:
        accumulate(acc, ids, tgt, ranked=True)
    new = finalize(acc)
    old = {}
    for k in ks:
        sliced = [(ids, t[:, :k], (t[:, :k] != -1).sum(1).clamp(max=k)) for ids, t, _ in chunks]
        old.update(_old_means(sliced, [k]))
    for key, v in old.items():
        assert abs(new[key] - v) <= 1e-9, (key, new[key], v)


def test_num_targets_derived_when_omitted():
    g = torch.Generator().manual_seed(2)
    chunks = _random_chunks(g, n_chunks=2)
    a, b = accumulator([10], "cpu"), accumulator([10], "cpu")
    for ids, tgt, nt in chunks:
        accumulate(a, ids, tgt, nt)
        accumulate(b, ids, tgt)  # nt == count of non -1 targets by construction
    assert finalize(a) == finalize(b)


def test_finalize_empty_accumulator():
    out = finalize(accumulator([3], "cpu"))
    assert out == {"recall@3": 0.0, "ndcg@3": 0.0, "precision@3": 0.0, "mrr@3": 0.0, "n": 0}


def test_jaccard_at_k():
    a = torch.tensor([[1, 2, 3, 4], [5, 6, -1, -1], [-1, -1, -1, -1]])
    b = torch.tensor([[4, 3, 9, 1], [6, 5, -1, -1], [-1, -1, -1, -1]])
    # row 0: {1,2,3} ∩ {4,3,9} = {3} / 5 → 0.2 at k=3; row 1: identical sets → 1;
    # row 2: both empty → 1 by convention.
    assert jaccard_at_k(a, b, 3) == pytest.approx((0.2 + 1.0 + 1.0) / 3)
    assert jaccard_at_k(a, b, 4) == pytest.approx((3 / 5 + 1.0 + 1.0) / 3)
    assert jaccard_at_k(a, a, 4) == 1.0
