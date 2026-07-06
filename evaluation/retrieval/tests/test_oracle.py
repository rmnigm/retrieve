"""CPU-only tests for the filtered-oracle padding semantics + cache fingerprint.

``compute_filtered_oracle`` brute-forces ``q @ E_t`` under an exact filter
mask. When the filter admits fewer than K_GT items, the losing top-k slots
tie at -inf and torch.topk would emit lowest-indexed junk (0, 1, 2, ...);
the oracle must rewrite those slots to -1 so they never score as real
ground truth against the algos' -1 padding. Skipped rows stay all -1.

``load_or_build_oracle`` caches to ``gt_topk_v3_<sweep>.pt`` dict blobs
keyed by a content fingerprint: same-shape content changes must recompute,
identical content must hit the cache, and legacy bare-tensor blobs must be
rebuilt (never migrated in-place).

Runs on CPU: the oracle only needs plain torch ops.
"""

from __future__ import annotations

import torch

from retrieval.oracle import compute_filtered_oracle, load_or_build_oracle

CPU = torch.device("cpu")


class _FixedMaskFilter:
    """Minimal FilterModule stand-in: admits a fixed per-item mask for
    every query (``evaluate_mask`` is all the oracle calls)."""

    def __init__(self, item_mask: torch.Tensor):
        self.item_mask = item_mask  # [N] bool

    def evaluate_mask(self, qa_narrow: torch.Tensor) -> torch.Tensor:
        return self.item_mask.unsqueeze(0).expand(qa_narrow.shape[0], -1)


def test_oracle_pads_short_rows_with_minus_one():
    # 3 items, filter admits only item 1 → with K_GT=2 every row is
    # [1, -1], not [1, 0] (index-0 junk from the -inf tie).
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    filt = _FixedMaskFilter(torch.tensor([False, True, False]))

    out = compute_filtered_oracle(
        item_embs, queries, qa, skip_mask=None, filter_mod=filt, K_GT=2, device=CPU
    )
    assert out.dtype == torch.long
    assert out.tolist() == [[1, -1], [1, -1]]


def test_oracle_skip_mask_rows_stay_minus_one():
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    filt = _FixedMaskFilter(torch.tensor([True, True, True]))
    skip_mask = torch.tensor([True, False])

    out = compute_filtered_oracle(
        item_embs, queries, qa, skip_mask=skip_mask, filter_mod=filt, K_GT=2, device=CPU
    )
    # Skipped row untouched; kept row ranked by score (item 1 > item 2).
    assert out[0].tolist() == [-1, -1]
    assert out[1].tolist() == [1, 2]


def test_oracle_k_gt_beyond_catalog_pads_tail():
    # K_GT=5 over a 3-item catalog: K_eff=3 real ids, tail stays -1.
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])  # distinct scores → order 0,1,2

    out = compute_filtered_oracle(
        item_embs, queries, None, skip_mask=None, filter_mod=None, K_GT=5, device=CPU
    )
    assert out.shape == (1, 5)
    assert out[0].tolist() == [0, 1, 2, -1, -1]


# ----- E6: content-fingerprint cache -------------------------------------------


def _build_oracle(gt_dir, item_embs: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """load_or_build_oracle with the unfiltered defaults the tests share."""
    return load_or_build_oracle(
        gt_dir,
        "s",
        2,
        item_embs=item_embs,
        queries=queries,
        qa_narrow_sweep=None,
        skip_mask=None,
        oracle_filter=None,
        device=CPU,
    )


def test_oracle_fingerprint_cache_hit_returns_stored_topk(tmp_path):
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])
    first = _build_oracle(tmp_path, item_embs, queries)
    assert first.tolist() == [[0, 1]]
    gt_path = tmp_path / "gt_topk_v3_s.pt"
    assert gt_path.exists()

    # Plant a sentinel topk in the cached blob: a true cache hit returns it
    # verbatim; a recompute would overwrite it with the real oracle.
    blob = torch.load(str(gt_path), map_location="cpu", weights_only=True)
    assert set(blob) == {"topk", "fingerprint", "k_gt"} and blob["k_gt"] == 2
    sentinel = torch.full_like(blob["topk"], 7)
    torch.save({**blob, "topk": sentinel}, str(gt_path))

    again = _build_oracle(tmp_path, item_embs, queries)
    assert torch.equal(again, sentinel)


def test_oracle_fingerprint_invalidates_on_content_change(tmp_path):
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])
    assert _build_oracle(tmp_path, item_embs, queries).tolist() == [[0, 1]]

    # Same shape/dtype, one mutated item embedding → fingerprint mismatch →
    # recompute (item 0 is zeroed out of the top-2).
    mutated = item_embs.clone()
    mutated[0] = 0.0
    assert _build_oracle(tmp_path, mutated, queries).tolist() == [[1, 2]]

    # Restore the original embeddings: the blob now carries the mutated
    # fingerprint, so this recomputes back to the original result...
    assert _build_oracle(tmp_path, item_embs, queries).tolist() == [[0, 1]]
    # ...and an unchanged rebuild is a cache hit (sentinel round-trip).
    gt_path = tmp_path / "gt_topk_v3_s.pt"
    blob = torch.load(str(gt_path), map_location="cpu", weights_only=True)
    sentinel = torch.full_like(blob["topk"], 7)
    torch.save({**blob, "topk": sentinel}, str(gt_path))
    assert torch.equal(_build_oracle(tmp_path, item_embs, queries), sentinel)


def test_oracle_legacy_bare_tensor_blob_recomputes(tmp_path):
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])
    gt_path = tmp_path / "gt_topk_v3_s.pt"
    torch.save(torch.full((1, 2), 9, dtype=torch.long), str(gt_path))

    out = _build_oracle(tmp_path, item_embs, queries)
    assert out.tolist() == [[0, 1]]  # recomputed, not the bare tensor

    # The legacy blob is replaced with the v3 dict format, not migrated.
    blob = torch.load(str(gt_path), map_location="cpu", weights_only=True)
    assert isinstance(blob, dict)
    assert torch.equal(blob["topk"], out)
    assert blob["k_gt"] == 2
