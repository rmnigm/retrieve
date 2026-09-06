"""CPU-only tests for the sweep driver's pure helpers.

``build_sweep_qa`` (loaders.py) synthesises per-sweep query attributes from
the full ``qa_narrow`` tensor: inactive clauses are coded ``-1`` ("always
pass" / "no bits queried") and rows left with no live clause are skip-masked.
``is_valid_combo`` (algos/__init__.py) encodes the silvertorch
``n_probe <= n_lists`` constraint; ``supports`` reads the declarative
``SUPPORTED_FILTER_KINDS`` eligibility table. Locked here so the E1/E2/E3
refactors are falsifiable without a GPU.
"""

from __future__ import annotations

import polars as pl
import pytest
import torch

from retrieval.algos import (
    ALGORITHMS,
    SUPPORTED_FILTER_KINDS,
    is_valid_combo,
    supports,
)
from retrieval.config import FilterCfg, FilterSweepCfg
from retrieval.loaders import (
    build_sweep_qa,
    drop_legacy_padding_row,
    load_filter_assets,
    load_query_attrs,
)

# ----- build_sweep_qa -----------------------------------------------------


def test_build_sweep_qa_masks_inactive_clauses():
    qa = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out, skip = build_sweep_qa(
        FilterSweepCfg(name="s", active_clauses=[1]), "clause", qa, 4
    )
    assert out is not None and skip is not None
    # Inactive clauses (0, 2, 3) coded -1; active clause 1 passed through.
    assert (out[:, [0, 2, 3]] == -1).all()
    assert (out[:, 1] == qa[:, 1]).all()
    # Every row kept a live clause → nothing skipped.
    assert not skip.any()
    # The input tensor is cloned, never mutated in place.
    assert (qa == torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])).all()


def test_build_sweep_qa_skip_mask_all_inactive():
    # Row 0's only active clause holds -1 (no surviving narrow clause for
    # that user) → after masking, the row is all -1 and must be skipped.
    qa = torch.tensor([[1, -1], [3, 4]])
    out, skip = build_sweep_qa(
        FilterSweepCfg(name="s", active_clauses=[1]), "clause", qa, 2
    )
    assert out is not None and skip is not None
    assert skip.tolist() == [True, False]
    assert (out[0] == -1).all()


def test_build_sweep_qa_no_active_clauses_is_noop():
    # No active clauses (yambda full_scan) or a non-filter kind → (None, None).
    qa = torch.tensor([[1, 2]])
    sweep = FilterSweepCfg(name="full_scan", active_clauses=None)
    assert build_sweep_qa(sweep, "clause", qa, 2) == (None, None)
    sweep = FilterSweepCfg(name="s", active_clauses=[0])
    assert build_sweep_qa(sweep, "none", qa, 2) == (None, None)


def test_build_sweep_qa_out_of_range_clause_raises():
    qa = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="out of range"):
        build_sweep_qa(FilterSweepCfg(name="s", active_clauses=[5]), "clause", qa, 4)


def test_build_sweep_qa_missing_qa_narrow_raises():
    with pytest.raises(ValueError, match="needs qa_narrow"):
        build_sweep_qa(FilterSweepCfg(name="s", active_clauses=[0]), "clause", None, 4)


# ----- is_valid_combo -------------------------------------------------------


def test_is_valid_combo_silvertorch_probe_constraint():
    # n_probe > n_lists would assert inside the IVF — combo is invalid.
    assert not is_valid_combo("silvertorch", {"n_lists": 16, "n_probe": 32})
    # Boundary: probing every list is allowed (exact IVF).
    assert is_valid_combo("silvertorch", {"n_lists": 16, "n_probe": 16})
    assert is_valid_combo("silvertorch", {"n_lists": 32, "n_probe": 4})
    # Missing keys → nothing to check → valid.
    assert is_valid_combo("silvertorch", {})
    assert is_valid_combo("silvertorch", {"n_lists": 16})


def test_is_valid_combo_other_algos_always_valid():
    assert is_valid_combo("linr_v3", {"n_lists": 4, "n_probe": 100})
    assert is_valid_combo("linr_v1_filter_mask", {})


# ----- supports --------------------------------------------------------------


def test_supports_linr_v2_is_filter_only():
    # The compact-candidate path's source IS the filter — no unfiltered mode.
    assert not supports("linr_v2", "none")
    assert supports("linr_v2", "clause")
    assert supports("linr_v2", "bloom")


def test_supports_full_coverage_algos():
    for algo in ("triton_knn", "linr_v1_filter_mask", "linr_v3", "linr_v4", "silvertorch"):
        for kind in ("none", "clause", "bloom"):
            assert supports(algo, kind)


def test_supports_table_covers_registry():
    # Every registered algo has an eligibility row (and no stale extras),
    # so run_one_sweep's supports() gate can never KeyError on a valid algo.
    assert set(SUPPORTED_FILTER_KINDS) == set(ALGORITHMS)


# ----- load_query_attrs ------------------------------------------------------


def _write_eval_split(path, n_rows: int, n_clauses: int = 4):
    pl.DataFrame(
        {
            "query_attrs_narrow": [
                [u * n_clauses + c for c in range(n_clauses)] for u in range(n_rows)
            ]
        }
    ).write_parquet(path)


def test_load_query_attrs_trims_to_already_trimmed_queries(tmp_path):
    # The checkpoint path (goodreads) hands load_query_attrs the row count of
    # queries that queries_cache already trimmed to users_limit, while the
    # parquet always holds the full split. users_limit is a prefix, so the
    # attrs must be trimmed to the same prefix instead of raising.
    path = tmp_path / "eval_split.parquet"
    _write_eval_split(path, n_rows=50)

    qa = load_query_attrs(path, 10)

    assert qa is not None
    assert qa.shape == (10, 4)
    # Prefix, not a sample: row i is still user i.
    assert torch.equal(qa[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(qa[9], torch.tensor([36, 37, 38, 39]))


def test_load_query_attrs_untrimmed_is_identity(tmp_path):
    # The arxiv path passes the untrimmed count; the trim happens later in
    # apply_users_limit, so here the parquet must come back whole.
    path = tmp_path / "eval_split.parquet"
    _write_eval_split(path, n_rows=7)

    qa = load_query_attrs(path, 7)

    assert qa is not None
    assert qa.shape == (7, 4)


def test_load_query_attrs_too_few_rows_raises(tmp_path):
    # Fewer attrs than queries is a real regen-the-parquet error.
    path = tmp_path / "eval_split.parquet"
    _write_eval_split(path, n_rows=3)

    with pytest.raises(RuntimeError, match="regen eval_split.parquet"):
        load_query_attrs(path, 10)


def test_load_query_attrs_missing_parquet_returns_none(tmp_path):
    assert load_query_attrs(tmp_path / "nope.parquet", 10) is None


# ----- legacy [N+1] layout ---------------------------------------------------
#
# The Hub copies of arxiv-papers and goodreads-work-id predate commit
# 3b1b5b3 and ship 1-indexed tensors with a padding row at index 0. A1's
# first GPU attempt died on it (goodreads) / would have run silently wrong
# (arxiv), so the loader drops that row and then checks the row counts.


def test_drop_legacy_padding_row_attrs():
    legacy = torch.tensor([[[-1, -1]], [[3, 4]], [[5, 6]]])   # [N+1, C, A]
    out = drop_legacy_padding_row(legacy, kind="attrs", what="t")
    assert out.shape == (2, 1, 2)
    assert torch.equal(out, legacy[1:])


def test_drop_legacy_padding_row_embeddings():
    legacy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = drop_legacy_padding_row(legacy, kind="emb", what="t")
    assert out.shape == (2, 2)
    assert torch.equal(out, legacy[1:])


def test_drop_legacy_padding_row_leaves_modern_layout_alone():
    # Row 0 of a 0-indexed attrs tensor may legitimately be a real item.
    modern = torch.tensor([[[3, 4]], [[5, 6]]])
    assert torch.equal(drop_legacy_padding_row(modern, kind="attrs", what="t"), modern)
    # ... and an all-(-1) row that is NOT the padding row is only dropped
    # when it sits at index 0, which is what the caller's n_items check
    # then has to agree with.
    embs = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    assert torch.equal(drop_legacy_padding_row(embs, kind="emb", what="t"), embs)


def _attrs_file(tmp_path, tensor):
    p = tmp_path / "item_attrs_narrow.pt"
    torch.save(tensor, str(p))
    return FilterCfg(sweeps=[], attrs_path=str(p))


def test_load_filter_assets_drops_legacy_pad_row(tmp_path):
    legacy = torch.full((4, 2, 2), -1)
    legacy[1:] = 7
    attrs, _ = load_filter_assets(
        "clause", _attrs_file(tmp_path, legacy), tmp_path, torch.device("cpu"), n_items=3
    )
    assert attrs is not None and attrs.shape[0] == 3
    assert (attrs == 7).all()


def test_load_filter_assets_accepts_modern_layout(tmp_path):
    modern = torch.full((3, 2, 2), 7)
    attrs, _ = load_filter_assets(
        "clause", _attrs_file(tmp_path, modern), tmp_path, torch.device("cpu"), n_items=3
    )
    assert attrs is not None and attrs.shape[0] == 3


def test_load_filter_assets_raises_on_misalignment(tmp_path):
    # Neither N nor N+1-with-a-pad-row: a mask built from this would score
    # every query against the wrong items, so it must not load.
    wrong = torch.full((5, 2, 2), 7)
    with pytest.raises(RuntimeError, match="filter mask would be misaligned"):
        load_filter_assets(
            "clause", _attrs_file(tmp_path, wrong), tmp_path, torch.device("cpu"), n_items=3
        )
