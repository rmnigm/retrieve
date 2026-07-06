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

import pytest
import torch

from retrieval.algos import (
    ALGORITHMS,
    SUPPORTED_FILTER_KINDS,
    is_valid_combo,
    supports,
)
from retrieval.config import FilterSweepCfg
from retrieval.loaders import build_sweep_qa

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
