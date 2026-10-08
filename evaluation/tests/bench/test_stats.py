"""G-stats: the report's statistics against answers known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

from bench import stats


def test_the_bootstrap_ci_of_a_constant_is_a_point():
    assert stats.median_ci([0.42] * 9) == (0.42, 0.42, 0.42)
    assert stats.mean_ci([0.7] * 1000) == pytest.approx((0.7, 0.7, 0.7), abs=1e-12)  # fp sum


def test_the_ci_brackets_the_estimate_and_is_reproducible():
    x = np.random.default_rng(0).normal(1.0, 0.1, 15).tolist()
    est, lo, hi = stats.median_ci(x)
    assert est == np.median(x) and lo < est < hi
    assert stats.median_ci(x) == (est, lo, hi)  # fixed generator seed


def test_mean_ci_over_many_queries_matches_the_normal_approximation():
    x = (np.arange(20_000) % 2).astype(float)  # recall 0.5, sd 0.5
    est, lo, hi = stats.mean_ci(x)
    half = 1.96 * 0.5 / np.sqrt(len(x))
    assert est == 0.5
    assert lo == pytest.approx(0.5 - half, abs=2e-3) and hi == pytest.approx(0.5 + half, abs=2e-3)


def test_a_paired_ratio_of_a_equal_two_b_is_two_and_differs():
    b = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2]
    ci = stats.paired_ratio_ci([2 * v for v in b], b)
    assert ci == (2.0, 2.0, 2.0) and stats.differs(ci)


def test_a_ratio_of_identical_arms_is_no_difference():
    a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2]
    assert not stats.differs(stats.paired_ratio_ci(a, list(a)))
    assert not stats.differs(stats.ratio_ci(a, list(a)))
    assert stats.differs(stats.ratio_ci([3 * v for v in a], a))


def test_a_paired_ratio_needs_equal_round_counts():
    with pytest.raises(ValueError, match="3 vs 2 rounds"):
        stats.paired_ratio_ci([1.0, 1.0, 1.0], [1.0, 1.0])


def test_interpolation_is_exact_on_a_linear_curve_and_never_extrapolates():
    curve = [(0.5 + 0.125 * i, 1.0 + 2.0 * i, f"n_probe={8 << i}") for i in (3, 0, 2, 1)]
    assert stats.at_recall(curve, 0.75) == {"latency": 5.0, "bracket": ("n_probe=32",) * 2}
    assert stats.at_recall(curve, 0.6875) == {
        "latency": 4.0,
        "bracket": ("n_probe=16", "n_probe=32"),
    }
    assert stats.at_recall(curve, 0.875) == {"latency": 7.0, "bracket": ("n_probe=64",) * 2}
    assert stats.at_recall(curve, 0.9) == {"latency": None, "reason": "not reached"}
    assert stats.at_recall(curve, 0.4) == {
        "latency": None,
        "reason": "first point already above (n_probe=8)",
    }
