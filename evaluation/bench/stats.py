"""The report's statistics (docs/system/evaluation.md § Statistics): percentile-bootstrap
CIs on medians, means and ratios, and latency interpolated at a matched recall.

Pure numpy over plain lists, so the known-answer tests need no records. Every resample uses
one fixed generator seed: a report regenerated from the same records prints the same CIs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

B = 10_000
SEED = 20261008
LEVEL = 0.95
_CHUNK_ELEMS = 1 << 24  # bounds one resample block to 128 MiB of float64


def _boot(x: np.ndarray, stat: Callable[..., np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = len(x)
    rows = max(1, _CHUNK_ELEMS // n)
    out = [stat(x[rng.integers(0, n, (min(rows, B - i), n))], axis=1) for i in range(0, B, rows)]
    return np.concatenate(out)


def _ci(x: Sequence[float], stat) -> tuple[float, float, float] | None:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return None
    lo, hi = np.percentile(_boot(a, stat), [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    return float(stat(a)), float(lo), float(hi)


def median_ci(x: Sequence[float]) -> tuple[float, float, float] | None:
    """Median of the pooled units (per-window medians over seed x window) and its CI."""
    return _ci(x, np.median)


def mean_ci(x: Sequence[float]) -> tuple[float, float, float] | None:
    """Mean over units (per-query recall) and its CI. A resample's mean depends only on how
    often it draws each distinct value, so the counts are drawn as one multinomial over the
    distinct values: the same bootstrap distribution in O(B x distinct) rather than O(B x n)
    (per-query recall@k takes few distinct values)."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return None
    vals, cnt = np.unique(a, return_counts=True)
    draws = np.random.default_rng(SEED).multinomial(a.size, cnt / a.size, size=B)
    lo, hi = np.percentile(draws @ vals / a.size, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    return float(a.mean()), float(lo), float(hi)


def paired_ratio_ci(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float] | None:
    """``a[i] / b[i]`` per round (window i of A over window i of B, pooled over seeds), its
    median and the CI over (seed x round)."""
    if len(a) != len(b):
        raise ValueError(f"paired ratio over {len(a)} vs {len(b)} rounds")
    return median_ci(np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64))


def ratio_ci(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float] | None:
    """Unpaired: median(a) / median(b), each side resampled independently."""
    if not len(a) or not len(b):
        return None
    xa, xb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    ra = np.median(xa[rng.integers(0, len(xa), (B, len(xa)))], axis=1)
    rb = np.median(xb[rng.integers(0, len(xb), (B, len(xb)))], axis=1)
    lo, hi = np.percentile(ra / rb, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    return float(np.median(xa) / np.median(xb)), float(lo), float(hi)


def differs(ci: tuple[float, float, float] | None) -> bool:
    """The plan's noise gate: a ratio whose CI contains 1.0 is "no difference"."""
    return ci is not None and (ci[1] > 1.0 or ci[2] < 1.0)


def at_recall(points: Sequence[tuple[float, float, str]], target: float) -> dict:
    """Latency at ``recall == target`` on one curve of ``(recall, latency, label)`` points:
    piecewise-linear between the two adjacent points of the recall-sorted curve that bracket
    the target, never extrapolated. ``latency`` is ``None`` with a ``reason`` when the curve
    does not reach the target or starts above it."""
    pts = sorted(points)
    if not pts or pts[-1][0] < target:
        return {"latency": None, "reason": "not reached"}
    if pts[0][0] > target:
        return {"latency": None, "reason": f"first point already above ({pts[0][2]})"}
    exact = [p for p in pts if p[0] == target]
    if exact:
        _, t, label = min(exact, key=lambda p: p[1])
        return {"latency": t, "bracket": (label, label)}
    (r0, t0, l0), (r1, t1, l1) = next(
        (p, q) for p, q in zip(pts, pts[1:], strict=False) if p[0] < target < q[0]
    )
    return {"latency": t0 + (t1 - t0) * (target - r0) / (r1 - r0), "bracket": (l0, l1)}


__all__ = [
    "B",
    "SEED",
    "at_recall",
    "differs",
    "mean_ci",
    "median_ci",
    "paired_ratio_ci",
    "ratio_ci",
]
