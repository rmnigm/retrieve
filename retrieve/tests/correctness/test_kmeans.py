"""``KMeansTorch.fit`` is bit-for-bit reproducible run to run.

SilverTorch's whole IVF layout — centroids, cluster membership, the int8 codes' scale — is a
pure function of ``fit``'s output, so a non-reproducible ``fit`` makes every SilverTorch quality
number non-reproducible with it. Roadmap C4 measured exactly that: two seed-0 builds of the same
index differed by 1.8e-2 in the centroids, because the centroid update reduced with
``index_add_``'s floating-point atomics, whose summation order follows block scheduling. Quality
then only reproduced to ~1e-4, under the golden gate's 1e-6.

The reduction is now order-fixed (``KMeansTorch._cluster_sums``: integer ``bincount`` for the
counts, a float64 one-hot GEMM accumulated panel by panel for the sums). These tests are the
gate on that, and on the claim that the replacement is the same algorithm at the same
precision — the pre-C4 atomic reduction is kept here and compared against.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.utils.kmeans import KMeansTorch
from tests.conftest import make_index


def _atomic_cluster_sums(embs, assignments, n_lists):
    """The pre-C4 reduction, verbatim: ``index_add_`` with float atomics."""
    sums = torch.zeros(n_lists, embs.shape[1], dtype=torch.float32, device=embs.device)
    counts = torch.zeros(n_lists, dtype=torch.float, device=embs.device)
    sums.index_add_(0, assignments, embs.float())
    counts.index_add_(
        0, assignments, torch.ones(embs.shape[0], dtype=torch.float, device=embs.device)
    )
    return sums, counts


def _atomic_fit(km: KMeansTorch, embs):
    """``KMeansTorch.fit`` with the pre-C4 atomic reduction, everything else identical."""
    n = embs.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(km.seed)
    perm = torch.randperm(n, generator=g)[: km.n_lists]
    centroids = embs[perm].clone().float()
    chunk = max(1, min(n, 1 << 14))
    for _ in range(km.n_iter):
        assignments = km._assign_chunked(embs, centroids, chunk)
        new_sums, counts = _atomic_cluster_sums(embs, assignments, km.n_lists)
        non_empty = counts > 0
        centroids = torch.where(
            non_empty.unsqueeze(1),
            new_sums / counts.clamp(min=1).unsqueeze(1),
            centroids,
        )
    return centroids, km._assign_chunked(embs, centroids, chunk)


# (N, D, n_lists, n_iter). 200k/128/1024 is the gate size from roadmap C4; both N values sit
# above the 1 << 14 assignment chunk and are non-multiples of it, so both the assignment chunk
# tail and `_cluster_sums`' short final one-hot panel are exercised.
SIZES = [(200_000, 128, 1024, 4), (20_001, 16, 32, 4)]


@pytest.mark.parametrize("n,d,n_lists,n_iter", SIZES)
def test_fit_is_bitwise_reproducible(n, d, n_lists, n_iter):
    """Two ``fit`` calls with the same seed agree bit for bit — centroids and assignments."""
    embs = make_index(n, d)
    km = KMeansTorch(n_lists=n_lists, n_iter=n_iter, seed=0)

    c1, a1 = km.fit(embs)
    c2, a2 = km.fit(embs)

    assert torch.equal(a1, a2), (
        f"N={n}: assignments differ between two seed-0 fits at "
        f"{(a1 != a2).sum().item()} of {n} items"
    )
    assert torch.equal(c1, c2), (
        f"N={n}: centroids differ between two seed-0 fits, max abs "
        f"{(c1 - c2).abs().max().item():.3e} — the reduction is order-dependent again"
    )


def test_fit_is_deterministic_on_cpu():
    """Same on CPU: the reduction must not depend on the device's scheduling either."""
    embs = make_index(3_001, 16).cpu()
    km = KMeansTorch(n_lists=16, n_iter=3, seed=0)

    c1, a1 = km.fit(embs)
    c2, a2 = km.fit(embs)

    assert torch.equal(a1, a2)
    assert torch.equal(c1, c2)


def _inertia(embs, centroids, assignments):
    """Mean squared distance from each item to its centroid — the k-means objective."""
    total = 0.0
    for start in range(0, embs.shape[0], 1 << 14):
        end = min(start + (1 << 14), embs.shape[0])
        total += (embs[start:end].float() - centroids[assignments[start:end]]).pow(2).sum().item()
    return total / embs.shape[0]


def test_one_update_matches_the_atomic_reduction():
    """One centroid update from a fixed assignment: the two reductions agree to float32 noise.

    Holding the assignment fixed removes Lloyd's chaotic amplification, so this isolates the
    reduction itself — the float64 one-hot GEMM against the float32 atomic accumulation of the
    same values. Measured on the A100: max abs 3.3e-6 on sums of order 1e2 (the error is the
    *atomic* side's float32 accumulation; the GEMM side is exact well past float32).
    """
    embs = make_index(200_000, 128)
    assignments = torch.randint(0, 1024, (200_000,), device="cuda")

    new_sums, new_counts = KMeansTorch._cluster_sums(embs, assignments, 1024, 1 << 14)
    old_sums, old_counts = _atomic_cluster_sums(embs, assignments, 1024)

    assert torch.equal(new_counts, old_counts)
    torch.testing.assert_close(new_sums, old_sums, rtol=0, atol=1e-5)


def test_fit_quality_matches_atomic_reference():
    """The order-fixed ``fit`` clusters as well as the atomic one did.

    Centroids are deliberately *not* compared elementwise: Lloyd's iteration is chaotic, a
    single flipped assignment cascades, and the atomic reference is not even stable against
    itself (roadmap C4 measured 1.8e-2 run to run on real embeddings; on this synthetic index
    the two implementations were seen both 6e-8 and 2.1e-2 apart on different runs, purely by
    which side flipped first). What must hold is that the objective is unchanged — the
    reduction was replaced, not the algorithm. Measured relative difference: 5.4e-9.
    """
    embs = make_index(200_000, 128)
    km = KMeansTorch(n_lists=1024, n_iter=4, seed=0)

    new_c, new_a = km.fit(embs)
    old_c, old_a = _atomic_fit(km, embs)

    new_i, old_i = _inertia(embs, new_c, new_a), _inertia(embs, old_c, old_a)
    rel = abs(new_i - old_i) / old_i
    assert rel < 1e-3, (
        f"k-means objective moved: {new_i:.9f} vs atomic {old_i:.9f} (rel {rel:.3e}) — the "
        f"order-fixed reduction is not computing the same sums"
    )
