"""``gemv_scores`` — the single-query GEMV in cuBLAS ``gemv2N``'s accumulation order — and its use
in ``PostfilterKNN`` (kernels.md § gemv_scores)."""

from __future__ import annotations

import pytest
import torch

from retrieve.modules.knn import PostfilterKNN, calibration_rows
from retrieve.ops import reference
from retrieve.ops.triton import gemv_scores
from tests.conftest import make_index, make_query


@pytest.mark.parametrize("n,d", [(800_000, 128), (300_000, 256), (1_000, 64)])
def test_postfilter_scores_equal_cublas_whatever_the_calibration(n, d):
    """A single query scores ``torch.equal`` to cuBLAS on every table: through the GEMV where
    ``register_index`` found it equal, through cuBLAS otherwise."""
    knn = PostfilterKNN(k=10)
    knn.register_index(make_index(n, d))
    for seed in range(4):
        q = make_query(1, d, seed=seed)
        want = torch.mm(q.half(), knn.item_embs_t, out_dtype=torch.float32)
        assert torch.equal(knn.score(q), want)


def test_gemv_engages_where_cublas_runs_gemv2n():
    """At D 128 / N 800 k cuBLAS runs ``gemv2N`` (A100, CUDA 12.8), whose order the GEMV takes."""
    knn = PostfilterKNN(k=10)
    knn.register_index(make_index(800_000, 128))
    assert knn.gemv_exact


def test_gemv_scores_is_its_twin_on_gemv2n_shapes():
    embs_t = make_index(800_000, 128).half().t().contiguous()
    q = make_query(1, 128).half()
    assert torch.equal(gemv_scores(q, embs_t), reference.gemv_scores(q, embs_t))


def test_torch_backend_never_takes_the_gemv():
    knn = PostfilterKNN(k=10, backend="torch")
    knn.register_index(make_index(800_000, 128))
    assert not knn.gemv_exact


@pytest.mark.parametrize("n", [8, 1_000, 2**24 + 3, 30_000_001, 2**31 + 5])
def test_calibration_rows_stay_in_range_past_2_24(n):
    """The check's rows span ``[0, n - 1]`` exactly: a float32 ``linspace`` rounded ``n - 1`` up
    to ``n`` past 2^24 and gathered column ``N`` (a device assert on 30 M tables)."""
    rows = calibration_rows(n, torch.device("cuda"))
    assert rows.dtype == torch.int64
    assert rows[0].item() == 0 and rows[-1].item() == n - 1
    assert (rows[1:] > rows[:-1]).all() and (rows < n).all()
