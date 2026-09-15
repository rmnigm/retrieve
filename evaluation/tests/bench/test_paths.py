"""``algos.PATHS`` is derived from ``retrieve.interfaces.DISPATCH`` (X §5); this pins the
derivation against the table the harness has always carried and the properties config.py
collapses on."""

from __future__ import annotations

from bench import algos as A
from retrieve.interfaces import DISPATCH

EXPECTED = {
    "linr_v1_filter_mask": {"none": "cublas", "clause": "cublas+{b}", "bloom": "cublas+{b}"},
    "linr_v2": {"none": None, "clause": "{b}", "bloom": "{b}"},
    "linr_v3": {"none": "{b}", "clause": "{b}", "bloom": "{b}"},
    "linr_v4": {"none": "cublas", "clause": "cublas+{b}", "bloom": "cublas+{b}"},
    "silvertorch": {"none": "{b}", "clause": "{b}", "bloom": "{b}"},
}


def test_paths_equal_the_derivation_of_dispatch():
    grid = {(a, f, b) for a in A.ALGOS for f in A.FILTER_KINDS for b in A.BACKENDS}
    assert set(A.PATHS) == grid
    for (algo, fk, backend), path in A.PATHS.items():
        if backend == "official":
            assert path == ("official" if algo == "silvertorch" else None)
            assert DISPATCH[A.ALGOS[algo].__name__]["official"] == path
            continue
        want = EXPECTED[algo][fk]
        assert path == (None if want is None else want.format(b=backend)), (algo, fk, backend)


def test_dispatch_covers_every_harness_algo():
    for algo, cls in A.ALGOS.items():
        assert set(DISPATCH[cls.__name__]) == set(A.BACKENDS), algo
    assert A.filter_backend("official") == "triton"
    assert [A.filter_backend(b) for b in ("triton", "torch")] == ["triton", "torch"]
