"""The compaction kernels' ordering contract (plan L3, deterministic-compaction.md §5 gates 1-2).

``clause_compact`` / ``bloom_compact`` return each row's surviving ids in ascending item order —
``torch.equal`` to ``ops.reference`` on ids *and* counts — and the same call returns the same
tensors ten launches running and in a fresh process. The reference's tail past ``counts[b]`` is
the argsort's and the kernels' is ``-1``, so rows are compared with both tails set to ``-1``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from retrieve.ops import reference
from retrieve.ops.triton.bloom_compact import BloomCompactConfig, _bloom_compact_impl, bloom_compact
from retrieve.ops.triton.clause_compact import (
    ClauseCompactConfig,
    _clause_compact_impl,
    clause_compact,
)
from tests.parity.conftest import make_bloom, make_exact

_NS = [64, 4096, 100_003]  # one tile, many tiles, a ragged last tile


def _canon(ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(ids.shape[1], device=ids.device)[None, :] < counts[:, None]
    return torch.where(valid, ids, torch.full_like(ids, -1))


def assert_compact_equal(out, ref) -> None:
    assert torch.equal(out[1], ref[1]), f"counts: {out[1].tolist()} vs {ref[1].tolist()}"
    assert torch.equal(_canon(*out), _canon(*ref))


@pytest.mark.parametrize("n", _NS)
@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("c, a_max", [(1, 1), (2, 2), (4, 4)])
def test_clause_compact_order_matches_reference(n, reverse, c, a_max):
    attrs, rev, q = make_exact(n, 8, c=c, a_max=a_max, reverse=reverse)
    assert_compact_equal(clause_compact(attrs, rev, q), reference.clause_compact(attrs, rev, q))


@pytest.mark.parametrize("n", _NS)
@pytest.mark.parametrize("m_bits", [256, 512, 1024])
def test_bloom_compact_order_matches_reference(n, m_bits):
    *_, sigs, qb = make_bloom(n, 8, m_bits=m_bits)
    assert_compact_equal(bloom_compact(qb, sigs), reference.bloom_compact(qb, sigs))


@pytest.mark.parametrize("block_n, num_warps", [(128, 2), (256, 8), (1024, 4)])
def test_order_holds_under_every_tile_config(block_n, num_warps):
    attrs, rev, q = make_exact(20_011, 4, c=2, a_max=2, reverse="mixed")
    cfg = ClauseCompactConfig(block_n, num_warps)
    assert_compact_equal(
        _clause_compact_impl(attrs, rev, q, config=cfg), reference.clause_compact(attrs, rev, q)
    )
    *_, sigs, qb = make_bloom(20_011, 4, m_bits=1024)
    cfg = BloomCompactConfig(block_n, num_warps)
    assert_compact_equal(
        _bloom_compact_impl(qb, sigs, config=cfg), reference.bloom_compact(qb, sigs)
    )


_INPUTS = """
from tests.parity.conftest import make_bloom, make_exact
attrs, rev, q = make_exact(200_003, 8, c=4, a_max=4, reverse="mixed")
*_, sigs, qb = make_bloom(200_003, 8, m_bits=1024)
"""

_FRESH_PROCESS = f"""
import sys, torch
from retrieve.ops.triton.bloom_compact import bloom_compact
from retrieve.ops.triton.clause_compact import clause_compact
{_INPUTS}
torch.save((clause_compact(attrs, rev, q), bloom_compact(qb, sigs)), sys.argv[1])
"""


def test_launch_to_launch_identity(tmp_path):
    """The check that would have caught the atomic-base ordering: ten launches in this process and
    one in a fresh interpreter all return the same ids and counts."""
    ns: dict = {}
    exec(_INPUTS, ns)
    attrs, rev, q, sigs, qb = ns["attrs"], ns["rev"], ns["q"], ns["sigs"], ns["qb"]
    first = (clause_compact(attrs, rev, q), bloom_compact(qb, sigs))
    for _ in range(9):
        again = (clause_compact(attrs, rev, q), bloom_compact(qb, sigs))
        for a, b in zip(first, again):
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    out = tmp_path / "fresh.pt"
    subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, str(out)],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    fresh = torch.load(out)
    for a, b in zip(first, fresh):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
