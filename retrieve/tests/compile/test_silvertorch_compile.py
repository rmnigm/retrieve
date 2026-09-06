"""End-to-end ``torch.compile`` parity + graph-break check for SilverTorch.

After the codesigned kernels were rewrapped as ``@torch.library.triton_op``
the layer's docstring promise (`callers wrap with torch.compile`) becomes
actually true: dynamo treats both kernels as opaque ops, the layer's forward
captures into a single graph, and cudagraph_trees can fuse it. This test
asserts that promise for all three filter modes on the Triton backend (the
official backend is eager-only by contract — ``test_official.py`` T7 asserts
that it *refuses* to compile).
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.silvertorch import build_silvertorch
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)

# (filter_mode, backend, D). The Triton kernels are shape-generic in D, so the cheap
# D=64 build covers them.
MODES = [
    ("none", "triton", 64),
    ("bloom", "triton", 64),
    ("exact", "triton", 64),
]


def _build(filter_mode, backend, *, n=512, d=64, n_lists=16, n_probe=4, k=8, c=2, a_max=2):
    embs = make_index(n, d)
    kw = dict(k=k, n_lists=n_lists, n_probe=n_probe, n_iter=3, backend=backend)
    if filter_mode == "bloom":
        attrs = make_attrs(n, c=c, a_max=a_max)
        return build_silvertorch(
            embs,
            filter_mode="bloom",
            m_bits=512,
            k_hash=4,
            item_clause_attrs=attrs,
            **kw,
        )
    if filter_mode == "exact":
        attrs = make_attrs(n, c=c, a_max=a_max)
        return build_silvertorch(
            embs,
            filter_mode="exact",
            item_clause_attrs=attrs,
            **kw,
        )
    return build_silvertorch(embs, **kw)


@pytest.mark.parametrize("filter_mode,backend,d", MODES)
def test_compiled_forward_matches_eager(filter_mode, backend, d):
    """Compiled SilverTorch returns the same top-K as the eager module."""
    torch.manual_seed(0)
    b, c = 4, 2

    eager = _build(filter_mode, backend, d=d)
    query = make_query(b, eager.item_codes.shape[1])
    q_attrs = make_query_attrs(b, c=c) if filter_mode != "none" else None

    eager_ids, eager_scores = eager(query, q_attrs)

    compiled = torch.compile(eager, dynamic=True, mode="reduce-overhead")
    out_ids, out_scores = compiled(query, q_attrs)

    # Top-K parity: ids match exactly (same kernel, same dummies); scores
    # bit-identical because the @triton_op body is byte-for-byte the same
    # eager call.
    torch.testing.assert_close(out_ids, eager_ids)
    torch.testing.assert_close(out_scores, eager_scores)


@pytest.mark.parametrize("filter_mode,backend,d", MODES)
def test_no_graph_breaks_on_forward(filter_mode, backend, d):
    """``torch._dynamo.explain`` reports zero graph breaks on the forward.

    A non-zero count means the kernel host wrappers are still graph-breaking
    (e.g. ``@torch._dynamo.disable`` slipped back in) or the layer reintroduced
    a host sync (``.item()`` on global_scale, Optional Tensor branching).
    """
    eager = _build(filter_mode, backend, d=d)
    b, c = 4, 2
    query = make_query(b, eager.item_codes.shape[1])
    q_attrs = make_query_attrs(b, c=c) if filter_mode != "none" else None

    explanation = torch._dynamo.explain(eager.forward)(query, q_attrs)
    assert explanation.graph_break_count == 0, (
        f"filter_mode={filter_mode}, backend={backend}, d={d}: expected 0 graph "
        f"breaks, got {explanation.graph_break_count}\n{explanation}"
    )
