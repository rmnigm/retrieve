"""End-to-end ``torch.compile`` parity + graph-break check for SilverTorch.

After the codesigned kernels were rewrapped as ``@torch.library.custom_op``
the layer's docstring promise (`callers wrap with torch.compile`) becomes
actually true: dynamo treats both kernels as opaque ops, the layer's forward
captures into a single graph, and cudagraph_trees can fuse it. This test
asserts that promise for all three filter modes.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.silvertorch import build_silvertorch
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs


def _build(filter_mode, *, n=512, d=64, n_lists=16, n_probe=4, k=8, c=2, a_max=2):
    embs = make_index(n, d)
    kw = dict(k=k, n_lists=n_lists, n_probe=n_probe, n_iter=3)
    if filter_mode == "bloom":
        attrs = make_attrs(n, c=c, a_max=a_max)
        return build_silvertorch(
            embs, filter="bloom", m_bits=512, k_hash=4,
            item_clause_attrs=attrs, **kw,
        )
    if filter_mode == "exact":
        attrs = make_attrs(n, c=c, a_max=a_max)
        return build_silvertorch(
            embs, filter="exact",
            item_clause_attrs=attrs, **kw,
        )
    return build_silvertorch(embs, **kw)


@pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
def test_compiled_forward_matches_eager(filter_mode):
    """Compiled SilverTorch returns the same top-K as the eager module."""
    torch.manual_seed(0)
    b, c = 4, 2

    eager = _build(filter_mode)
    query = make_query(b, eager.item_codes.shape[1])
    q_attrs = make_query_attrs(b, c=c) if filter_mode != "none" else None

    eager_ids, eager_scores = eager(query, q_attrs)

    compiled = torch.compile(eager, dynamic=True, mode="reduce-overhead")
    out_ids, out_scores = compiled(query, q_attrs)

    # Top-K parity: ids match exactly (same kernel, same dummies); scores
    # bit-identical because the @custom_op body is byte-for-byte the same
    # eager call.
    torch.testing.assert_close(out_ids, eager_ids)
    torch.testing.assert_close(out_scores, eager_scores)


@pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
def test_no_graph_breaks_on_forward(filter_mode):
    """``torch._dynamo.explain`` reports zero graph breaks on the forward.

    A non-zero count means the kernel host wrappers are still graph-breaking
    (e.g. ``@torch._dynamo.disable`` slipped back in) or the layer reintroduced
    a host sync (``.item()`` on global_scale, Optional Tensor branching).
    """
    eager = _build(filter_mode)
    b, c = 4, 2
    query = make_query(b, eager.item_codes.shape[1])
    q_attrs = make_query_attrs(b, c=c) if filter_mode != "none" else None

    explanation = torch._dynamo.explain(eager.forward)(query, q_attrs)
    assert explanation.graph_break_count == 0, (
        f"filter={filter_mode}: expected 0 graph breaks, got "
        f"{explanation.graph_break_count}\n{explanation}"
    )
