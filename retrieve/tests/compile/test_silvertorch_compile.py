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

from functools import partial

import pytest
import torch
from torch._dynamo.utils import counters

from retrieve.modules.silvertorch import SilverTorchBuilder
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
)
from tests.parity.conftest import assert_topk_equal

# (filter_mode, backend, D). The Triton kernels are shape-generic in D, so the cheap
# D=64 build covers them.
MODES = [
    ("none", "triton", 64),
    ("bloom", "triton", 64),
    ("exact", "triton", 64),
]


def _build(filter_mode, backend, *, n=512, d=64, n_lists=16, n_probe=4, k=8, c=2, a_max=2):
    embs = make_index(n, d)
    kw = {"k": k, "n_lists": n_lists, "n_probe": n_probe, "n_iter": 3, "backend": backend}
    if filter_mode == "bloom":
        kw.update(filter_mode="bloom", m_bits=512, k_hash=4)
    if filter_mode == "exact":
        kw.update(filter_mode="exact")
    b = SilverTorchBuilder(**kw).set_item_embeddings(embs)
    if filter_mode != "none":
        b.set_item_attributes(make_attrs(n, c=c, a_max=a_max))
    return b.build()


def _inputs(filter_mode, b, c, dim, seed):
    qa = make_query_attrs(b, c=c, seed=seed) if filter_mode != "none" else None
    return make_query(b, dim, seed=seed), qa


@pytest.mark.parametrize("filter_mode,backend,d", MODES)
def test_compiled_forward_matches_eager(filter_mode, backend, d):
    """Warm up the ``reduce-overhead`` module on one query, then replay the captured graph on
    two different queries: each equals eager bit for bit (scores ``torch.equal``, ids up to
    ties), and inductor skipped no cudagraph."""

    b, c = 4, 2
    eager = _build(filter_mode, backend, d=d)
    inputs = partial(_inputs, filter_mode, b, c, eager.item_codes.shape[1])
    torch._dynamo.reset()
    skips_before = int(counters["inductor"]["cudagraph_skips"])
    compiled = torch.compile(eager, dynamic=True, mode="reduce-overhead")
    for _ in range(3):
        compiled(*inputs(1))
    for seed in (21, 22):
        query, q_attrs = inputs(seed)
        # cudagraph-tree outputs are reclaimed on the next call — clone before comparing.
        out_ids, out_scores = (t.clone() for t in compiled(query, q_attrs))
        eager_ids, eager_scores = eager(query, q_attrs)
        assert_topk_equal(out_ids, out_scores, eager_ids, eager_scores)
    assert int(counters["inductor"]["cudagraph_skips"]) == skips_before


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
