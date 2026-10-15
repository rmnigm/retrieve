"""The adapted code's compile gate: ``torch.library.opcheck`` on every ``stfork::``
op (schema, fake, autograd registration, AOT dispatch); ``SilverTorch(backend="official-fork")``
compiles ``fullgraph`` with 0 graph breaks in the default and ``reduce-overhead`` modes, with
``dynamic`` False and True (batch size varying), outputs ``torch.equal`` to eager; ``torch.export``
of the forward runs and equals eager. ``backend="official"`` keeps refusing to compile
(``test_official.py`` T7).
"""

from __future__ import annotations

import pytest
import torch

from retrieve.modules.silvertorch import OfficialConfig, SilverTorchBuilder
from retrieve.ops import official as of
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs, require_official
from tests.parity.conftest import make_probe_family

FORK = "official-fork"
N, D, C, K = 4096, 128, 2, 32


@pytest.fixture(autouse=True)
def _needs_fork():
    require_official(FORK)


# --- opcheck on every stfork op ----------------------------------------------------------


def _family(b=4, d=64):
    lay = make_probe_family(b, 64, 120, 8)
    sizes = lay.cluster_offsets.diff()
    codes = torch.randint(-127, 128, (lay.n, d), dtype=torch.int8, device="cuda")
    q_codes = torch.randint(-127, 128, (b, d), dtype=torch.int8, device="cuda")
    return lay, sizes, codes, q_codes


def _bloom(n=4096, c=2):
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=20, seed=41)
    fid, foff, fval = of.attrs_to_features(attrs)
    index, boff = torch.ops.stfork.bloom_index_build(fid, foff, fval, 10.0, 3, False)
    exprs = of.queries_to_expressions(make_query_attrs(4, c=c, n_vocab=20, seed=42))
    ks = torch.ones(len(exprs), dtype=torch.int64)
    _, (data, offsets) = torch.ops.stfork.parse_expression_query_batch(exprs, ks, 7, True, 5)
    return (fid, foff, fval), index, boff, exprs, data, offsets


def _cases():
    lay, sizes, codes, q_codes = _family()
    ids = lay.probe_ids.long().contiguous()
    length = sizes[ids].long().contiguous()
    offsets = lay.cluster_offsets.long().contiguous()
    width = lay.width
    feats, index, boff, exprs, data, plan_offsets = _bloom()
    sel_off = (torch.arange(4, device="cuda") * 512).view(4, 1).repeat(1, 3).contiguous()
    sel_len = torch.full((4, 3), 300, dtype=torch.int64, device="cuda")
    dplans = of.prepare_plans(exprs, 7, 5, 2, torch.device("cuda"), slots=4)
    partial = torch.ops.stfork.bloom_index_search_batch_return_partial_response(
        index, boff, data, plan_offsets, sel_off, sel_len, 3, 7, None
    )
    scores = torch.randn(4, 50, device="cuda")
    st = torch.ops.stfork
    common = (offsets, ids, length, codes, q_codes, width)
    return {
        "fused_kmean_ann": (st.fused_kmean_ann, common, {}),
        "fused_kmean_ann_with_partial_masks": (
            st.fused_kmean_ann_with_partial_masks,
            (
                offsets,
                sel_off.new_zeros(4, 3),
                sel_len,
                codes.new_zeros(4096, 64),
                q_codes,
                900,
                *partial,
            ),
            {},
        ),  # fmt: skip
        "fused_kmean_ann_with_partial_masks_multiple": (
            st.fused_kmean_ann_with_partial_masks_multiple,
            (
                q_codes,
                [offsets],
                [sel_off.new_zeros(4, 3)],
                [sel_len],
                [codes.new_zeros(4096, 64)],
                [900],
                [partial[0]],
                [partial[1]],
                [partial[2]],
            ),
            {},
        ),  # fmt: skip
        "bloom_index_build": (st.bloom_index_build, (*feats, 10.0, 3, False), {}),
        "bloom_index_build_fixed": (st.bloom_index_build_fixed, (*feats, 256, 3, False), {}),
        "parse_expression_query_batch": (
            st.parse_expression_query_batch,
            (exprs, torch.ones(len(exprs), dtype=torch.int64), 7, True, 5),
            {},
        ),
        "bloom_index_search_batch": (
            st.bloom_index_search_batch,
            (index, boff, data, plan_offsets, 3, 7, False),
            {},
        ),
        "bloom_index_search_batch_return_partial_response": (
            st.bloom_index_search_batch_return_partial_response,
            (index, boff, data, plan_offsets, sel_off, sel_len, 3, 7, None),
            {},
        ),
        "bloom_index_search_batch_return_partial_response_multiple": (
            st.bloom_index_search_batch_return_partial_response_multiple,
            ([index], [boff], data, plan_offsets, [sel_off], [sel_len], 3, 7, None, False),
            {},
        ),
        "bloom_index_search_batch_prepared": (
            st.bloom_index_search_batch_prepared,
            (index, boff, dplans.blob, dplans.slots, dplans.max_stack_size, 3, 7, False),
            {},
        ),
        "bloom_index_search_batch_return_partial_response_prepared": (
            st.bloom_index_search_batch_return_partial_response_prepared,
            (
                index,
                boff,
                dplans.blob,
                dplans.slots,
                dplans.max_stack_size,
                sel_off,
                sel_len,
                3,
                7,
                None,
                of.max_partial_columns(4, 3, 900),
            ),
            {},
        ),  # fmt: skip
        "generate_column_info_for_clusters": (
            st.generate_column_info_for_clusters,
            (sel_off, sel_len),
            {},
        ),
        "is_topk": (
            st.is_topk,
            (scores, torch.full((4,), 5, device="cuda"), torch.empty_like(scores)),
            {},
        ),
        "take_top_k_and_gather_from_main_and_fresh": (
            st.take_top_k_and_gather_from_main_and_fresh,
            (
                scores,
                scores[:, :10].contiguous(),
                torch.arange(200, device="cuda").view(4, 50),
                torch.arange(40, device="cuda").view(4, 10),
                8,
                ["x"],
                [torch.randn(200, 3, device="cuda")],
                [torch.randn(40, 3, device="cuda")],
            ),
            {},
        ),  # fmt: skip
        "probe_prep": (
            st.probe_prep,
            (make_query(4, 64), ids, offsets, sizes.long().contiguous()),
            {},
        ),
        "search": (
            st.search,
            (
                make_query(4, 64),
                ids,
                offsets,
                sizes.long().contiguous(),
                codes,
                lay.sort_perm,
                torch.tensor(0.01, device="cuda"),
                10,
                width,
            ),
            {},
        ),  # fmt: skip
    }


OPS = (
    "fused_kmean_ann", "fused_kmean_ann_with_partial_masks",
    "fused_kmean_ann_with_partial_masks_multiple", "bloom_index_build", "bloom_index_build_fixed",
    "parse_expression_query_batch", "bloom_index_search_batch",
    "bloom_index_search_batch_return_partial_response",
    "bloom_index_search_batch_return_partial_response_multiple",
    "bloom_index_search_batch_prepared",
    "bloom_index_search_batch_return_partial_response_prepared",
    "generate_column_info_for_clusters", "is_topk", "take_top_k_and_gather_from_main_and_fresh",
    "probe_prep", "search",
)  # fmt: skip


def test_every_stfork_op_is_checked():
    registered = {
        name.split("::")[1].split(".")[0]
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith("stfork::")
    }
    assert registered == set(OPS)


@pytest.mark.parametrize("name", OPS)
def test_opcheck(name):
    op, args, kwargs = _cases()[name]
    torch.library.opcheck(op, args, kwargs)


# --- the layer: compile fullgraph, 0 graph breaks, equal to eager; export ------------------


def _layer(mode):
    kw = {"k": K, "n_lists": 64, "n_probe": 8, "n_iter": 3, "backend": FORK,
          "official": OfficialConfig(score_path="int32")}  # fmt: skip
    if mode == "bloom":
        kw |= {"filter_mode": "bloom", "m_bits": 512, "k_hash": 5}
    elif mode == "exact":
        kw |= {"filter_mode": "exact"}
    b = SilverTorchBuilder(**kw).set_item_embeddings(make_index(N, D))
    if mode != "none":
        b.set_item_attributes(make_attrs(N, c=C, a_max=2))
    return b.build()


def _inputs(m, mode, bs, seed=0):
    q = make_query(bs, D, seed=seed)
    p = None if mode == "none" else m.prepare_queries(make_query_attrs(bs, c=C, seed=seed + 1))
    return q, p


@pytest.mark.parametrize("mode", ["none", "exact", "bloom"])
def test_no_graph_breaks(mode):
    m = _layer(mode)
    q, p = _inputs(m, mode, 8)
    torch._dynamo.reset()
    explain = torch._dynamo.explain(m)(q, p)
    assert explain.graph_break_count == 0, explain.break_reasons
    assert explain.graph_count == 1


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead"])
@pytest.mark.parametrize("mode", ["none", "exact", "bloom"])
def test_compiled_equals_eager(mode, compile_mode, dynamic):
    m = _layer(mode)
    torch._dynamo.reset()
    compiled = torch.compile(m, fullgraph=True, mode=compile_mode, dynamic=dynamic)
    with torch.inference_mode():
        for bs, seed in ((8, 0), (8, 5), (4, 9), (16, 13)):  # bs varies: dynamic shapes
            q, p = _inputs(m, mode, bs, seed)
            want = m(q, p)
            for _ in range(3):  # reduce-overhead records, then replays
                got = compiled(q, p)
            assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), (bs, seed)
    torch._dynamo.reset()


@pytest.mark.parametrize("mode", ["none", "exact", "bloom"])
def test_export(mode):
    m = _layer(mode)
    q, p = _inputs(m, mode, 8)
    exported = torch.export.export(m, (q, p))
    q2, p2 = _inputs(m, mode, 8, seed=3)
    got, want = exported.module()(q2, p2), m(q2, p2)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
