"""Triton ``codesigned_probe_score`` (+ bloom) vs ``retrieve.ops.reference`` (the pure-torch
phase 2+3) on the compact CSR probe layout, plus two private oracles where independence from the
shared twin is the point: a per-row loop that concatenates the probed clusters (it shares no
slot arithmetic with either implementation), and the row-wise bloom subset test (the transposed
index must answer exactly what the row-wise signatures do)."""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing.quantize import quantize_int8, quantize_int8_global
from retrieve.ops import reference
from retrieve.ops.triton.codesigned_probe_score import (
    DEFAULT_CONFIG,
    CodesignedProbeScoreConfig,
    _codesigned_probe_score_impl,
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from tests.conftest import make_index, make_query
from tests.parity.conftest import (
    POISON,
    ProbeLayout,
    assert_topk_equal,
    make_bloom,
    make_probe_family,
    poison_empty,
)


def _scored(b=4, n_lists=32, max_size=100, n_probe=6, d=64, seed=7):
    lay = make_probe_family(b, n_lists, max_size, n_probe, seed=seed)
    codes, global_scale = quantize_int8_global(make_index(lay.n, d))
    return make_query(b, d), lay, codes, global_scale


def _args(lay: ProbeLayout, codes):
    return lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm


def _oracle(query, lay: ProbeLayout, codes, global_scale, k, keep=None):
    """Per row: the probed clusters' sorted positions concatenated in probe order, scored with
    the same dequant expression, top-k over a ``width``-long row padded with ``-inf``."""
    q_codes, q_scales = quantize_int8(query)
    out_s, out_i = [], []
    for r in range(query.shape[0]):
        pos = torch.cat(
            [
                torch.arange(int(lay.cluster_offsets[c]), int(lay.cluster_offsets[c + 1]))
                for c in lay.probe_ids[r].tolist()
            ]
        ).cuda()
        s = (codes[pos].float() @ q_codes[r].float()) * q_scales[r] * global_scale
        if keep is not None:
            s = torch.where(keep(r, pos), s, float("-inf"))
        s = torch.cat([s, torch.full((lay.width - pos.numel(),), float("-inf"), device="cuda")])
        ids = torch.cat([lay.sort_perm[pos], torch.full_like(s[pos.numel() :], -1).long()])
        v, i = s.topk(k)
        out_s.append(v)
        out_i.append(torch.where(torch.isfinite(v), ids[i], -1))
    return torch.stack(out_i), torch.stack(out_s)


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k", [(64, 40, 4, 64, 8), (128, 700, 16, 128, 32)]
)
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_no_filters_matches_ref(n_lists, max_size, n_probe, d, k, b):
    query, lay, codes, gs = _scored(b, n_lists, max_size, n_probe, d)
    out = codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width)
    ref = reference.codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width)
    assert_topk_equal(*out, *ref)
    assert_topk_equal(*out, *_oracle(query, lay, codes, gs, k))


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k", [(64, 80, 4, 64, 16), (128, 700, 16, 128, 32)]
)
@pytest.mark.parametrize("b", [1, 16])
def test_codesigned_with_bloom_matches_ref(n_lists, max_size, n_probe, d, k, b):
    query, lay, codes, gs = _scored(b, n_lists, max_size, n_probe, d)
    qpos, bt, sigs, qb = make_bloom(lay.n, b)
    out = codesigned_probe_score_bloom(query, *_args(lay, codes), qpos, bt, gs, k, lay.width)
    ref = reference.codesigned_probe_score_bloom(
        query, *_args(lay, codes), qpos, bt, gs, k, lay.width
    )
    assert_topk_equal(*out, *ref)

    def rowwise(r, pos):
        return ((qb[r] & sigs[pos]) == qb[r]).all(-1)

    assert_topk_equal(*out, *_oracle(query, lay, codes, gs, k, keep=rowwise))
    assert torch.isinf(out[1]).any() and torch.isfinite(out[1]).any(), "bloom regime not hit"


def test_config_override_matches_default():
    """Plumbing: a different ``config`` reaches the launch (tiles of 32 vs 128 lanes, so the
    cluster-aligned tiling differs) and produces identical ids / scores, with and without
    bloom."""
    query, lay, codes, gs = _scored()
    qpos, bt, _, _ = make_bloom(lay.n, query.shape[0])
    cfg_a = CodesignedProbeScoreConfig(block_p=32, num_warps=4)
    cfg_b = CodesignedProbeScoreConfig(block_p=128, num_warps=8)
    for bloom in ({}, {"query_bit_positions": qpos, "bloom_transposed": bt}):
        a = _codesigned_probe_score_impl(
            query, *_args(lay, codes), gs, 8, lay.width, **bloom, config=cfg_a
        )
        b = _codesigned_probe_score_impl(
            query, *_args(lay, codes), gs, 8, lay.width, **bloom, config=cfg_b
        )
        assert_topk_equal(*a, *b)


@pytest.mark.parametrize("with_bloom", [False, True])
def test_empty_score_buffer_does_not_leak(monkeypatch, with_bloom):
    """The ``[B, width]`` score buffer is ``torch.empty``; the kernel must write every slot: a
    dot, or ``-inf`` for a bloom reject, a lane of the ``-inf`` tail past the row's items.
    Poisoned allocator, hit count, exact parity."""
    query, lay, codes, gs = _scored(seed=11)
    b, k = query.shape[0], 8
    qpos, bt, _, _ = make_bloom(lay.n, b)
    if with_bloom:
        ref = reference.codesigned_probe_score_bloom(
            query, *_args(lay, codes), qpos, bt, gs, k, lay.width
        )
        hits = poison_empty(monkeypatch, (b, lay.width))
        out = codesigned_probe_score_bloom(query, *_args(lay, codes), qpos, bt, gs, k, lay.width)
    else:
        ref = reference.codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width)
        hits = poison_empty(monkeypatch, (b, lay.width))
        out = codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width)
    assert hits, "the score buffer no longer comes from torch.empty — the poison never ran"
    assert not (out[1] == POISON).any(), "poison leaked into top-K: a slot went unwritten"
    assert_topk_equal(*out, *ref)


def test_all_pass_query_bloom_equals_no_bloom():
    """A query with no set bits (every position ``-1``) is a subset of every item signature: the
    bloom op must equal the no-bloom op bit for bit (catches ``HAS_QB`` keyed wrongly)."""
    query, lay, codes, gs = _scored()
    qpos, bt, _, _ = make_bloom(lay.n, query.shape[0])
    k = 16
    out = codesigned_probe_score_bloom(
        query, *_args(lay, codes), torch.full_like(qpos, -1), bt, gs, k, lay.width
    )
    assert_topk_equal(*out, *codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width))


def test_row_alone_equals_row_in_batch_and_id_relabel():
    """Row-local reduction (the query's int8 scale is per row): each row run alone equals its
    row in the batch; relabelling the original ids (``sort_perm``) relabels the returned ids
    and leaves every score unchanged. At ``k = width`` so no tie run is cut by the K
    boundary."""
    query, lay, codes, gs = _scored()
    k = lay.width
    ids, scores = codesigned_probe_score(query, *_args(lay, codes), gs, k, lay.width)
    for bi in range(query.shape[0]):
        one = codesigned_probe_score(
            query[bi : bi + 1], lay.probe_ids[bi : bi + 1], lay.cluster_offsets, codes,
            lay.sort_perm, gs, k, lay.width,
        )  # fmt: skip
        assert_topk_equal(*one, ids[bi : bi + 1], scores[bi : bi + 1])
    relabel = torch.randperm(lay.n, generator=torch.Generator().manual_seed(9)).cuda()
    r_ids, r_scores = codesigned_probe_score(
        query, lay.probe_ids, lay.cluster_offsets, codes, relabel[lay.sort_perm], gs, k, lay.width
    )
    assert_topk_equal(torch.where(ids >= 0, relabel[ids.clamp_min(0)], -1), scores, r_ids, r_scores)


@pytest.mark.parametrize("r", [0, 1])
def test_across_tile_cutoff(r):
    """A probed cluster of ``2·block_p + r`` items (``r`` in {0, 1}: a full last tile and a
    one-lane last tile) next to a one-item cluster, so the cluster-aligned tiling and the slot
    arithmetic straddle a tile boundary; read from the kernel's shipped ``block_p``."""
    bp = DEFAULT_CONFIG.block_p
    sizes = torch.tensor([2 * bp + r, 1, 5], device="cuda")
    assert sizes[0] % bp == r
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), sizes.cumsum(0)])
    n = int(offsets[-1])
    lay = ProbeLayout(
        torch.tensor([[0, 1], [1, 0]], device="cuda"), offsets,
        torch.randperm(n, device="cuda"), int(sizes[0] + sizes[2]), n,
    )  # fmt: skip
    codes, gs = quantize_int8_global(make_index(n, 64))
    query = make_query(2, 64)
    out = codesigned_probe_score(query, *_args(lay, codes), gs, 32, lay.width)
    assert_topk_equal(
        *out, *reference.codesigned_probe_score(query, *_args(lay, codes), gs, 32, lay.width)
    )
    assert_topk_equal(*out, *_oracle(query, lay, codes, gs, 32))


def test_degenerate_rows_give_exact_sentinels():
    """A row whose probed clusters are all empty is ``(-1, -inf)`` in every slot; a row whose
    only item is one probed singleton has it in slot 0 and ``(-1, -inf)`` after it."""
    sizes = torch.tensor([0, 0, 1, 9], device="cuda")
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), sizes.cumsum(0)])
    n = int(offsets[-1])
    sort_perm = torch.randperm(n, device="cuda")
    lay = ProbeLayout(torch.tensor([[0, 1], [1, 2]], device="cuda"), offsets, sort_perm, 10, n)
    codes, gs = quantize_int8_global(make_index(n, 64))
    k = 8
    ids, scores = codesigned_probe_score(make_query(2, 64), *_args(lay, codes), gs, k, lay.width)
    assert torch.equal(ids[0], torch.full((k,), -1, device="cuda"))
    assert torch.equal(ids[1, 1:], torch.full((k - 1,), -1, device="cuda"))
    assert ids[1, 0].item() == sort_perm[0].item() and torch.isfinite(scores[1, 0])
    tail = torch.cat([scores[0], scores[1, 1:]])
    assert torch.equal(tail, torch.full_like(tail, float("-inf")))
