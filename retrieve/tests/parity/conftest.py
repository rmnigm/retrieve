"""Shared helpers for kernel parity tests."""

from __future__ import annotations

import torch

from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import build_transposed_sigs
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.utils.quantize import quantize_int8
from tests.conftest import make_attrs, make_query_attrs


def make_probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1, seed=7):
    """Synthetic padded IVF layout + probed clusters for the two-kernel (cuda / cute)
    backends: ``padded [n_lists, max_size]`` holds each item id at most once (scattered
    via randperm, ``-1`` padding), ``flat = padded[probe_ids].reshape(b, -1)`` mirrors
    the layer's phase 1. Returns ``(padded, probe_ids, flat, n)``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = n_lists * max_size
    padded = torch.randperm(n, generator=g, device="cuda").reshape(n_lists, max_size)
    pad = torch.rand(n_lists, max_size, generator=g, device="cuda") < pad_rate
    padded[pad] = -1
    probe_ids = torch.randint(0, n_lists, (b, n_probe), generator=g, device="cuda")
    flat = padded[probe_ids].reshape(b, -1)
    return padded, probe_ids, flat, n


def make_bloom(n, b, padded, *, m_bits=512, k_hash=5):
    """Row-wise item signatures, their transposed (cluster-major) index over ``padded``,
    and query signatures: ``(sigs, sigs_t, qb)``."""
    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=k_hash, device=attrs.device)
    w = m_bits // 64
    sigs = build_signatures(attrs.long(), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    qb = build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w
    )
    sigs_t = build_transposed_sigs(sigs, padded)
    return sigs, sigs_t, qb


def make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8):
    """Item / query clause attrs + reverse flags. The small vocabulary keeps the
    predicate's pass rate high enough that top-K is not all ``-inf``; ``make_query_attrs``
    leaves ~20% of clauses inactive (``-1``). ``reverse="mixed"`` flips clause 0 only, so
    one call exercises both branches of the XOR."""
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab).long()
    q_attrs = make_query_attrs(b, c=c, n_vocab=n_vocab).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    if reverse == "mixed":
        rev[0] = True
    return attrs, rev, q_attrs


def ref_cps_phase23(
    query,
    flat_items,
    item_codes,
    global_scale,
    k,
    *,
    qb=None,
    bloom_sigs=None,
    item_clause_attrs=None,
    clause_is_reverse=None,
    query_clause_attrs=None,
):
    """Reference for the paper-faithful int8 ANN (SilverTorch phases 2+3): int8 × int8
    → int32 dot with one global scale + per-row query scale, optionally gated by a
    filter — the row-wise bloom subset test (``qb`` / ``bloom_sigs``) or the exact
    AND-of-OR clause predicate (``item_clause_attrs`` / ``clause_is_reverse`` /
    ``query_clause_attrs``: AND over clauses, OR over ``A_max`` within a clause, XOR
    with reverse, OR with the ``-1`` inactive sentinel). Computed in fp32 because the
    integer products fit in fp32 mantissa at this D — bit-identical to an int32
    accumulator. Shared by the Triton and CUDA parity suites; both CUDA filter kernels
    read a different layout (transposed index / cluster-major mask) but compute a
    boolean-identical predicate to these row-wise forms."""
    valid = flat_items >= 0
    safe = flat_items.clamp(min=0)

    keep = valid
    if qb is not None:
        probed_sigs = bloom_sigs[safe]
        match = (qb.unsqueeze(1) & probed_sigs) == qb.unsqueeze(1)
        keep = keep & match.all(dim=-1)
    if query_clause_attrs is not None:
        gathered = item_clause_attrs[safe]  # [B, P, C, A_max]
        keep = keep & clause_subset_match(gathered, query_clause_attrs.long(), clause_is_reverse)

    q_codes, q_scales = quantize_int8(query)
    codes = item_codes[safe].float()  # [B, P, D]
    scores = torch.einsum("bd,bpd->bp", q_codes.float(), codes)
    scores = scores * q_scales.unsqueeze(1) * global_scale
    scores = scores.masked_fill(~keep, float("-inf"))

    actual_k = min(k, scores.shape[1])
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = flat_items.gather(1, topk_local)
    return topk_ids, topk_scores


def assert_ids_equal_up_to_ties(
    ids_a: torch.Tensor,
    ids_b: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Assert two top-K id tensors agree, allowing permutation *within tied scores*.

    For the bit-exact backend comparisons the score tensor is the hard gate: the
    caller asserts ``torch.equal(scores_a, scores_b)`` first and passes that tensor
    here. Ids are one notch weaker on purpose — ``torch.topk``'s documentation says
    tie order is "not guaranteed stable across invocations", and ties genuinely occur
    in this data (int32 dots of ~±1e5 magnitude over P≈768 candidates land on roughly
    one tied pair per row). So the honest gate is: ids equal, or every mismatching
    position lies inside a run of equal scores whose id *multisets* match.

    ``scores`` comes from ``topk`` and is therefore sorted descending, so tie runs are
    contiguous; a run of length 1 makes the comparison exact equality at that slot."""
    assert ids_a.shape == ids_b.shape == scores.shape, "ids/scores shape mismatch"
    if torch.equal(ids_a, ids_b):
        return
    k = ids_a.shape[1]
    for bi in range(ids_a.shape[0]):
        row_s = scores[bi].tolist()
        row_a = ids_a[bi].tolist()
        row_b = ids_b[bi].tolist()
        if row_a == row_b:
            continue
        start = 0
        while start < k:
            end = start + 1
            while end < k and row_s[end] == row_s[start]:
                end += 1
            assert sorted(row_a[start:end]) == sorted(row_b[start:end]), (
                f"row {bi}: ids differ at positions [{start}, {end}) where the scores "
                f"are not tied (score {row_s[start]}): {row_a[start:end]} vs "
                f"{row_b[start:end]}"
            )
            start = end


def assert_topk_matches(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Assert two top-K implementations agree on finite-id sets and sorted scores.

    Tie-breaking on the id permutation can differ between backends, so we compare
    *sets* of finite-score ids per row and the *sorted* descending scores
    (with -inf replaced by 0 so ``allclose`` still works on padded rows).
    """
    b, k = out_ids.shape
    for bi in range(b):
        out_pairs = [
            (out_ids[bi, j].item(), out_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(out_scores[bi, j])
        ]
        ref_pairs = [
            (ref_ids[bi, j].item(), ref_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(ref_scores[bi, j])
        ]
        out_set = {p[0] for p in out_pairs}
        ref_set = {p[0] for p in ref_pairs}
        if out_set == ref_set:
            continue
        # Tensor-core matmul (`tl.dot`) and torch `@` differ in accumulator
        # order — score-tied items can swap at the K-th boundary. Allow that
        # provided each side's unique ids lie within `atol` of its own min.
        out_min = min(s for _, s in out_pairs) if out_pairs else float("-inf")
        ref_min = min(s for _, s in ref_pairs) if ref_pairs else float("-inf")
        for i in ref_set - out_set:
            s = next(sc for idx, sc in ref_pairs if idx == i)
            assert s <= ref_min + atol + rtol * abs(ref_min), (
                f"row {bi}: ref-only id {i} score={s:.6f} not at boundary {ref_min:.6f}"
            )
        for i in out_set - ref_set:
            s = next(sc for idx, sc in out_pairs if idx == i)
            assert s <= out_min + atol + rtol * abs(out_min), (
                f"row {bi}: out-only id {i} score={s:.6f} not at boundary {out_min:.6f}"
            )

    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    out_finite = torch.where(torch.isfinite(out_sorted), out_sorted, torch.zeros_like(out_sorted))
    ref_finite = torch.where(torch.isfinite(ref_sorted), ref_sorted, torch.zeros_like(ref_sorted))
    assert torch.allclose(out_finite, ref_finite, atol=atol, rtol=rtol)
