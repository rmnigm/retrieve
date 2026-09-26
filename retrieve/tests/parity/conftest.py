"""Shared helpers for kernel parity tests."""

from __future__ import annotations

import torch

from retrieve.indexing.bloom_hash import build_signatures, generate_seeds
from tests.conftest import assert_topk_id_sets_match, make_attrs, make_query_attrs


def make_probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1, seed=7):
    """Synthetic padded IVF layout + probed clusters, the shape a backend's phase 2 decodes
    (``test_official.py`` builds the CSR view from it too, so every backend scores the same
    candidates): ``padded [n_lists, max_size]`` holds each item id at most once (scattered
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


def make_bloom(n, b, *, m_bits=512, k_hash=5):
    """Row-wise item signatures and query signatures: ``(sigs, qb)``."""
    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=k_hash, device=attrs.device)
    w = m_bits // 64
    sigs = build_signatures(attrs.long(), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    qb = build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w
    )
    return sigs, qb


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


def assert_scores_match(out: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> None:
    """Slot-wise score comparison: the non-finite pattern must be identical (``isfinite``,
    ``isnan`` and the sign of every infinity), finite slots within ``atol + rtol·|ref|``. A
    failure prints the mismatch count, the first slots and both values."""
    assert out.shape == ref.shape, f"shape {tuple(out.shape)} vs {tuple(ref.shape)}"
    fin_o, fin_r = torch.isfinite(out), torch.isfinite(ref)
    bad = (fin_o != fin_r) | (out.isnan() != ref.isnan())
    bad |= out.isinf() & ref.isinf() & (out != ref)
    both = fin_o & fin_r
    bad |= both & ~torch.isclose(out, ref, atol=atol, rtol=rtol)
    if bad.any():
        idx = bad.nonzero()[:5]
        detail = ", ".join(
            f"{tuple(i.tolist())}: {out[tuple(i)].item()!r} vs {ref[tuple(i)].item()!r}"
            for i in idx
        )
        raise AssertionError(
            f"{int(bad.sum())} score slots differ (atol={atol}, rtol={rtol}); first: {detail}"
        )


def assert_topk_matches(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Assert two top-K implementations agree on finite-id sets and sorted scores.

    Tie-breaking on the id permutation can differ between backends, so every row goes through
    ``assert_topk_id_sets_match`` (sets of finite-score ids, boundary ties within ``atol``) and
    the *sorted* descending scores through ``assert_scores_match`` (exact non-finite pattern,
    finite values within tolerance). No default tolerance: each caller states its own.
    """
    for bi in range(out_ids.shape[0]):
        assert_topk_id_sets_match(
            out_ids, out_scores, ref_ids, ref_scores, bi, atol=atol, rtol=rtol
        )
    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    assert_scores_match(out_sorted, ref_sorted, atol=atol, rtol=rtol)


def assert_topk_equal(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
) -> None:
    """The bit-exact top-K gate: scores equal slot by slot (non-finite pattern included), ids
    equal up to permutation within tied scores."""
    assert_scores_match(out_scores, ref_scores, atol=0.0, rtol=0.0)
    assert_ids_equal_up_to_ties(out_ids, ref_ids, ref_scores)


POISON = 1e30


def poison_empty(monkeypatch, shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Patch ``torch.empty`` so every fp32 allocation of ``shape`` comes back filled with
    ``POISON`` (it outranks every real score, so an unwritten slot reaches top-K). Returns the
    list the patched allocator appends to: a test asserts it is non-empty, which fails if the
    kernel's score buffer moved to another allocator and the poison was never used."""
    real_empty = torch.empty
    hits: list[tuple[int, ...]] = []

    def poisoned(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if t.dtype == torch.float32 and tuple(t.shape) == shape:
            t.fill_(POISON)
            hits.append(shape)
        return t

    monkeypatch.setattr(torch, "empty", poisoned)
    return hits
