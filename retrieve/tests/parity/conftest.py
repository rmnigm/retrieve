"""Shared helpers for kernel parity tests."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from retrieve.indexing.ivf import probe_width
from tests.conftest import assert_topk_id_sets_match, make_attrs, make_query_attrs


@dataclass(frozen=True)
class ProbeLayout:
    """A synthetic CSR probe, the layout every SilverTorch backend scores: ``probe_ids [B,
    n_probe]`` (distinct per row), ``cluster_offsets [n_lists + 1]``, ``sort_perm [N]`` (sorted
    position → original id, a random permutation), the compact ``width`` and ``n``."""

    probe_ids: torch.Tensor
    cluster_offsets: torch.Tensor
    sort_perm: torch.Tensor
    width: int
    n: int


def make_probe_family(b, n_lists, max_size, n_probe, *, empty_rate=0.1, seed=7):
    """Cluster sizes uniform in ``[0, max_size]`` with one cluster at ``max_size`` and about
    ``empty_rate`` of them empty, ``n_probe`` distinct clusters per row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    sizes = torch.randint(0, max_size + 1, (n_lists,), generator=g, device="cuda")
    sizes[torch.rand(n_lists, generator=g, device="cuda") < empty_rate] = 0
    sizes[0] = max_size
    offsets = torch.zeros(n_lists + 1, dtype=torch.long, device="cuda")
    offsets[1:] = sizes.cumsum(0)
    n = int(offsets[-1])
    probe_ids = torch.stack(
        [torch.randperm(n_lists, generator=g, device="cuda")[:n_probe] for _ in range(b)]
    )
    sort_perm = torch.randperm(n, generator=g, device="cuda")
    return ProbeLayout(probe_ids, offsets, sort_perm, probe_width(sizes, n_probe), n)


def make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8):
    """Item / query clause attrs + reverse flags. The small vocabulary keeps the predicate's pass
    rate high enough that top-K is not all ``-inf``; ``make_query_attrs`` leaves ~20% of clauses
    inactive (``-1``). ``reverse="mixed"`` flips clause 0 only, so one call exercises both
    branches of the XOR."""
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
    """Two top-K id tensors agree up to a permutation within tied scores. The caller asserts
    ``torch.equal(scores_a, scores_b)`` first and passes that tensor here; ``torch.topk``'s tie
    order is not guaranteed stable, and ties occur in this data, so the gate is: ids equal, or
    every mismatching position lies inside a run of equal scores whose id multisets match.
    ``scores`` is sorted descending (a top-k output), so tie runs are contiguous."""
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
    """Two top-K implementations agree on finite-id sets and sorted scores: every row through
    ``assert_topk_id_sets_match`` (sets of finite-score ids, boundary ties within ``atol``) and
    the sorted descending scores through ``assert_scores_match``. No default tolerance: each
    caller states its own."""
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


def poison_empty(
    monkeypatch, shape: tuple[int, ...], dtype: torch.dtype = torch.float32, value=POISON
) -> list[tuple[int, ...]]:
    """Patch ``torch.empty`` so every ``dtype`` allocation of ``shape`` comes back filled with
    ``value`` (the fp32 default outranks every real score, so an unwritten slot reaches top-K).
    Returns the list the patched allocator appends to: a test asserts it is non-empty, which
    fails if the kernel's output buffer moved to another allocator and the poison was never
    used."""
    hits: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        torch, "empty", partial(_poisoned_empty, torch.empty, shape, dtype, value, hits)
    )
    return hits


def _poisoned_empty(real_empty, target_shape, target_dtype, value, hits, *args, **kwargs):
    # ``target_shape`` / ``target_dtype``, not ``shape`` / ``dtype``: a caller's own
    # ``torch.empty(..., dtype=...)`` would otherwise collide with the bound parameter.
    t = real_empty(*args, **kwargs)
    if t.dtype == target_dtype and tuple(t.shape) == target_shape:
        t.fill_(value)
        hits.append(target_shape)
    return t
