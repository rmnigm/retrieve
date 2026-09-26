"""bloom_hash builders — chunked vs loop-free equivalence, seed determinism.

The hash math is pinned (persisted ``bloom_sigs`` buffers must stay bit-valid
across refactors), so every assertion here is exact int64 equality — never a
tolerance."""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing import bloom_hash
from retrieve.indexing.bloom_hash import (
    build_query_bit_positions,
    build_query_signatures,
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
)
from retrieve.modules import BloomFilter
from retrieve.modules.silvertorch import SilverTorchBuilder
from tests.conftest import make_attrs, make_index, make_query_attrs

M_BITS, K_HASH = 512, 5
WORD_COUNT = M_BITS // 64


def test_index_and_query_builders_agree_rowwise():
    """``build_signatures(attrs)[i] == build_query_signatures(attrs[i:i+1])[0]`` —
    the chunked (index-side) and loop-free (query-side) paths share one core and
    must agree exactly; this property was assumed but never asserted before."""
    attrs = make_attrs(n=257, c=3, a_max=4, n_vocab=100, pad_rate=0.3, seed=90)
    seeds = generate_seeds(K_HASH, device=attrs.device)
    sigs = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    for i in (0, 1, 128, 256):
        row = build_query_signatures(attrs[i : i + 1], seeds, M_BITS, K_HASH, WORD_COUNT)
        assert torch.equal(sigs[i], row[0]), f"row {i}: chunked != loop-free"


def test_chunked_build_crosses_batch_boundary(monkeypatch):
    """Force several chunks through the index-side loop (incl. a ragged tail);
    output must be bit-identical to the loop-free build of the same slab."""
    monkeypatch.setattr(bloom_hash, "_BUILD_SIGS_BATCH", 7)
    attrs = make_attrs(n=100, c=2, a_max=3, n_vocab=50, pad_rate=0.2, seed=91)
    seeds = generate_seeds(K_HASH, device=attrs.device)
    chunked = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    loop_free = build_query_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    assert torch.equal(chunked, loop_free)


def test_generate_seeds_deterministic_and_odd():
    """Fixed CPU generator → identical seeds on every call; multipliers are odd."""
    a = generate_seeds(7, device=torch.device("cuda"))
    b = generate_seeds(7, device=torch.device("cuda"))
    assert torch.equal(a, b)
    assert a.shape == (7, 2)
    assert bool((a & 1).eq(1).all())


# --- clause_salt as a registered buffer (roadmap B5 / official-integration plan §8 TF-2) ---


def _inline_salt_reference(attrs, seeds, m_bits, k_hash, word_count):
    """The pre-B5 computation: the per-clause salt built inline, per call, from the
    splitmix64 constants materialised as fresh device tensors. Kept here verbatim so
    the buffer path is pinned against the *previous* bits, not against itself."""
    n, c_dim, a_max = attrs.shape
    device = attrs.device
    clause_ids = (
        torch.arange(c_dim, dtype=torch.int64, device=device)
        .view(c_dim, 1)
        .expand(c_dim, a_max)
        .reshape(1, c_dim * a_max, 1)
    )
    salt = bloom_hash._mix64(
        clause_ids,
        torch.tensor(bloom_hash._SALT_C1, dtype=torch.int64, device=device),
        torch.tensor(bloom_hash._SALT_C2, dtype=torch.int64, device=device),
    )
    flat = attrs.reshape(n, c_dim * a_max)
    return bloom_hash._signature_batch(flat, flat != -1, seeds, m_bits, k_hash, word_count, salt)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_clause_salt_buffer_is_bit_identical_to_inline_salt(device):
    """``build_signatures(..., clause_salt=buffer)`` ≡ ``build_signatures(...)`` (salt
    derived on the fly) ≡ the pre-B5 inline computation, bit for bit, on both the
    index-side and the query-side builder. The hash core is plain tensor math, so the
    same check runs on CPU: the salt is a pure function of the clause index and
    carries no device-specific bits."""
    attrs = make_attrs(n=300, c=3, a_max=4, n_vocab=100, pad_rate=0.3, seed=92).to(device)
    seeds = generate_seeds(K_HASH, device=torch.device(device))
    salt = generate_clause_salt(attrs.shape[1], device=torch.device(device))
    assert salt.shape == (3,) and salt.dtype == torch.int64

    ref = _inline_salt_reference(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    with_buf = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT, clause_salt=salt)
    without = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    assert torch.equal(with_buf, ref)
    assert torch.equal(without, ref)

    q = attrs[:16, :, :1]  # query-side shape [B, C, 1]
    q_ref = _inline_salt_reference(q, seeds, M_BITS, K_HASH, WORD_COUNT)
    q_buf = build_query_signatures(q, seeds, M_BITS, K_HASH, WORD_COUNT, clause_salt=salt)
    assert torch.equal(q_buf, q_ref)


def test_clause_salt_is_device_independent():
    a = generate_clause_salt(5, device=torch.device("cpu"))
    b = generate_clause_salt(5, device=torch.device("cuda"))
    assert torch.equal(a, b.cpu())


def test_clause_salt_shape_mismatch_rejected():
    attrs = make_attrs(n=8, c=3, a_max=2, seed=93)
    seeds = generate_seeds(K_HASH, device=attrs.device)
    wrong = generate_clause_salt(4, device=attrs.device)
    with pytest.raises(ValueError, match="clause_salt must be"):
        build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT, clause_salt=wrong)


def test_clause_salt_registered_as_buffer_and_moves_with_module():
    """``BloomFilter`` and ``SilverTorch(filter_mode="bloom")`` register ``clause_salt``
    as a ``[C]`` int64 buffer (it appears in ``state_dict``), it follows ``.to()``, and
    the query signatures built through the buffer equal the standalone builder's."""
    attrs = make_attrs(n=256, c=2, a_max=2, seed=94)
    q = make_query_attrs(b=8, c=2, seed=95)

    bf = BloomFilter(m_bits=M_BITS, k_hash=K_HASH).to("cuda")
    bf.register_index(attrs)
    assert "clause_salt" in bf.state_dict()
    assert bf.clause_salt.shape == (2,) and bf.clause_salt.dtype == torch.int64
    assert bf.clause_salt.device.type == "cuda"
    expected_q = build_query_signatures(
        q.long().unsqueeze(-1), bf.hash_seeds, M_BITS, K_HASH, WORD_COUNT
    )
    assert torch.equal(bf._build_query_sigs(q), expected_q)
    mask_before = bf.evaluate_mask(q)
    bf_cpu = bf.cpu()
    assert bf_cpu.clause_salt.device.type == "cpu"
    assert torch.equal(bf_cpu.clause_salt, generate_clause_salt(2, torch.device("cpu")))
    bf_back = bf_cpu.cuda()
    assert bf_back.clause_salt.device.type == "cuda"
    assert torch.equal(bf_back.evaluate_mask(q), mask_before)

    embs = make_index(256, 64)
    kw = dict(k=8, n_lists=8, n_probe=4, n_iter=2)
    kw.update(filter_mode="bloom", m_bits=M_BITS, k_hash=K_HASH)
    st = SilverTorchBuilder(**kw).set_item_embeddings(embs).set_item_attributes(attrs).build()
    assert "clause_salt" in st.state_dict()
    assert torch.equal(st.clause_salt, bf.clause_salt.cuda())
    assert torch.equal(_pack_positions(st._query_bit_positions(q)), expected_q)
    # Registered without attributes → empty salt, derived on the fly at query time.
    st_noattr = SilverTorchBuilder(**kw).set_item_embeddings(embs).build()
    assert st_noattr.clause_salt.numel() == 0
    assert torch.equal(_pack_positions(st_noattr._query_bit_positions(q)), expected_q)


def _pack_positions(pos: torch.Tensor, m_bits: int = M_BITS) -> torch.Tensor:
    """``[B, n]`` set-bit positions (``-1`` = none) → ``[B, m_bits // 64]`` packed signature, by a
    scatter independent of ``bloom_hash``'s own pack."""
    b = pos.shape[0]
    grid = torch.zeros(b, m_bits + 1, dtype=torch.bool, device=pos.device)
    grid.scatter_(1, torch.where(pos >= 0, pos, m_bits), True)
    shifts = torch.arange(64, device=pos.device)
    return (grid[:, :m_bits].view(b, m_bits // 64, 64).long() << shifts).sum(-1)


def test_query_bit_positions_are_the_query_signature():
    """``build_query_bit_positions`` scattered is ``build_query_signatures`` bit for bit, with
    ``-1`` exactly at the slots of inactive (``-1``) clauses; the transposed scorer reads these
    positions instead of the packed signature."""
    q = make_query_attrs(b=32, c=3, inactive_rate=0.3, seed=96)
    seeds = generate_seeds(K_HASH, torch.device("cuda"))
    salt = generate_clause_salt(3, torch.device("cuda"))
    pos = build_query_bit_positions(q, seeds, M_BITS, K_HASH, clause_salt=salt)
    assert pos.shape == (32, 3 * K_HASH)
    inactive = (q == -1).repeat_interleave(K_HASH, dim=1)
    assert torch.equal(pos < 0, inactive)
    expected = build_query_signatures(
        q.unsqueeze(-1), seeds, M_BITS, K_HASH, WORD_COUNT, clause_salt=salt
    )
    assert torch.equal(_pack_positions(pos), expected)


def test_build_transposed_sigs_bits():
    """Bit-level roundtrip of the transposed index: bit ``s % 64`` of word ``s // 64`` of row
    ``m`` is bit ``m`` of ``sigs[s]``, and the pad past ``N`` (a non-multiple of 64) is 0."""
    n, w = 200, 4
    g = torch.Generator(device="cuda").manual_seed(3)
    ii = torch.iinfo(torch.int64)
    sigs = torch.randint(ii.min, ii.max, (n, w), generator=g, dtype=torch.int64, device="cuda")

    sigs_t = build_transposed_sigs(sigs)
    assert sigs_t.shape == (w * 64, 4)

    m = torch.arange(w * 64, device="cuda")
    for s in range(4 * 64):
        actual = (sigs_t[:, s // 64] >> (s % 64)) & 1
        if s >= n:
            expected = torch.zeros(w * 64, dtype=torch.int64, device="cuda")
        else:
            expected = (sigs[s][m // 64] >> (m % 64)) & 1
        assert torch.equal(actual, expected), f"slot {s}"
