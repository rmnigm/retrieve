"""bloom_hash builders — chunked vs loop-free equivalence, seed determinism.

The hash math is pinned (persisted ``bloom_sigs`` buffers must stay bit-valid
across refactors), so every assertion here is exact int64 equality — never a
tolerance."""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.filters import BloomFilter, bloom_hash
from retrieve.layers.filters.bloom_hash import (
    build_query_signatures,
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
    words_per_cluster,
)
from retrieve.layers.silvertorch import build_silvertorch
from tests.conftest import make_attrs, make_index, make_query_attrs
from tests.parity.conftest import make_probe_family

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
    st = build_silvertorch(
        embs,
        k=8,
        n_lists=8,
        n_probe=4,
        filter_mode="bloom",
        m_bits=M_BITS,
        k_hash=K_HASH,
        n_iter=2,
        item_clause_attrs=attrs,
    )
    assert "clause_salt" in st.state_dict()
    assert torch.equal(st.clause_salt, bf.clause_salt.cuda())
    assert torch.equal(st._query_bits(q), expected_q)
    # Registered without attributes → empty salt, derived on the fly at query time.
    st_noattr = build_silvertorch(
        embs, k=8, n_lists=8, n_probe=4, filter_mode="bloom", m_bits=M_BITS, k_hash=K_HASH, n_iter=2
    )
    assert st_noattr.clause_salt.numel() == 0
    assert torch.equal(st_noattr._query_bits(q), expected_q)


def test_build_transposed_sigs_bits():
    """Bit-level roundtrip of the transposed (cluster-major) index: for every (cluster,
    slot), row m of ``sigs_t`` carries exactly bit m of the slot item's row-wise
    signature (0 for padding slots). No shipped backend reads this layout since roadmap
    B4; it is kept for the Triton transposed-bloom kernel (O §8 TF-1)."""
    n_lists, max_size, w = 8, 90, 4  # non-multiple-of-64 max_size exercises the pad tail
    padded, _, _, n = make_probe_family(1, n_lists, max_size, 1, pad_rate=0.2)
    g = torch.Generator(device="cuda").manual_seed(3)
    ii = torch.iinfo(torch.int64)
    sigs = torch.randint(ii.min, ii.max, (n, w), generator=g, dtype=torch.int64, device="cuda")

    sigs_t = build_transposed_sigs(sigs, padded)
    wpc = words_per_cluster(max_size)
    assert sigs_t.shape == (w * 64, n_lists * wpc)

    m_bits = w * 64
    m = torch.arange(m_bits, device="cuda")
    for c in range(n_lists):
        for s in range(max_size):
            word = sigs_t[:, c * wpc + s // 64]  # [m_bits]
            actual = (word >> (s % 64)) & 1
            item = int(padded[c, s].item())
            if item < 0:
                expected = torch.zeros(m_bits, dtype=torch.int64, device="cuda")
            else:
                expected = (sigs[item][m // 64] >> (m % 64)) & 1
            assert torch.equal(actual, expected), f"cluster {c} slot {s}"
