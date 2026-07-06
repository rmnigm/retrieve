"""_PackedBitsKNN base — construction/registration invariants for OneBitKNN / SimHashKNN.

The behavioral spec for these layers is ``test_linr.py`` (GPU, both backends, cross-backend
strict equality). This file pins what the K4.2 base-class fold must not disturb and what the
GPU suite doesn't reach: constructor attrs, the ``k > n`` guard living in the base
``register_index``, buffer names and registration order (state_dict layout), the ``k_bits``
sentinel resolution — including the non-corrupting re-registration fix — and the
``pad_to_k=False`` column contract on the torch-eager candidates path. OPORP/SimHash
quantization and the torch-eager forward are device-agnostic torch, so small CPU tensors
suffice; the suite-wide CUDA gate in ``tests/conftest.py`` still applies at collection time.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.interfaces import RetrievalModule
from retrieve.layers.linr._bit_knn import _PackedBitsKNN
from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.linr.simhash_knn import SimHashKNN

N, D, B = 32, 128, 4


def make_cpu_embs(n: int = N, d: int = D, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    embs = torch.randn(n, d, generator=g)
    return embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)


def make_one_bit(k: int = 4, **kw) -> OneBitKNN:
    return OneBitKNN(k=k, backend="torch", **kw)


def make_simhash(k: int = 4, k_bits: int = D, **kw) -> SimHashKNN:
    return SimHashKNN(k=k, k_bits=k_bits, backend="torch", **kw)


class TestConstruction:
    def test_class_hierarchy(self):
        """Both bit layers share the base, which is a RetrievalModule."""
        assert issubclass(_PackedBitsKNN, RetrievalModule)
        assert issubclass(OneBitKNN, _PackedBitsKNN)
        assert issubclass(SimHashKNN, _PackedBitsKNN)

    def test_one_bit_ctor_attrs(self):
        m = OneBitKNN(k=5, seed=3, backend="torch", k_bits=64)
        assert (m.k, m.seed, m.backend, m.k_bits) == (5, 3, "torch", 64)

    def test_one_bit_k_bits_default_is_zero_sentinel(self):
        """Before register_index the sentinel is a plain int 0 (no Optional attr)."""
        m = OneBitKNN(k=5)
        assert m.k_bits == 0
        assert isinstance(m.k_bits, int)

    def test_simhash_ctor_attrs(self):
        m = SimHashKNN(k=5, k_bits=256, seed=3, backend="torch")
        assert (m.k, m.k_bits, m.seed, m.backend) == (5, 256, 3, "torch")


class TestRegisterIndex:
    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_k_gt_n_raises(self, make):
        """The k > n guard lives in the base register_index (full-scan topk has no pad tail)."""
        m = make(k=N + 1)
        with pytest.raises(ValueError, match="exceeds corpus size"):
            m.register_index(make_cpu_embs())

    def test_one_bit_buffer_names_and_order(self):
        """State-dict layout is frozen: item_bits, oporp_signs, oporp_perm — in that order."""
        m = make_one_bit()
        m.register_index(make_cpu_embs())
        assert [name for name, _ in m.named_buffers()] == [
            "item_bits",
            "oporp_signs",
            "oporp_perm",
        ]

    def test_simhash_buffer_names_and_order(self):
        """State-dict layout is frozen: item_bits, simhash_R — in that order."""
        m = make_simhash()
        m.register_index(make_cpu_embs())
        assert [name for name, _ in m.named_buffers()] == ["item_bits", "simhash_R"]

    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_item_bits_shape_and_d_total(self, make):
        m = make()
        m.register_index(make_cpu_embs())
        assert m.item_bits.dtype == torch.int64
        assert m.item_bits.shape == (N, D // 64)
        assert m.d_total == D


class TestKBitsSentinel:
    def test_zero_sentinel_resolves_to_d(self):
        m = make_one_bit(k_bits=0)
        m.register_index(make_cpu_embs())
        assert m.k_bits == D
        assert isinstance(m.k_bits, int)  # plain int after registration (Dynamo stability)

    def test_explicit_k_bits_kept(self):
        m = make_one_bit(k_bits=64)
        m.register_index(make_cpu_embs())
        assert m.k_bits == 64
        assert m.item_bits.shape == (N, 1)

    def test_re_register_re_resolves_sentinel(self):
        """Re-registering a different-D corpus re-resolves the 0 sentinel instead of freezing
        the first D (which used to make quantize_oporp_1bit raise, or worse, silently bin)."""
        m = make_one_bit(k_bits=0)
        m.register_index(make_cpu_embs(d=128))
        assert m.k_bits == 128
        m.register_index(make_cpu_embs(d=64))
        assert m.k_bits == 64
        assert m.item_bits.shape == (N, 1)


class TestTorchEagerCandidatesPath:
    """Frozen epilogue of the candidates path: pad_to_k=False — min(k, P) columns, no -1/-inf
    tail; the -1 sentinel applies only to masked/short rows."""

    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_p_lt_k_returns_p_columns(self, make):
        m = make(k=6)
        m.register_index(make_cpu_embs())
        g = torch.Generator().manual_seed(7)
        cand = torch.randint(0, N, (B, 3), generator=g)
        ids, scores = m(make_cpu_embs(n=B, seed=1), candidate_ids=cand)
        assert ids.shape == (B, 3)
        assert scores.shape == (B, 3)
        assert torch.isfinite(scores).all()

    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_zero_counts_returns_sentinels(self, make):
        m = make(k=4)
        m.register_index(make_cpu_embs())
        cand = torch.zeros(B, 8, dtype=torch.long)
        counts = torch.zeros(B, dtype=torch.long)
        ids, scores = m(make_cpu_embs(n=B, seed=1), candidate_ids=cand, counts=counts)
        assert ids.shape == (B, 4)
        assert (ids == -1).all()
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_returned_ids_come_from_candidates(self, make):
        m = make(k=4)
        m.register_index(make_cpu_embs())
        g = torch.Generator().manual_seed(7)
        cand = torch.randint(0, N, (B, 8), generator=g)
        ids, _ = m(make_cpu_embs(n=B, seed=1), candidate_ids=cand)
        for b in range(B):
            allowed = set(cand[b].tolist())
            assert all(i in allowed for i in ids[b].tolist() if i >= 0)

    @pytest.mark.parametrize("make", [make_one_bit, make_simhash], ids=["one_bit", "simhash"])
    def test_full_scan_shape(self, make):
        m = make(k=4)
        m.register_index(make_cpu_embs())
        ids, scores = m(make_cpu_embs(n=B, seed=1))
        assert ids.shape == (B, 4)
        assert scores.shape == (B, 4)
        assert (scores[:, :-1] >= scores[:, 1:]).all()
        assert ((ids >= 0) & (ids < N)).all()
