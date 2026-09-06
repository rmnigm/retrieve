"""CPU-only tests for ``retrieval.data`` (harness v2 WP-2).

A tiny pre-encoded (arxiv-shaped) dataset is written into ``tmp_path`` by
``conftest.write_tiny_dataset`` — ``content/`` with ``text_emb.pt`` / ``query_emb.pt`` and
their meta sidecars, ``heldout.parquet``, ``item_attrs_narrow.pt``,
``clause_is_reverse_narrow.pt``, ``eval_split.parquet`` — so ``load_inputs`` can be checked
end to end: fp16 → fp32 + normalisation, the −1 shift of
held-out ids, the prefix assertion, the eval_split row-count check against the *full* split,
and ``users_limit`` applied once to queries / targets / attrs together. Then ``sweep_qa``
(the old ``build_sweep_qa`` semantics), ``build_filters`` keyed by *filter* backend on the
torch path, ``exact_filter``, and the fixed-seed ``query_pool``. The SASRec path needs a
checkpoint and is exercised in C4.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import write_tiny_dataset  # pytest puts this directory on sys.path

from retrieval import data, oracle
from retrieval.config import Dataset
from retrieve import BloomFilter, ExactAttributeFilter

N, U, D, C = 12, 6, 8, 2


def _write_dataset(root: Path, *, doc_prefix="search_document: ", n_split=U) -> Dataset:
    """The conftest writer at this file's smaller shape, as a resolved ``Dataset``."""
    write_tiny_dataset(root, n=N, u=U, doc_prefix=doc_prefix, n_split=n_split)
    return Dataset(
        name="tiny",
        dim=D,
        data_dir=root,
        checkpoint=None,
        content_dir=root / "content",
        users_limit=None,
        encode={},
        attrs=root / "item_attrs_narrow.pt",
        reverse=root / "clause_is_reverse_narrow.pt",
        clauses={"clause": {"c0": (0,), "c0c1": (0, 1)}, "bloom": {"c0": (0,)}},
    )


def test_load_inputs_pre_encoded(tmp_path):
    ds = _write_dataset(tmp_path)
    inp = data.load_inputs(ds, torch.device("cpu"))
    assert inp["item_embs"].shape == (N, D) and inp["item_embs"].dtype == torch.float32
    assert torch.allclose(inp["item_embs"].norm(dim=1), torch.ones(N), atol=1e-5)
    assert torch.allclose(inp["queries"].norm(dim=1), torch.ones(U), atol=1e-5)
    assert inp["targets"].tolist() == [[i] for i in range(U)]  # 1-indexed on disk → 0-indexed
    assert inp["n_targets"].tolist() == [1] * U
    assert inp["qa"].shape == (U, C) and inp["qa"][4].tolist() == [-1, -1]
    assert inp["item_attrs"].shape == (N, C, 1) and inp["clause_is_reverse"].tolist() == [
        False,
        True,
    ]
    assert (inp["n_items"], inp["n_queries"]) == (N, U)
    assert inp["attrs_digest"] == oracle.attrs_digest(inp["item_attrs"], inp["clause_is_reverse"])
    # Without filters nothing attribute-shaped is loaded.
    lean = data.load_inputs(ds, torch.device("cpu"), with_filters=False)
    assert lean["qa"] is None and lean["item_attrs"] is None and lean["clause_is_reverse"] is None
    assert lean["attrs_digest"] == oracle.attrs_digest(None, None) != inp["attrs_digest"]


def test_users_limit_is_one_prefix_over_every_query_tensor(tmp_path):
    ds = _write_dataset(tmp_path)
    full = data.load_inputs(ds, torch.device("cpu"))
    ds3 = Dataset(**{**ds.__dict__, "users_limit": 3})
    inp = data.load_inputs(ds3, torch.device("cpu"))
    assert inp["n_queries"] == 3 and inp["n_items"] == N
    for key in ("queries", "targets", "n_targets", "qa"):
        assert torch.equal(inp[key], full[key][:3]), key
    # A limit at or above the split is a no-op, never an error.
    big = Dataset(**{**ds.__dict__, "users_limit": 10_000})
    assert data.load_inputs(big, torch.device("cpu"))["n_queries"] == U


def test_load_inputs_checks_prefix_and_eval_split_rows(tmp_path):
    ds = _write_dataset(tmp_path / "a", doc_prefix="search_query: ")
    with pytest.raises(RuntimeError, match="prefix"):
        data.load_inputs(ds, torch.device("cpu"))
    ds = _write_dataset(tmp_path / "b", n_split=U - 1)
    with pytest.raises(RuntimeError, match="rows=5 != queries=6"):
        data.load_inputs(ds, torch.device("cpu"))
    # ...even with users_limit below the split: the check is against the full split.
    ds = Dataset(**{**ds.__dict__, "users_limit": 2})
    with pytest.raises(RuntimeError, match="rows=5 != queries=6"):
        data.load_inputs(ds, torch.device("cpu"))


# ----- sweep_qa ---------------------------------------------------------------------------


def test_sweep_qa_masks_inactive_clauses_and_skips_dead_rows():
    qa = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, 9, -1, -1]])
    out, skip = data.sweep_qa(qa, (1,))
    assert (out[:, [0, 2, 3]] == -1).all() and (out[:, 1] == qa[:, 1]).all()
    assert skip.tolist() == [False, False, False]
    out, skip = data.sweep_qa(qa, (0, 2))
    assert skip.tolist() == [False, False, True] and (out[2] == -1).all()
    assert qa[2, 1] == 9  # input never mutated
    assert data.sweep_qa(qa, None) == (None, None)
    with pytest.raises(ValueError, match="out of range"):
        data.sweep_qa(qa, (4,))
    with pytest.raises(ValueError, match="needs query attrs"):
        data.sweep_qa(None, (0,))


# ----- filters + pool -----------------------------------------------------------------------


def test_build_filters_keyed_by_filter_backend(tmp_path):
    inp = data.load_inputs(_write_dataset(tmp_path), torch.device("cpu"))
    assert (
        data.build_filters("none", inp, ["triton", "torch"], bloom={"m_bits": 64, "k_hash": 3})
        == {}
    )
    mods = data.build_filters("clause", inp, ["torch"], bloom={"m_bits": 64, "k_hash": 3})
    assert set(mods) == {"torch"} and isinstance(mods["torch"], ExactAttributeFilter)
    qa = torch.tensor([[0, -1], [-1, 1]])
    mask = mods["torch"].evaluate_mask(qa)
    assert mask[0].tolist() == [(i % 3 == 0) for i in range(N)]  # clause 0 = 0
    assert mask[1].tolist() == [(i % 2 != 1) for i in range(N)]  # clause 1 is reverse: != 1
    assert data.exact_filter("clause", mods, inp, "torch") is mods["torch"]
    bl = data.build_filters("bloom", inp, ["torch"], bloom={"m_bits": 64, "k_hash": 3})
    assert isinstance(bl["torch"], BloomFilter) and bl["torch"].m_bits == 64
    ex = data.exact_filter("bloom", bl, inp, "torch")
    assert isinstance(ex, ExactAttributeFilter) and ex is not bl["torch"]
    assert data.exact_filter("none", {}, inp, "torch") is None
    # official → triton filter backend (O §6.2), deduplicated across cell backends; the
    # Triton module registers on CPU (no kernel until evaluate), so this is a CPU check.
    tri = data.build_filters(
        "clause", inp, ["triton", "official"], bloom={"m_bits": 64, "k_hash": 3}
    )
    assert set(tri) == {"triton"} and isinstance(tri["triton"], ExactAttributeFilter)
    assert data.exact_filter("clause", tri, inp, "official") is tri["triton"]


def test_query_pool_is_seeded_and_skips_masked_rows(tmp_path):
    inp = data.load_inputs(_write_dataset(tmp_path), torch.device("cpu"))
    qa_s, skip = data.sweep_qa(inp["qa"], (0,))
    assert skip.tolist() == [False, False, False, False, True, False]
    pool, qa_pool = data.query_pool(
        inp, qa_s, skip, bs=2, seed=7, n_pool=50, device=torch.device("cpu")
    )
    assert pool.shape == (50, 2, D) and qa_pool.shape == (50, 2, C)
    # Every pooled query is a kept row, and the attrs travel with their query.
    for b in range(50):
        for i in range(2):
            row = (inp["queries"] == pool[b, i]).all(dim=1).nonzero().item()
            assert row != 4 and torch.equal(qa_pool[b, i], qa_s[row])
    again, _ = data.query_pool(inp, qa_s, skip, bs=2, seed=7, n_pool=50, device=torch.device("cpu"))
    assert torch.equal(pool, again)
    other, _ = data.query_pool(inp, qa_s, skip, bs=2, seed=8, n_pool=50, device=torch.device("cpu"))
    assert not torch.equal(pool, other)
    # Same draw as the old harness at the same seed (torch.randint over kept rows).
    g = torch.Generator().manual_seed(7)
    keep = (~skip).nonzero().reshape(-1)
    rows = keep[torch.randint(0, keep.numel(), (50, 2), generator=g).reshape(-1)]
    assert torch.equal(pool, inp["queries"][rows].reshape(50, 2, -1))
    pool_none, qa_none = data.query_pool(
        inp, None, None, bs=1, seed=0, n_pool=8, device=torch.device("cpu")
    )
    assert pool_none.shape == (8, 1, D) and qa_none is None
    with pytest.raises(ValueError, match="skip-masked"):
        data.query_pool(
            inp, qa_s, torch.ones(U, dtype=torch.bool), bs=1, seed=0, device=torch.device("cpu")
        )
