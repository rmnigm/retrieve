"""``eval_datasets.layout``: the legacy ``[N+1]`` pad-row rule, ``apply_users_limit`` as one
prefix, and ``validate_layout`` on the conftest writer's tiny dataset — clean on both
layouts, and flagging the breakages the review found (a dropped ``eval_split`` row, a missing
prefix sidecar, a swapped prefix, misaligned attrs)."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import write_tiny_dataset

from eval_datasets import layout


def test_drop_legacy_padding_row_attrs():
    legacy = torch.tensor([[[-1, -1]], [[3, 4]], [[5, 6]]])  # [N+1, C, A]
    out = layout.drop_legacy_padding_row(legacy, kind="attrs", what="t")
    assert out.shape == (2, 1, 2) and torch.equal(out, legacy[1:]) and out.is_contiguous()


def test_drop_legacy_padding_row_embeddings():
    legacy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = layout.drop_legacy_padding_row(legacy, kind="emb", what="t")
    assert out.shape == (2, 2) and torch.equal(out, legacy[1:])
    # Recognition is by content per kind: an all-zero row is not an attrs pad row and an
    # all-(-1) row is not an embedding pad row.
    assert torch.equal(layout.drop_legacy_padding_row(legacy, kind="attrs", what="t"), legacy)
    neg = torch.tensor([[-1.0, -1.0], [1.0, 0.0]])
    assert torch.equal(layout.drop_legacy_padding_row(neg, kind="emb", what="t"), neg)
    with pytest.raises(ValueError, match="kind"):
        layout.drop_legacy_padding_row(neg, kind="ids", what="t")


def test_drop_legacy_padding_row_leaves_modern_layout_alone():
    modern = torch.tensor([[[3, 4]], [[5, 6]]])
    assert torch.equal(layout.drop_legacy_padding_row(modern, kind="attrs", what="t"), modern)
    embs = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # a pad-looking row anywhere but 0 is data
    assert torch.equal(layout.drop_legacy_padding_row(embs, kind="emb", what="t"), embs)
    assert layout.drop_legacy_padding_row(torch.empty(0, 3), kind="emb", what="t").shape == (0, 3)


def test_apply_users_limit_is_one_prefix():
    a, b = torch.arange(6), torch.arange(12).reshape(6, 2)
    assert [t.tolist() for t in layout.apply_users_limit(4, a, b)] == [[0, 1, 2, 3], b[:4].tolist()]
    assert layout.apply_users_limit(4, a, None)[1] is None
    assert layout.apply_users_limit(None, a) == (a,) and layout.apply_users_limit(9, a) == (a,)


def test_validate_layout_text_shape(tmp_path):
    root = write_tiny_dataset(tmp_path / "modern")
    assert layout.validate_layout(root, root / "content") == []
    legacy = write_tiny_dataset(tmp_path / "legacy", legacy=True)
    assert layout.validate_layout(legacy, legacy / "content") == []
    # PubMed's cmd_queries breakage: eval_split shorter than the query set.
    short = write_tiny_dataset(tmp_path / "short", n_split=5)
    assert layout.validate_layout(short, short / "content") == ["eval_split rows 5 != queries 8"]
    # YFCC's breakage: no prefix sidecars.
    bare = write_tiny_dataset(tmp_path / "bare")
    (bare / "content" / "text_emb.meta.json").unlink()
    (problem,) = layout.validate_layout(bare, bare / "content")
    assert problem.startswith("missing") and "text_emb.meta.json" in problem
    swapped = write_tiny_dataset(tmp_path / "swapped", doc_prefix="search_query: ")
    assert layout.validate_layout(swapped, swapped / "content") == [
        f"{swapped / 'content' / 'text_emb.meta.json'}: prefix != 'search_document: '"
    ]
    wrong = write_tiny_dataset(tmp_path / "wrong")
    torch.save(torch.full((5, 2, 1), 7, dtype=torch.long), wrong / "item_attrs_narrow.pt")
    assert layout.validate_layout(wrong, wrong / "content") == [
        "item_attrs_narrow.pt rows 5 != items 24"
    ]
    assert layout.validate_layout(tmp_path / "nowhere", tmp_path / "nowhere" / "content") == [
        f"missing {tmp_path / 'nowhere' / 'content' / 'query_emb.pt'}",
        f"missing {tmp_path / 'nowhere' / 'content' / 'text_emb.pt'}",
        f"missing {tmp_path / 'nowhere' / 'content' / 'text_emb.meta.json'} "
        "(no prefix assertion possible)",
        f"missing {tmp_path / 'nowhere' / 'content' / 'query_emb.meta.json'} "
        "(no prefix assertion possible)",
        f"missing {tmp_path / 'nowhere' / 'heldout.parquet'}",
    ]


def test_validate_layout_sequential_shape(tmp_path):
    import polars as pl

    root = tmp_path / "seq"
    root.mkdir()
    (root / "item_id_map.json").write_text(json.dumps({str(i): i for i in range(1, 25)}))
    assert layout.validate_layout(root) == [f"missing {root / 'test.parquet'}"]
    pl.DataFrame({"item_ids": [[1, 2]] * 8, "targets": [[3]] * 8}).write_parquet(
        root / "test.parquet"
    )
    assert layout.validate_layout(root) == []  # train / val are not the harness's business
    torch.save(torch.full((25, 2, 1), -1, dtype=torch.long), root / "item_attrs_narrow.pt")
    assert layout.validate_layout(root) == [
        f"missing {root / 'clause_is_reverse_narrow.pt'}",
        f"missing {root / 'eval_split.parquet'}",
    ]
    torch.save(torch.tensor([False, True]), root / "clause_is_reverse_narrow.pt")
    pl.DataFrame({"query_attrs_narrow": [[0, 1]] * 7}).write_parquet(root / "eval_split.parquet")
    assert layout.validate_layout(root) == ["eval_split rows 7 != queries 8"]
