"""Unit tests for the YFCC-10M loader (roadmap E1).

Everything here runs on CPU against synthetic fixtures written in the
upstream binary formats, except the last class, which opens the real
prepared dataset if it happens to be on this machine and skips otherwise.
``eval_datasets/`` had no tests at all before this file; the fixture
builders at the top are meant to be reused by the E2-E4 loaders.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from eval_datasets import yfcc
from eval_datasets import yfcc_check_gt as chk

# ----- fixture writers (the upstream binary formats) --------------------------


def write_u8bin(path: Path, mat: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.array(mat.shape, dtype="uint32").tofile(f)
        mat.astype(np.uint8).tofile(f)


def write_knn_result(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.array(ids.shape, dtype="uint32").tofile(f)
        ids.astype(np.int32).tofile(f)
        dists.astype(np.float32).tofile(f)


def write_spmat(path: Path, rows: list[list[int]], ncol: int) -> None:
    """Write bags-of-tags as the big-ann-benchmarks CSR ``.spmat`` format."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r) for r in rows])
    indices = np.array([t for r in rows for t in r], dtype=np.int32)
    with open(path, "wb") as f:
        np.array([len(rows), ncol, indices.size], dtype=np.int64).tofile(f)
        indptr.tofile(f)
        indices.tofile(f)
        np.ones(indices.size, dtype=np.float32).tofile(f)


# ----- readers ----------------------------------------------------------------


class TestReaders:
    def test_u8bin_roundtrip(self, tmp_path):
        mat = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)
        p = tmp_path / "x.u8bin"
        write_u8bin(p, mat)
        got = yfcc.read_u8bin(p, expect_dim=4)
        assert got.shape == (6, 4)
        assert np.array_equal(np.asarray(got), mat)

    def test_u8bin_rejects_wrong_dim(self, tmp_path):
        p = tmp_path / "x.u8bin"
        write_u8bin(p, np.zeros((3, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match="dim 4 != expected 192"):
            yfcc.read_u8bin(p, expect_dim=192)

    def test_u8bin_rejects_truncated(self, tmp_path):
        p = tmp_path / "x.u8bin"
        write_u8bin(p, np.zeros((3, 4), dtype=np.uint8))
        with open(p, "r+b") as f:
            f.truncate(p.stat().st_size - 1)
        with pytest.raises(ValueError, match="size"):
            yfcc.read_u8bin(p)

    def test_knn_result_roundtrip(self, tmp_path):
        ids = np.array([[3, 1, 2], [9, 8, 7]], dtype=np.int32)
        d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        p = tmp_path / "gt.ibin"
        write_knn_result(p, ids, d)
        gi, gd = yfcc.read_knn_result(p)
        assert np.array_equal(gi, ids)
        assert np.array_equal(gd, d)

    def test_spmat_roundtrip(self, tmp_path):
        rows = [[0, 5, 9], [], [2], [1, 3, 4, 7]]
        p = tmp_path / "m.spmat"
        write_spmat(p, rows, ncol=10)
        indptr, indices, ncol = yfcc.read_spmat(p)
        assert ncol == 10
        assert indptr.tolist() == [0, 3, 3, 4, 8]
        assert indices.tolist() == [0, 5, 9, 2, 1, 3, 4, 7]
        for i, want in enumerate(rows):
            assert indices[indptr[i] : indptr[i + 1]].tolist() == want

    def test_spmat_rejects_out_of_range_tag(self, tmp_path):
        p = tmp_path / "m.spmat"
        write_spmat(p, [[0, 11]], ncol=10)
        with pytest.raises(ValueError, match="column index out of range"):
            yfcc.read_spmat(p)

    def test_spmat_rejects_non_membership_values(self, tmp_path):
        p = tmp_path / "m.spmat"
        write_spmat(p, [[0, 1]], ncol=4)
        # Overwrite the trailing float32 values block with something != 1.0.
        with open(p, "r+b") as f:
            f.seek(p.stat().st_size - 8)
            np.array([0.5, 1.0], dtype=np.float32).tofile(f)
        with pytest.raises(ValueError, match="set-membership"):
            yfcc.read_spmat(p)


# ----- tag vocabulary and capping ---------------------------------------------


class TestTagRank:
    def test_orders_by_query_frequency_desc(self):
        # tag 7 asked 3x, tag 2 asked 2x, tag 5 asked once, tag 4 never.
        q = np.array([7, 7, 7, 2, 2, 5], dtype=np.int32)
        vocab, rank = yfcc.build_tag_rank(q, n_tags=10)
        assert vocab.tolist() == [7, 2, 5]
        assert rank[7] == 0 and rank[2] == 1 and rank[5] == 2
        assert rank[4] == -1 and rank[0] == -1

    def test_ties_break_by_upstream_id(self):
        q = np.array([9, 3], dtype=np.int32)
        vocab, _ = yfcc.build_tag_rank(q, n_tags=10)
        assert vocab.tolist() == [3, 9]


class TestCapTagBags:
    @staticmethod
    def _bags(rows, q_tags, n_tags, max_tags):
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r) for r in rows])
        indices = np.array([t for r in rows for t in r], dtype=np.int32)
        _, rank = yfcc.build_tag_rank(np.array(q_tags, dtype=np.int32), n_tags)
        return yfcc.cap_tag_bags(indptr, indices, rank, max_tags)

    def test_drops_tags_outside_the_query_vocabulary(self):
        # Only tags 1 and 2 are ever queried; 0 and 3 must vanish.
        bags, stats = self._bags([[0, 1, 2, 3]], q_tags=[1, 2], n_tags=4, max_tags=4)
        assert sorted(v for v in bags[0].tolist() if v != -1) == [0, 1]
        assert stats["tag_entries_total"] == 4
        assert stats["tag_entries_in_query_vocab"] == 2
        assert stats["tag_entries_kept"] == 2

    def test_keeps_most_query_frequent_first_and_pads_with_minus_one(self):
        # query freq: tag 5 -> 3, tag 6 -> 2, tag 7 -> 1  =>  dense 0, 1, 2.
        bags, stats = self._bags(
            [[7, 6, 5]], q_tags=[5, 5, 5, 6, 6, 7], n_tags=8, max_tags=2
        )
        assert bags.shape == (1, 2)
        assert bags[0].tolist() == [0, 1]  # dense ids of tags 5 and 6
        assert stats["tag_entries_kept"] == 2
        assert stats["items_uncapped_frac"] == 0.0

    def test_empty_and_short_rows_are_all_pad(self):
        bags, stats = self._bags([[], [1], []], q_tags=[1], n_tags=4, max_tags=3)
        assert bags[0].tolist() == [-1, -1, -1]
        assert bags[1].tolist() == [0, -1, -1]
        assert bags[2].tolist() == [-1, -1, -1]
        assert stats["items_with_no_tags"] == 2
        assert stats["n_items"] == 3

    def test_cap_is_subtractive_only(self):
        rows = [[0, 1, 2, 3], [1], [2, 3]]
        full, _ = self._bags(rows, q_tags=[0, 1, 2, 3], n_tags=4, max_tags=4)
        capped, _ = self._bags(rows, q_tags=[0, 1, 2, 3], n_tags=4, max_tags=2)
        for i in range(len(rows)):
            kept = {v for v in capped[i].tolist() if v != -1}
            everything = {v for v in full[i].tolist() if v != -1}
            assert kept <= everything

    def test_stats_report_the_real_distribution(self):
        bags, stats = self._bags(
            [[1, 2, 3], [1], [1, 2]], q_tags=[1, 2, 3], n_tags=4, max_tags=2
        )
        assert stats["tags_per_item_mean"] == pytest.approx(2.0)
        assert stats["tags_per_item_max"] == 3
        assert stats["restricted_tags_per_item_max"] == 3
        assert stats["items_uncapped_frac"] == pytest.approx(2 / 3)
        assert bags.shape == (3, 2)


# ----- the narrow clause tensor ----------------------------------------------


class TestNarrowTensor:
    def test_shape_dtype_and_clause_duplication(self):
        bags = np.array([[0, 1, -1], [2, -1, -1]], dtype=np.int64)
        narrow = yfcc.bags_to_narrow(bags, n_clauses=yfcc.C_NARROW)
        assert narrow.shape == (2, 2, 3)
        assert narrow.dtype == torch.int64
        assert narrow.is_contiguous()
        # Both clauses hold the same bag — that is what makes "tag j in
        # clause j" AND out to the organisers' conjunctive predicate.
        assert torch.equal(narrow[:, 0, :], narrow[:, 1, :])
        assert narrow[0, 0, :].tolist() == [0, 1, -1]

    def test_reverse_flags_are_all_false(self):
        # YFCC has no negated predicate: every clause is a forward tag match.
        flags = torch.zeros(yfcc.C_NARROW, dtype=torch.bool)
        assert flags.shape == (2,)
        assert flags.dtype == torch.bool
        assert not flags.any()

    def test_clause_semantics_are_conjunctive_and(self):
        """The narrow tensor + query attrs must reproduce "bag contains all
        query tags" under the library's own clause-match definition."""
        clause_subset_match = pytest.importorskip(
            "retrieve.layers.filters.exact_attribute"
        ).clause_subset_match
        bags = np.array(
            [
                [0, 1, -1],  # item 0: tags {0, 1}
                [1, -1, -1],  # item 1: tags {1}
                [0, 2, -1],  # item 2: tags {0, 2}
            ],
            dtype=np.int64,
        )
        narrow = yfcc.bags_to_narrow(bags, n_clauses=2)
        rev = torch.zeros(2, dtype=torch.bool)
        cand = torch.arange(3).unsqueeze(0).expand(3, -1)  # every item, 3 queries
        qa = torch.tensor([[0, 1], [1, -1], [2, 0]], dtype=torch.long)
        got = clause_subset_match(narrow[cand], qa, rev)
        # q0 wants {0,1} -> item 0 only; q1 wants {1} -> items 0,1;
        # q2 wants {2,0} -> item 2 only.
        assert got.tolist() == [
            [True, False, False],
            [True, True, False],
            [False, False, True],
        ]


class TestGtSurvivesCap:
    def test_flags_lost_ground_truth_entries(self):
        bags = np.array([[0, 1], [0, -1], [1, -1]], dtype=np.int64)
        gt_ids = np.array([[0, 1], [0, 2]], dtype=np.int64)
        qa = np.array([[0, 1], [1, -1]], dtype=np.int64)
        ok = yfcc.gt_survives_cap(bags, gt_ids, qa)
        # q0 needs both tags: item 0 has them, item 1 lost tag 1.
        # q1 needs tag 1: item 0 has it, item 2 has it.
        assert ok.tolist() == [[True, False], [True, True]]
        assert ok.all(axis=1).tolist() == [False, True]

    def test_all_survive_when_nothing_was_capped(self):
        bags = np.array([[0, 1, 2]], dtype=np.int64)
        gt_ids = np.zeros((1, 3), dtype=np.int64)
        qa = np.array([[2, 0]], dtype=np.int64)
        assert yfcc.gt_survives_cap(bags, gt_ids, qa).all()


# ----- the gate check's set algebra -------------------------------------------


class TestCheckGtHelpers:
    def test_candidates_full_intersects_postings(self, tmp_path):
        rows = [[0, 1], [1], [0], [0, 1, 2]]
        blob_path = tmp_path / "item_tags_csr.pt"
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(r) for r in rows])
        indices = np.array([t for r in rows for t in r], dtype=np.int32)
        torch.save(
            {
                "indptr": torch.from_numpy(indptr),
                "indices": torch.from_numpy(indices),
                "n_items": len(rows),
                "n_tags": 3,
            },
            blob_path,
        )
        ptr, idx = chk.load_tag_csc(tmp_path, np.array([0, 1, 2]))
        assert idx[ptr[0] : ptr[1]].tolist() == [0, 2, 3]
        assert idx[ptr[1] : ptr[2]].tolist() == [0, 1, 3]
        assert chk.candidates_full(ptr, idx, np.array([0, 1])).tolist() == [0, 3]
        assert chk.candidates_full(ptr, idx, np.array([1, -1])).tolist() == [0, 1, 3]
        assert chk.candidates_full(ptr, idx, np.array([2, 0])).tolist() == [3]

    def test_candidates_narrow_matches_the_clause_filter(self):
        bags = np.array([[0, 1], [1, -1], [0, -1]], dtype=np.int64)
        narrow = yfcc.bags_to_narrow(bags, n_clauses=2)
        assert chk.candidates_narrow(narrow, np.array([0, 1])).tolist() == [0]
        assert chk.candidates_narrow(narrow, np.array([1, -1])).tolist() == [0, 1]

    def test_topk_l2_is_exact_over_uint8_valued_fp16(self):
        embs = torch.tensor(
            [[0, 0], [3, 4], [255, 255], [1, 0]], dtype=torch.float16
        )
        q = torch.tensor([[0.0, 0.0]])
        cand = torch.tensor([0, 1, 2, 3])
        ids, sc = chk.topk_for_candidates(embs, q, cand, k=3, metric="l2", chunk=2)
        assert ids.tolist() == [0, 3, 1]
        assert sc.tolist() == [0.0, 1.0, 25.0]

    def test_topk_chunking_does_not_change_the_answer(self):
        g = torch.Generator().manual_seed(0)
        embs = torch.randint(0, 256, (200, 8), generator=g).to(torch.float16)
        q = torch.randint(0, 256, (1, 8), generator=g).float()
        cand = torch.arange(200)
        a = chk.topk_for_candidates(embs, q, cand, 10, "l2", chunk=200)
        b = chk.topk_for_candidates(embs, q, cand, 10, "l2", chunk=7)
        assert torch.equal(a[1], b[1])
        assert set(a[0].tolist()) == set(b[0].tolist())


# ----- the real dataset, if it is on this machine ------------------------------


def _real_dir() -> Path | None:
    root = os.environ.get("RETRIEVE_DATA_ROOT")
    if not root:
        return None
    d = Path(root) / "yfcc10m"
    needed = [
        "item_attrs_narrow.pt",
        "clause_is_reverse_narrow.pt",
        "item_tags_csr.pt",
        "gt_shipped.pt",
        "eval_split.parquet",
        "tag_vocab.json",
        Path("content_d192") / "text_emb.pt",
    ]
    return d if all((d / n).exists() for n in needed) else None


REAL = _real_dir()
real_only = pytest.mark.skipif(REAL is None, reason="prepared yfcc10m not on this machine")


@real_only
class TestRealSlice:
    """Round-trip a 1k-item slice of the prepared dataset against the raw CSR."""

    def test_layout_shapes_and_dtypes(self):
        rev = torch.load(REAL / "clause_is_reverse_narrow.pt", weights_only=False)
        assert rev.shape == (yfcc.C_NARROW,) and rev.dtype == torch.bool
        assert not rev.any()
        gt = torch.load(REAL / "gt_shipped.pt", map_location="cpu", weights_only=False)
        assert gt["format"] == "yfcc-shipped-gt-v1"
        assert gt["ids"].shape == (yfcc.N_QUERY, yfcc.GT_K)
        assert gt["metric"] == "squared_l2"

    def test_narrow_slice_matches_the_uncapped_csr(self):
        csr = torch.load(REAL / "item_tags_csr.pt", map_location="cpu", weights_only=False)
        indptr = csr["indptr"].numpy()
        indices = csr["indices"].numpy()
        with open(REAL / "tag_vocab.json") as f:
            vocab = np.asarray(json.load(f)["upstream_ids"], dtype=np.int64)
        rank = np.full(yfcc.N_TAGS, -1, dtype=np.int64)
        rank[vocab] = np.arange(vocab.size)

        narrow = torch.load(REAL / "item_attrs_narrow.pt", map_location="cpu", weights_only=False)
        k = narrow.shape[-1]
        assert narrow.shape[:2] == (yfcc.N_BASE, yfcc.C_NARROW)
        for i in range(1000):
            want = rank[indices[indptr[i] : indptr[i + 1]]]
            want = np.sort(want[want >= 0])[:k]
            got = narrow[i, 0].numpy()
            got = np.sort(got[got >= 0])
            assert got.tolist() == want.tolist(), f"item {i}"
            assert torch.equal(narrow[i, 0], narrow[i, 1])

    def test_embeddings_are_a_lossless_uint8_copy(self):
        emb = torch.load(
            REAL / "content_d192" / "text_emb.pt", map_location="cpu", weights_only=False
        )
        assert emb.shape == (yfcc.N_BASE, yfcc.DIM) and emb.dtype == torch.float16
        head = emb[:1000].float()
        assert torch.equal(head, head.round())
        assert head.min() >= 0 and head.max() <= 255

    def test_query_predicates_round_trip_to_upstream_tag_ids(self):
        tags, dense = chk.load_query_predicates(REAL)
        assert tags.shape == (yfcc.N_QUERY, yfcc.C_NARROW)
        assert (dense[:, 0] >= 0).all(), "every query has a first tag"
        q_indptr, q_indices, _ = yfcc.read_spmat(
            yfcc.ROOT / "query.metadata.public.100K.spmat"
        )
        for r in (0, 1, 17, 99_999):
            want = q_indices[q_indptr[r] : q_indptr[r + 1]].tolist()
            got = [t for t in tags[r].tolist() if t >= 0]
            assert got == want, f"query {r}"
