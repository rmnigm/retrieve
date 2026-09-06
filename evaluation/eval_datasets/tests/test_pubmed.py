"""CPU-only tests for the PubMed + MedCPT loader (roadmap E2).

Everything here runs on tiny synthetic fixtures written into ``tmp_path`` — no
network, no CUDA, no real shard. What is pinned down:

* the chunk-JSON ``m`` field → MeSH descriptor parsing,
* the K=4 MeSH cap rule (globally-rarest-first) that decides clause selectivity,
* MeSH tree-top categories out of a ``desc*.gz``-shaped file,
* MEDLINE baseline ``.xml.gz`` → (pmid, journal, language),
* ``convert`` → ``item_id_map.json`` / article parquet / native-768 item matrix,
* ``attrs`` → ``item_attrs_narrow.pt`` shape, reverse flags, and the fact that
  row ``i`` describes ``item_id i+1``.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
import pytest
import torch

from eval_datasets import pubmed

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# Three articles. PMIDs are deliberately out of order so the id map has to sort.
SHARD_PMIDS = ["1000002", "1000000", "1000001"]
SHARD_CONTENT = {
    "1000000": {
        "d": "20071115",
        "t": "Alpha study.",
        "a": "An abstract.",
        # "humans" is common, "rare gene" is rare, and the qualifier/major-topic
        # forms of the same descriptor must collapse to one entry.
        "m": "humans!|rare gene!|rare gene*|rare gene!genetics|animals!|",
    },
    "1000001": {
        "d": "19850102",
        "t": "Beta study.",
        "a": "",
        "m": "humans!|animals!|",
    },
    "1000002": {
        "d": "20220301",
        "t": "Gamma study.",
        "a": "Another abstract.",
        "m": "humans!|mid term!|rare gene!|",
    },
}


def _write_shard(root, shard: int, *, dim: int = 8, with_embeds: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"pmids_chunk_{shard}.json").write_text(json.dumps(SHARD_PMIDS))
    (root / f"pubmed_chunk_{shard}.json").write_text(json.dumps(SHARD_CONTENT))
    if with_embeds:
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((len(SHARD_PMIDS), dim)).astype(np.float32)
        np.save(root / f"embeds_chunk_{shard}.npy", arr)
        return arr
    return None


@pytest.fixture
def raw_root(tmp_path, monkeypatch):
    root = tmp_path / "_raw" / "pubmed"
    monkeypatch.setattr(pubmed, "ROOT", root)
    return root


# ---------------------------------------------------------------------------
# JSON → attribute parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("", []),
        (None, []),
        ("humans!|", ["humans"]),
        # descriptor!qualifier, the bare descriptor and the major-topic '*'
        # form are all the same descriptor and collapse, in first-seen order.
        (
            "humans!|rare gene!|rare gene*|rare gene!genetics|",
            ["humans", "rare gene"],
        ),
        ("Databases, Nucleic Acid*|", ["databases, nucleic acid"]),
        ("a!|  |b!|", ["a", "b"]),
    ],
)
def test_parse_mesh_field(field, expected):
    assert pubmed.parse_mesh_field(field) == expected


@pytest.mark.parametrize(
    ("d", "expected"),
    [("20071115", 2007), ("1952", 1952), ("", None), (None, None), ("abcd", None)],
)
def test_parse_year(d, expected):
    assert pubmed.parse_year(d) == expected


def test_year_to_bucket_is_monotone_and_covers_every_edge():
    assert pubmed.year_to_bucket(None) == -1
    buckets = [pubmed.year_to_bucket(y) for y in (1900, 1980, 1995, 2005, 2012, 2017, 2024)]
    assert buckets == [0, 1, 2, 3, 4, 5, 6]
    assert len(pubmed.YEAR_BUCKET_NAMES) == len(pubmed.YEAR_BUCKET_EDGES) + 1


# ---------------------------------------------------------------------------
# the MeSH cap
# ---------------------------------------------------------------------------


def test_cap_mesh_by_rarity_keeps_the_k_rarest_rarest_first():
    vocab = {"common": 0, "mid": 1, "rare": 2, "rarest": 3, "other": 4}
    freq = np.array([1000, 100, 10, 1, 500], dtype=np.int64)
    got = pubmed.cap_mesh_by_rarity(["common", "mid", "rare", "rarest", "other"], vocab, freq, k=3)
    # rarest first — that ordering is what makes the query-side clause value
    # (synthesize_qa_narrow takes the first non-pad entry) selective.
    assert got == [vocab["rarest"], vocab["rare"], vocab["mid"]]


def test_cap_mesh_by_rarity_drops_out_of_vocab_and_dedups():
    vocab = {"a": 0, "b": 1}
    freq = np.array([5, 1], dtype=np.int64)
    assert pubmed.cap_mesh_by_rarity(["zzz", "a", "a", "b"], vocab, freq, k=4) == [1, 0]
    assert pubmed.cap_mesh_by_rarity(["zzz"], vocab, freq) == []
    assert pubmed.cap_mesh_by_rarity([], vocab, freq) == []


def test_cap_mesh_by_rarity_is_deterministic_under_frequency_ties():
    vocab = {"a": 0, "b": 1, "c": 2}
    freq = np.array([7, 7, 7], dtype=np.int64)
    assert pubmed.cap_mesh_by_rarity(["c", "b", "a"], vocab, freq, k=2) == [0, 1]


# ---------------------------------------------------------------------------
# MeSH descriptor file → tree-top category
# ---------------------------------------------------------------------------


def test_mesh_category_of():
    assert pubmed.mesh_category_of(["D12.776.157"]) == pubmed.MESH_CATEGORIES.index("D")
    assert pubmed.mesh_category_of(["B01.050"]) == pubmed.MESH_CATEGORIES.index("B")
    assert pubmed.mesh_category_of([]) == -1
    assert pubmed.mesh_category_of(["Q99"]) == -1


def test_load_mesh_tree_tops(tmp_path):
    xml = """<?xml version="1.0"?>
<DescriptorRecordSet>
<DescriptorRecord>
  <DescriptorName>
   <String>Rare Gene</String>
  </DescriptorName>
  <TreeNumberList>
   <TreeNumber>G05.360</TreeNumber>
  </TreeNumberList>
</DescriptorRecord>
<DescriptorRecord>
  <DescriptorName>
   <String>Humans</String>
  </DescriptorName>
  <TreeNumberList>
   <TreeNumber>B01.050.150</TreeNumber>
  </TreeNumberList>
</DescriptorRecord>
</DescriptorRecordSet>
"""
    p = tmp_path / "desc.gz"
    with gzip.open(p, "wt") as f:
        f.write(xml)
    tops = pubmed.load_mesh_tree_tops(p)
    # keys are lower-cased so they join against the chunk JSON's lower-case names
    assert tops == {
        "rare gene": pubmed.MESH_CATEGORIES.index("G"),
        "humans": pubmed.MESH_CATEGORIES.index("B"),
    }
    assert pubmed.load_mesh_tree_tops(tmp_path / "missing.gz") == {}


# ---------------------------------------------------------------------------
# MEDLINE baseline join
# ---------------------------------------------------------------------------


def test_parse_medline_gz(tmp_path):
    xml = """<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">1000000</PMID>
      <Article><Journal><Title>Journal One</Title></Journal><Language>eng</Language></Article>
      <MedlineJournalInfo><MedlineTA>J One</MedlineTA></MedlineJournalInfo>
      <CommentsCorrectionsList><CommentsCorrections>
        <PMID Version="1">999</PMID>
      </CommentsCorrections></CommentsCorrectionsList>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">1000001</PMID>
      <Article><Language>ger</Language></Article>
      <MedlineJournalInfo><MedlineTA>J Two</MedlineTA></MedlineJournalInfo>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
    p = tmp_path / "pubmed26n0001.xml.gz"
    with gzip.open(p, "wt") as f:
        f.write(xml)
    pmids, journals, langs = pubmed.parse_medline_gz(p)
    # the record's *own* PMID wins over the one nested in CommentsCorrections
    assert pmids == [1000000, 1000001]
    assert journals == ["J One", "J Two"]
    assert langs == ["eng", "ger"]


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def _convert_args(out, **kw):
    base = dict(
        output_dir=str(out),
        shards="0",
        workers=1,
        batch_rows=2,
        skip_embeds=False,
        delete_raw=False,
        emb_scratch=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_convert_builds_a_sorted_one_indexed_id_map(raw_root, tmp_path):
    _write_shard(raw_root, 0)
    out = tmp_path / "out"
    assert pubmed.cmd_convert(_convert_args(out, skip_embeds=True)) == 0

    id_map = json.loads((out / "item_id_map.json").read_text())
    # ids are 1-indexed and dense in ascending numeric PMID order, regardless of
    # the order the PMIDs appear in inside the shard.
    assert id_map == {"1000000": 1, "1000001": 2, "1000002": 3}

    import polars as pl

    arts = pl.read_parquet(out / "staging" / "articles_chunk_0.parquet")
    assert arts.height == 3
    # article parquet keeps the *shard* row order, not the id order
    assert arts["pmid"].to_list() == [1000002, 1000000, 1000001]
    assert arts["year"].to_list() == [2022, 2007, 1985]
    assert arts["has_abstract"].to_list() == [True, True, False]
    assert arts["mesh"].to_list()[1] == ["humans", "rare gene", "animals"]


def test_convert_writes_a_normalised_native_768_style_matrix(raw_root, tmp_path):
    src = _write_shard(raw_root, 0, dim=8)
    out = tmp_path / "out"
    # The production dim is 768; the fixture uses 8 so the test stays cheap. Point
    # EMB_DIM_NATIVE at the fixture width for the duration of the call.
    orig = pubmed.EMB_DIM_NATIVE
    pubmed.EMB_DIM_NATIVE = 8
    try:
        assert pubmed.cmd_convert(_convert_args(out)) == 0
        emb = torch.load(out / f"content_d{8}" / "text_emb.pt")
    finally:
        pubmed.EMB_DIM_NATIVE = orig

    assert emb.shape == (3, 8)
    assert emb.dtype == torch.float16
    # rows are L2-normalised (the harness scores by inner product)
    assert torch.allclose(emb.float().norm(dim=-1), torch.ones(3), atol=2e-3)
    # row i holds item_id i+1, i.e. the i-th PMID in ascending order. Shard row 1
    # is PMID 1000000 == item_id 1 == output row 0.
    want = torch.nn.functional.normalize(torch.from_numpy(src[1]), dim=-1).half()
    assert torch.allclose(emb[0].float(), want.float(), atol=2e-3)

    meta = json.loads((out / "content_d8" / "text_emb.meta.json").read_text())
    assert meta["reduction"] == "none" and meta["normalization"] == "l2"
    # the accumulator is cleaned up
    assert not list((out / "_staging_emb").glob("*.npy"))


def test_convert_is_rerunnable(raw_root, tmp_path):
    _write_shard(raw_root, 0)
    out = tmp_path / "out"
    assert pubmed.cmd_convert(_convert_args(out, skip_embeds=True)) == 0
    assert pubmed.cmd_convert(_convert_args(out, skip_embeds=True)) == 0
    assert len(json.loads((out / "item_id_map.json").read_text())) == 3


def test_convert_without_pmids_fails_cleanly(raw_root, tmp_path):
    raw_root.mkdir(parents=True, exist_ok=True)
    assert pubmed.cmd_convert(_convert_args(tmp_path / "out", skip_embeds=True)) == 1


# ---------------------------------------------------------------------------
# attrs
# ---------------------------------------------------------------------------


def _attrs_args(out, **kw):
    base = dict(
        output_dir=str(out),
        mesh_vocab=100,
        mesh_min_count=1,
        journal_vocab=10,
        mesh_desc=None,
        n_heldout=2,
        seed=0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def converted(raw_root, tmp_path):
    _write_shard(raw_root, 0)
    out = tmp_path / "out"
    assert pubmed.cmd_convert(_convert_args(out, skip_embeds=True)) == 0
    return out


def test_attrs_narrow_tensor_shape_and_reverse_flags(converted, tmp_path):
    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(tmp_path / "nope.gz"))) == 0

    narrow = torch.load(converted / "item_attrs_narrow.pt")
    assert narrow.shape == (3, pubmed.C_NARROW, pubmed.A_MAX_NARROW)
    assert narrow.dtype == torch.long

    rev = torch.load(converted / "clause_is_reverse_narrow.pt")
    assert rev.dtype == torch.bool
    assert rev.tolist() == pubmed.CLAUSE_IS_REVERSE
    # exactly one reverse clause, and it is C3 (journal), per §4.1
    assert rev.sum().item() == 1
    assert bool(rev[3])


def test_attrs_row_i_describes_item_id_i_plus_one(converted, tmp_path):
    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(tmp_path / "nope.gz"))) == 0
    narrow = torch.load(converted / "item_attrs_narrow.pt")

    # row 0 == item_id 1 == PMID 1000000 (2007, has abstract)
    assert narrow[0, 2, 0].item() == pubmed.year_to_bucket(2007)
    assert narrow[0, 4, 0].item() == 1
    # row 1 == item_id 2 == PMID 1000001 (1985, no abstract)
    assert narrow[1, 2, 0].item() == pubmed.year_to_bucket(1985)
    assert narrow[1, 4, 0].item() == 0
    # no MEDLINE join was staged, so the journal clause is all padding
    assert (narrow[:, 3, 0] == -1).all()


def test_attrs_mesh_clause_is_capped_and_rarest_first(converted, tmp_path):
    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(tmp_path / "nope.gz"))) == 0
    narrow = torch.load(converted / "item_attrs_narrow.pt")
    names = json.loads((converted / "mesh_vocab.json").read_text())["names"]

    row = narrow[0, 0]  # PMID 1000000: humans(3) animals(2) rare gene(2)
    kept = [names[v] for v in row.tolist() if v != -1]
    assert len(kept) <= pubmed.A_MAX_NARROW
    assert set(kept) == {"humans", "animals", "rare gene"}
    # "humans" occurs in all three articles, so it must sort last of the three.
    assert kept[-1] == "humans"
    # padding is right-aligned: no -1 before a real value
    vals = row.tolist()
    assert vals == sorted(vals, key=lambda v: v == -1)


def test_attrs_mesh_category_clause_uses_the_descriptor_file(converted, tmp_path):
    desc = tmp_path / "desc.gz"
    with gzip.open(desc, "wt") as f:
        f.write(
            "<DescriptorRecordSet>"
            "<DescriptorRecord><DescriptorName><String>Rare Gene</String></DescriptorName>"
            "<TreeNumberList><TreeNumber>G05.360</TreeNumber></TreeNumberList></DescriptorRecord>"
            "<DescriptorRecord><DescriptorName><String>Animals</String></DescriptorName>"
            "<TreeNumberList><TreeNumber>B01.050</TreeNumber></TreeNumberList></DescriptorRecord>"
            "</DescriptorRecordSet>".replace("><", ">\n<")
        )
    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(desc))) == 0
    narrow = torch.load(converted / "item_attrs_narrow.pt")
    names = json.loads((converted / "mesh_vocab.json").read_text())["names"]

    for row in range(3):
        first = narrow[row, 0, 0].item()
        if first == -1:
            continue
        want = {"rare gene": "G", "animals": "B"}.get(names[first])
        expect = pubmed.MESH_CATEGORIES.index(want) if want else -1
        assert narrow[row, 1, 0].item() == expect


def test_attrs_eval_split_matches_heldout_and_the_narrow_tensor(converted, tmp_path):
    import polars as pl

    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(tmp_path / "nope.gz"))) == 0
    heldout = pl.read_parquet(converted / "heldout.parquet")
    split = pl.read_parquet(converted / "eval_split.parquet")
    narrow = torch.load(converted / "item_attrs_narrow.pt")

    assert split.height == heldout.height == 2
    assert split["target_id"].to_list() == heldout["item_id"].to_list()
    id_map = json.loads((converted / "item_id_map.json").read_text())
    for pmid, iid in zip(heldout["pmid"].to_list(), heldout["item_id"].to_list(), strict=True):
        assert id_map[pmid] == iid

    # query_attrs_narrow[c] is the target's first non-pad value in clause c
    pairs = zip(split["target_id"].to_list(), split["query_attrs_narrow"].to_list(), strict=True)
    for tgt, qa in pairs:
        for c in range(pubmed.C_NARROW):
            assert qa[c] == narrow[tgt - 1, c, 0].item()


def test_attrs_writes_every_vocab_file(converted, tmp_path):
    assert pubmed.cmd_attrs(_attrs_args(converted, mesh_desc=str(tmp_path / "nope.gz"))) == 0
    for name in (
        "mesh_vocab.json",
        "mesh_cat_vocab.json",
        "year_vocab.json",
        "journal_vocab.json",
        "lang_vocab.json",
    ):
        assert (converted / name).exists(), name
    log = json.loads((converted / "prep_log.json").read_text())
    assert log["attrs"]["mesh_cap_k"] == pubmed.A_MAX_NARROW
    assert "rarest" in log["attrs"]["mesh_cap_rule"]


def test_attrs_without_convert_fails_cleanly(tmp_path):
    assert pubmed.cmd_attrs(_attrs_args(tmp_path / "nothing")) == 1


# ---------------------------------------------------------------------------
# misc plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, list(range(38))),
        ("0", [0]),
        ("0-3", [0, 1, 2, 3]),
        ("5,0-1", [0, 1, 5]),
        ("2,2", [2]),
    ],
)
def test_parse_shards(spec, expected):
    assert pubmed._parse_shards(spec) == expected


def test_clause_layout_is_the_documented_one():
    assert pubmed.C_NARROW == 5
    assert pubmed.A_MAX_NARROW == 4
    assert pubmed.CLAUSE_NAMES == ["mesh", "mesh_cat", "year", "journal", "has_abstract"]
    assert len(pubmed.CLAUSE_IS_REVERSE) == pubmed.C_NARROW
