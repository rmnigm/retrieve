"""CPU-only tests for ``retrieval.oracle`` (harness v2 WP-2, blob v4).

Padding semantics first (unchanged since E6): under an exact mask the losing top-k slots tie
at ``-inf`` and must come back as ``-1``, never as low-index junk; skipped rows stay all
``-1``; ``k_gt`` beyond the catalog pads the tail. Then the v4 blob: its fields and their
arithmetic (``pass_counts`` / ``pass_rate`` / ``targets_in_filter`` / ``target_in_filter``)
on hand-built masks, the content fingerprint in the file name (same content → cache hit,
mutated content → a different file, so a stale blob is never read), the bloom false-positive
rate, and the resume key with ``code_version`` (git tree hash, or the ``files:`` fallback
outside git). Plain torch ops only; no data files, no GPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from retrieval import bench, oracle
from retrieval.oracle import compute, compute_filtered_oracle, load_or_build, load_or_build_oracle

CPU = torch.device("cpu")


class _FixedMaskFilter:
    """FilterModule stand-in: one fixed per-item mask for every query."""

    def __init__(self, item_mask):
        self.item_mask = torch.as_tensor(item_mask, dtype=torch.bool)

    def evaluate_mask(self, qa):
        return self.item_mask.unsqueeze(0).expand(qa.shape[0], -1)


class _PerQueryFilter:
    """FilterModule stand-in: row ``i`` of ``masks`` is the mask of query ``qa[i, 0]``."""

    def __init__(self, masks):
        self.masks = torch.as_tensor(masks, dtype=torch.bool)

    def evaluate_mask(self, qa):
        return self.masks[qa[:, 0]]


# ----- padding semantics ----------------------------------------------------------------


def test_oracle_pads_short_rows_with_minus_one():
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    out = compute_filtered_oracle(
        item_embs, queries, qa, None, _FixedMaskFilter([False, True, False]), 2, device=CPU
    )
    assert out.dtype == torch.long and out.tolist() == [[1, -1], [1, -1]]


def test_oracle_skip_mask_rows_stay_minus_one():
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    blob = compute(
        item_embs,
        queries,
        qa,
        torch.tensor([True, False]),
        _FixedMaskFilter([1, 1, 1]),
        2,
        device=CPU,
    )
    assert blob["topk"].tolist() == [[-1, -1], [1, 2]]
    assert blob["pass_counts"].tolist() == [-1, 3]  # skipped row carries -1


def test_oracle_k_gt_beyond_catalog_pads_tail():
    out = compute_filtered_oracle(
        torch.eye(3), torch.tensor([[3.0, 2.0, 1.0]]), None, None, None, 5, device=CPU
    )
    assert out.shape == (1, 5) and out[0].tolist() == [0, 1, 2, -1, -1]


# ----- v4 fields ------------------------------------------------------------------------


def _v4_inputs():
    """4 items, 3 queries; per-query masks admit {0,1,2}, {2}, {} (all -1 on query 2's row is
    the caller's skip case, so here it is a mask that admits nothing)."""
    item_embs = torch.eye(4)
    queries = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    qa = torch.tensor([[0], [1], [2]])
    filt = _PerQueryFilter([[1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    targets = torch.tensor([[0, 3], [2, -1], [1, -1]])
    return item_embs, queries, qa, filt, targets


def test_v4_fields_and_pass_rate_arithmetic():
    item_embs, queries, qa, filt, targets = _v4_inputs()
    blob = compute(item_embs, queries, qa, None, filt, 2, targets=targets, device=CPU)
    assert blob["topk"].tolist() == [[0, 1], [2, -1], [-1, -1]]
    assert blob["pass_counts"].tolist() == [3, 1, 0]
    assert blob["pass_rate"] == pytest.approx((3 + 1 + 0) / 3 / 4)
    # target_in_filter: q0 targets {0 in, 3 out}; q1 target 2 in; q2 target 1 out.
    assert blob["targets_in_filter"].tolist() == [[True, False], [True, False], [False, False]]
    assert blob["target_in_filter"].tolist() == [True, True, False]
    assert blob["targets_in_filter"].shape == targets.shape
    # Skipped rows: -1 counts, excluded from pass_rate, no targets in filter.
    blob = compute(
        item_embs,
        queries,
        qa,
        torch.tensor([False, True, False]),
        filt,
        2,
        targets=targets,
        device=CPU,
    )
    assert blob["pass_counts"].tolist() == [3, -1, 0]
    assert blob["pass_rate"] == pytest.approx((3 + 0) / 2 / 4)
    assert blob["target_in_filter"].tolist() == [True, False, False]
    # No filter: everything passes, every valid target is in filter.
    blob = compute(item_embs, queries, None, None, None, 2, targets=targets, device=CPU)
    assert blob["pass_counts"].tolist() == [4, 4, 4] and blob["pass_rate"] == 1.0
    assert blob["targets_in_filter"].tolist() == [[True, True], [True, False], [True, False]]


def test_pass_counts_and_bloom_fp_rate():
    _, _, qa, filt, _ = _v4_inputs()
    exact = oracle.pass_counts(filt, qa, None, device=CPU)
    assert exact.tolist() == [3, 1, 0]
    assert oracle.pass_rate(exact, 4) == pytest.approx(4 / 12)
    assert oracle.pass_rate(torch.tensor([-1, -1]), 4) != oracle.pass_rate(
        torch.tensor([-1, -1]), 4
    )  # nan
    # A "bloom" that admits one extra item on queries 0 and 2, none on query 1.
    bloom = _PerQueryFilter([[1, 1, 1, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
    bcounts = oracle.pass_counts(bloom, qa, None, device=CPU)
    assert bcounts.tolist() == [4, 1, 1]
    # fp rate per query: (4-3)/(4-3)=1, (1-1)/(4-1)=0, (1-0)/(4-0)=0.25 → mean 0.41666
    assert oracle.bloom_fp_rate(bcounts, exact, 4) == pytest.approx((1 + 0 + 0.25) / 3)
    # Skipped rows (-1) drop out of the mean.
    skip = torch.tensor([False, True, False])
    assert oracle.pass_counts(bloom, qa, skip, device=CPU).tolist() == [4, -1, 1]
    assert oracle.bloom_fp_rate(
        oracle.pass_counts(bloom, qa, skip, device=CPU),
        oracle.pass_counts(filt, qa, skip, device=CPU),
        4,
    ) == pytest.approx((1 + 0.25) / 2)


# ----- cache: fingerprint in the file name ----------------------------------------------


def _build(gt_dir, item_embs, queries, targets=None, clauses=(0,)):
    _, _, qa, filt, _ = _v4_inputs()
    return load_or_build(
        gt_dir,
        "s",
        2,
        item_embs=item_embs,
        queries=queries,
        targets=targets,
        qa_sweep=qa,
        skip_mask=None,
        clauses=clauses,
        filter_mod=filt,
        device=CPU,
    )


def test_blob_v4_round_trip_and_cache_hit(tmp_path):
    item_embs, queries, _, _, targets = _v4_inputs()
    blob = _build(tmp_path, item_embs, queries, targets)
    files = sorted(tmp_path.glob("oracle_v4_s_*.pt"))
    assert len(files) == 1 and files[0].name == f"oracle_v4_s_{blob['fingerprint'][:16]}.pt"
    assert blob["version"] == 4 and blob["k_gt"] == 2 and blob["sweep"] == "s"
    assert blob["clauses"] == [0] and blob["n_items"] == 4 and blob["n_queries"] == 3
    assert blob["n_kept"] == 3 and blob["pass_rate"] == pytest.approx(4 / 12)
    assert blob["code_version"] == bench.code_version() and blob["torch"] == str(torch.__version__)
    assert isinstance(blob["harness_commit"], str) and blob["created"].endswith("+00:00")
    # A true cache hit returns the stored blob verbatim (sentinel round-trip).
    stored = torch.load(str(files[0]), map_location="cpu", weights_only=True)
    torch.save({**stored, "topk": torch.full_like(stored["topk"], 7)}, str(files[0]))
    again = _build(tmp_path, item_embs, queries, targets)
    assert torch.equal(again["topk"], torch.full_like(stored["topk"], 7))
    assert torch.equal(again["targets_in_filter"], blob["targets_in_filter"])


def test_fingerprint_invalidates_on_content_targets_and_clauses(tmp_path):
    item_embs, queries, _, _, targets = _v4_inputs()
    first = _build(tmp_path, item_embs, queries, targets)
    # Same shape, one mutated embedding → a different file, freshly computed.
    mutated = item_embs.clone()
    mutated[0] = 0.0
    second = _build(tmp_path, mutated, queries, targets)
    assert second["fingerprint"] != first["fingerprint"]
    assert second["topk"].tolist()[0] == [1, 2]
    assert len(list(tmp_path.glob("oracle_v4_s_*.pt"))) == 2
    # Targets and the clause set are in the hash too (v4 fields depend on them).
    other_t = targets.clone()
    other_t[0, 0] = 3
    assert _build(tmp_path, item_embs, queries, other_t)["fingerprint"] != first["fingerprint"]
    assert (
        _build(tmp_path, item_embs, queries, targets, clauses=(0, 1))["fingerprint"]
        != first["fingerprint"]
    )
    assert _build(tmp_path, item_embs, queries, targets)["fingerprint"] == first["fingerprint"]
    assert len(list(tmp_path.glob("oracle_v4_s_*.pt"))) == 4


def test_non_v4_file_at_the_v4_path_is_rebuilt(tmp_path):
    item_embs, queries, _, _, targets = _v4_inputs()
    _, _, qa, _, _ = _v4_inputs()
    fp = oracle.fingerprint(item_embs, queries, targets, qa, (0,), 2)
    path = oracle.blob_path(tmp_path, "s", fp)
    torch.save(torch.zeros(3, 2, dtype=torch.long), str(path))  # a bare tensor
    blob = _build(tmp_path, item_embs, queries, targets)
    assert blob["topk"].tolist() == [[0, 1], [2, -1], [-1, -1]]
    stored = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(stored, dict) and stored["version"] == 4


def test_old_wrapper_returns_topk_from_a_v4_blob(tmp_path):
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])
    out = load_or_build_oracle(
        tmp_path,
        "s",
        2,
        item_embs=item_embs,
        queries=queries,
        qa_narrow_sweep=None,
        skip_mask=None,
        oracle_filter=None,
        device=CPU,
    )
    assert out.tolist() == [[0, 1]]
    assert len(list(tmp_path.glob("oracle_v4_s_*.pt"))) == 1


# ----- code_version / resume_key ----------------------------------------------------------


def test_code_version_is_the_library_tree_hash_or_a_files_hash(monkeypatch):
    v = bench.code_version()
    assert v == bench._git("rev-parse", "HEAD:retrieve/src/retrieve") and len(v) == 40
    monkeypatch.setattr(bench, "_git", lambda *a: None)
    fb = bench.code_version()
    assert fb.startswith("files:") and len(fb) == len("files:") + 40
    assert bench.code_version() == fb  # deterministic
    assert bench.provenance()["code_version"] == fb


def test_resume_key_is_canonical_and_includes_code_version():
    key = {
        "dataset": "goodreads",
        "dim": 128,
        "suite": "filter",
        "filter_kind": "clause",
        "sweep": "c0_genre",
        "algo": "silvertorch",
        "backend": "triton",
        "params": {"n_probe": 24, "n_lists": 1024},
        "seed": 0,
    }
    a = oracle.resume_key(key, "abc")
    assert a == oracle.resume_key(dict(reversed(list(key.items()))), "abc")  # order-free
    assert a != oracle.resume_key(key, "abd")  # a kernel change invalidates the cell
    assert a != oracle.resume_key({**key, "params": {"n_probe": 32, "n_lists": 1024}}, "abc")
    assert json.loads(a)["code_version"] == "abc" and set(json.loads(a)) == set(
        oracle.KEY_FIELDS
    ) | {"code_version"}
