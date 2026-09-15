"""``bench.upload``: the publish path, with no network.

The gates this pins — the command had none until now (V §11 listed it "exercised by no
test"): campaign scratch never leaves the box, the manifest's checksums describe the files
actually listed, the citability verdict travels with the payload and ``--gate`` cannot beat
the evidence (CLAUDE.md rule 2), the generated README says so in prose, the round-trip check
catches a changed byte, and the commit carries every file plus the two generated ones with
``private=True`` unless ``--public`` is passed.

``HfApi`` is replaced by a recorder: the real upload is a validation-record item, not a CI
dependency (coding-guidelines D6 — tests are gates, not a deliverable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench import records, upload

ENV = {
    "gpu": "NVIDIA A100-SXM4-80GB", "driver": "580.159.04", "cuda": "12.8",
    "torch": "2.10.0+cu128", "triton": "3.6.0", "commit": "abc1234", "dirty": False,
    "git_branch": "development", "code_version": "f" * 40, "host": "box", "python": "3.11.15",
    "started": "2026-09-15T08:00:00+00:00",
}  # fmt: skip


def _rec(**over):
    rec = {
        "schema_version": records.SCHEMA_VERSION, "status": "ok", "dataset": "goodreads",
        "dim": 128, "suite": "filter", "filter_kind": "clause", "sweep": "c0_genre",
        "algo": "silvertorch", "backend": "triton", "params": {}, "seed": 0, "path": "triton",
        "perf": None, "unstable": False, "env": dict(ENV),
    }
    rec["env"].update(over.pop("env", {}))
    return {**rec, **over}


@pytest.fixture
def tree(tmp_path):
    """A results tree with one record file, its samples sidecar, and both scratch dirs."""
    root = tmp_path / "results"
    records.append_record(root / "filter" / "goodreads-d128.jsonl", _rec())
    records.append_record(root / "filter" / "goodreads-d128.jsonl", _rec(backend="torch"))
    records.append_record(
        root / "filter" / "goodreads-d128.samples.jsonl",
        {"dataset": "goodreads", "k": 100, "bs": 1, "mode": "eager", "ms": [0.5, 0.51]},
    )
    (root / "flat.csv").write_text("dataset,dim\ngoodreads,128\n")
    (root / "_logs").mkdir()
    (root / "_logs" / "campaign.log").write_text("noise")
    (root / "_parity").mkdir()
    (root / "_parity" / "deadbeef.npz").write_bytes(b"\x00" * 4096)
    return root


class FakeApi:
    """Records what would have gone to the Hub."""

    def __init__(self, repo_files=()):
        self.repo_files = list(repo_files)
        self.created = None
        self.commits = []

    def create_repo(self, **kw):
        self.created = kw

    def list_repo_files(self, **kw):
        return list(self.repo_files)

    def create_commit(self, **kw):
        self.commits.append(kw)


@pytest.fixture
def fake_hub(monkeypatch):
    api = FakeApi()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)
    return api


def _run(*args):
    return CliRunner().invoke(upload.upload, list(args))


# ----- the listing: scratch never leaves the box ------------------------------


def test_files_skips_campaign_scratch_and_generated_files(tree):
    (tree / "MANIFEST.json").write_text("{}")
    (tree / "README.md").write_text("stale")
    rel = [p.relative_to(tree).as_posix() for p in upload.files(tree)]
    assert rel == [
        "filter/goodreads-d128.jsonl",
        "filter/goodreads-d128.samples.jsonl",
        "flat.csv",
    ]


# ----- the manifest ----------------------------------------------------------


def test_manifest_checksums_match_the_listed_files(tree):
    listing = upload.files(tree)
    man = upload.manifest(tree, listing, "c4", None)
    assert man["n_files"] == len(listing) == len(man["files"])
    assert man["bytes"] == sum(p.stat().st_size for p in listing)
    assert man["n_records"] == 2  # the samples sidecar is not a record file
    for f in man["files"]:
        p = tree / f["path"]
        assert f["sha256"] == upload.sha256(p) and f["bytes"] == p.stat().st_size


def test_manifest_is_not_citable_without_a_gate(tree):
    man = upload.manifest(tree, upload.files(tree), "c4", None)
    assert man["citable"] is False
    assert any("no --gate" in b for b in man["blockers"])
    assert man["commits"] == ["abc1234"] and man["branches"] == ["development"]
    assert man["schema_versions"] == [records.SCHEMA_VERSION]
    assert man["code_versions"] == ["f" * 40]


def test_gate_makes_clean_records_citable(tree):
    man = upload.manifest(tree, upload.files(tree), "c4", "D1")
    assert man["citable"] is True and man["blockers"] == [] and man["gate"] == "D1"


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"status": "partial"}, "partial"),
        ({"status": "failed"}, "failed"),
        ({"env": {"dirty": True}}, "env.dirty"),
        ({"env": {"git_branch": "dev/results-storage"}}, "dev/results-storage"),
    ],
)
def test_evidence_beats_the_gate(tmp_path, over, reason):
    root = tmp_path / "results"
    records.append_record(root / "filter" / "goodreads-d128.jsonl", _rec(**over))
    man = upload.manifest(root, upload.files(root), "probe", "D1")
    assert man["citable"] is False
    assert any(reason in b for b in man["blockers"]), man["blockers"]


# ----- the generated README --------------------------------------------------


def test_readme_carries_every_subtree_and_its_verdict(tree):
    a = upload.manifest(tree, upload.files(tree), "c4", None)
    b = upload.manifest(tree, upload.files(tree), "d1-a", "D1")
    text = upload.readme([b, a])
    assert text.index("## `c4`") < text.index("## `d1-a`")  # sorted, not insertion order
    assert "**NOT CITABLE.** Reasons:" in text and "no --gate" in text
    assert "**CITABLE** — the uploader declared gate `D1` green." in text
    assert "| `c4` | 2 | ok=2 | NO |" in text
    assert "| `d1-a` | 2 | ok=2 | **yes** |" in text
    assert "abc1234" in text and "NVIDIA A100-SXM4-80GB" in text


# ----- the round trip --------------------------------------------------------


def test_verify_accepts_a_faithful_copy_and_catches_a_changed_byte(tree, tmp_path):
    man = upload.manifest(tree, upload.files(tree), "c4", None)
    copy = tmp_path / "downloaded"
    for f in man["files"]:
        dst = copy / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((tree / f["path"]).read_bytes())
    assert upload.verify(copy, man) == []

    (copy / "flat.csv").write_bytes(b"dataset,dim\ngoodreads,256\n")
    assert upload.verify(copy, man) == ["sha256 differs: flat.csv"]
    (copy / "flat.csv").unlink()
    assert upload.verify(copy, man) == ["missing: flat.csv"]


# ----- the CLI ---------------------------------------------------------------


def test_dry_run_touches_no_hub(tree, monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("--dry-run must not reach the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", boom)
    r = _run("--results", str(tree), "--path-in-repo", "c4", "--dry-run")
    assert r.exit_code == 0, r.output
    man = json.loads(r.output[r.output.index("{") :])
    assert man["path_in_repo"] == "c4" and man["citable"] is False
    assert "_parity" not in r.output and "_logs" not in r.output


def test_upload_commits_every_file_plus_the_two_generated_ones(tree, fake_hub):
    r = _run("--results", str(tree), "--repo-id", "u/r", "--path-in-repo", "c4")
    assert r.exit_code == 0, r.output
    assert fake_hub.created == {
        "repo_id": "u/r", "repo_type": "dataset", "private": True, "exist_ok": True,
    }  # fmt: skip
    (commit,) = fake_hub.commits
    paths = [op.path_in_repo for op in commit["operations"]]
    assert paths == [
        "c4/filter/goodreads-d128.jsonl",
        "c4/filter/goodreads-d128.samples.jsonl",
        "c4/flat.csv",
        "c4/MANIFEST.json",
        "README.md",
    ]
    assert "NOT CITABLE" in commit["commit_message"]


def test_public_is_never_the_default(tree, fake_hub):
    assert _run("--results", str(tree), "--repo-id", "u/r").exit_code == 0
    assert fake_hub.created["private"] is True
    assert _run("--results", str(tree), "--repo-id", "u/r", "--public").exit_code == 0
    assert fake_hub.created["private"] is False


def test_empty_and_missing_trees_fail_loudly(tmp_path):
    assert _run("--results", str(tmp_path / "nope")).exit_code != 0
    scratch = tmp_path / "results" / "_logs"
    scratch.mkdir(parents=True)
    (scratch / "x.log").write_text("noise")
    r = _run("--results", str(scratch.parent))
    assert r.exit_code != 0 and "no files to upload" in r.output


def test_readme_is_rebuilt_from_the_manifests_already_in_the_repo(tree, fake_hub, monkeypatch):
    """A second upload must not drop the first subtree from the front page."""
    old = upload.manifest(tree, upload.files(tree), "c4", None)
    old["source"] = "/elsewhere/c4"
    fetched = Path(tree.parent / "old.json")
    fetched.write_text(json.dumps(old))
    fake_hub.repo_files = ["c4/MANIFEST.json", "c4/flat.csv", "README.md"]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kw: str(fetched))
    assert _run("--results", str(tree), "--repo-id", "u/r", "--path-in-repo", "b3").exit_code == 0
    (readme_op,) = [o for o in fake_hub.commits[0]["operations"] if o.path_in_repo == "README.md"]
    body = readme_op.path_or_fileobj.decode()
    assert "## `c4`" in body and "## `b3`" in body and "/elsewhere/c4" in body
