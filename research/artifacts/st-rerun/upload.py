"""Publish one dataset's rerun tree to pinkmeme/eval-results under artifacts/st-rerun/<dataset>/.

usage (from evaluation/): python ../research/artifacts/st-rerun/upload.py TREE DATASET

`bench upload`'s manifest (provenance + sha256 per file, NOT CITABLE), committed with the files, then read
back and checked. It does not regenerate the repo README: `bench upload` fails there on manifests other
uploads wrote without `path_in_repo` (KeyError in `upload.readme`).
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from bench import records
from bench.upload import _at, _download, files, manifest, verify
from huggingface_hub import CommitOperationAdd, HfApi

REPO = "pinkmeme/eval-results"
tree, ds = Path(sys.argv[1]).resolve(), sys.argv[2]
prefix = f"artifacts/st-rerun/{ds}"
records.aggregate(tree)
man = manifest(tree, files(tree), prefix, None)
ops = [CommitOperationAdd(_at(prefix, f["path"]), str(tree / f["path"])) for f in man["files"]]
man_bytes = json.dumps(man, indent=2).encode()
ops.append(CommitOperationAdd(_at(prefix, "MANIFEST.json"), man_bytes))
HfApi().create_commit(
    repo_id=REPO,
    repo_type="dataset",
    operations=ops,
    commit_message=f"st-rerun {ds}: {man['n_records']} records, NOT CITABLE",
)
with tempfile.TemporaryDirectory() as tmp:
    bad = verify(_download(REPO, prefix, tmp), man)
assert not bad, bad
print(f"{prefix}: {man['n_files']} files, {man['bytes']:,} bytes, {man['n_records']} records, round trip ok")
print(f"MANIFEST.json sha256 {hashlib.sha256(man_bytes).hexdigest()}")
