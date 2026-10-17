"""Check a Hub copy of the archive against the local MANIFEST.json without downloading the payload:
python -I verify_hub.py REPO_ID LOCAL_ROOT. Every manifest entry must be on the Hub with its size; LFS files
are compared by the Hub's sha256, non-LFS files by git blob sha1 of the local file (MANIFEST.json itself
included). Extra Hub files are listed, .gitattributes excepted. Read-only; exit 1 on any problem."""

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

REPO, ROOT = sys.argv[1], Path(sys.argv[2])


def git_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


manifest = json.loads((ROOT / "MANIFEST.json").read_text())
remote = {
    f.path: f
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True)
    if isinstance(f, RepoFile)
}
bad = 0
for e in manifest["files"] + [{"path": "MANIFEST.json", "bytes": (ROOT / "MANIFEST.json").stat().st_size}]:
    r = remote.pop(e["path"], None)
    if r is None:
        print(f"missing: {e['path']}")
        bad += 1
    elif r.size != e["bytes"]:
        print(f"size {r.size} != {e['bytes']}: {e['path']}")
        bad += 1
    elif r.lfs is not None:
        if "sha256" not in e or r.lfs.sha256 != e["sha256"]:
            print(f"lfs sha256 differs: {e['path']}")
            bad += 1
    elif r.blob_id != git_sha1(ROOT / e["path"]):
        print(f"blob sha1 differs: {e['path']}")
        bad += 1
remote.pop(".gitattributes", None)
for p in sorted(remote):
    print(f"extra on hub: {p}")
    bad += 1
print(f"{len(manifest['files']) + 1} files checked, {bad} problems")
sys.exit(1 if bad else 0)
