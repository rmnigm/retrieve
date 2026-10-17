"""Split the archive's bytes into copies of files already in OWNER's Hub repos and bytes found nowhere else:
python -I hub_overlap.py OWNER LOCAL_ROOT [EXCLUDE_REPO ...]. Matches by content (LFS sha256, else git blob sha1
of the local file), across every dataset and model repo of OWNER, by file list only. Read-only."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

OWNER, ROOT, EXCLUDE = sys.argv[1], Path(sys.argv[2]), set(sys.argv[3:])
api = HfApi()


def git_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


known: dict[str, str] = {}
repo_bytes = {}
for kind, repos in (("dataset", api.list_datasets(author=OWNER)), ("model", api.list_models(author=OWNER))):
    for r in repos:
        if r.id in EXCLUDE:
            continue
        n = 0
        for f in api.list_repo_tree(r.id, repo_type=kind, recursive=True, expand=True):
            if isinstance(f, RepoFile):
                n += f.size
                known.setdefault(f.lfs.sha256 if f.lfs else f.blob_id, f"{kind}:{r.id}")
        repo_bytes[f"{kind}:{r.id}"] = n

manifest = json.loads((ROOT / "MANIFEST.json").read_text())
entries = manifest["files"] + [{"path": "MANIFEST.json", "bytes": (ROOT / "MANIFEST.json").stat().st_size}]
copied, new = defaultdict(int), defaultdict(int)
new_files = []
for e in entries:
    part = e["path"].split("/")[0] if "/" in e["path"] else "(top level)"
    src = known.get(e.get("sha256")) or known.get(git_sha1(ROOT / e["path"]))
    if src:
        copied[part] += e["bytes"]
    else:
        new[part] += e["bytes"]
        new_files.append((e["bytes"], e["path"]))
print(json.dumps({"repos_scanned_bytes": repo_bytes, "copied": copied, "new": new,
                  "copied_total": sum(copied.values()), "new_total": sum(new.values()),
                  "new_files": len(new_files)}, indent=1))
for b, p in sorted(new_files, reverse=True):
    print(f"NEW {b:>14,} {p}")
