"""Publish one run's outputs to pinkmeme/eval-results under artifacts/gpu-followups/<run>/ in one commit with a
MANIFEST.json ({path, bytes, sha256} per file), read back and checked; prints the manifest's sha256. As
gpu-checks/upload.py: `bench upload` stops on the gpu-checks manifests already in the repo (they carry no
`path_in_repo`, which its README generator reads), so this bypasses the generated README.

    python upload.py DIR RUN
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "pinkmeme/eval-results"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(src: str, run: str) -> None:
    root = Path(src)
    named = {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}
    prefix = f"artifacts/gpu-followups/{run}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    entries = [{"path": k, "bytes": p.stat().st_size, "sha256": sha(p)} for k, p in named.items()]
    manifest = {"plan": "gpu-followups", "run": run, "path_in_repo": prefix, "git_head": head,
                "citable": False, "files": entries}  # fmt: skip
    body = (json.dumps(manifest, indent=1) + "\n").encode()
    ops = [CommitOperationAdd(f"{prefix}/{k}", str(p)) for k, p in named.items()]
    ops.append(CommitOperationAdd(f"{prefix}/MANIFEST.json", body))
    HfApi().create_commit(REPO, ops, repo_type="dataset", commit_message=f"gpu-followups: {run}")
    with tempfile.TemporaryDirectory(dir="/scratch/tmp") as d:
        for e in entries:
            got = hf_hub_download(REPO, f"{prefix}/{e['path']}", repo_type="dataset", local_dir=d)
            assert sha(Path(got)) == e["sha256"], e["path"]
    print(f"{prefix} files={len(entries) + 1} bytes={sum(e['bytes'] for e in entries) + len(body)}")
    print("manifest sha256", hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
