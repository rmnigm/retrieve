"""Publish one check's raw outputs to pinkmeme/eval-results under artifacts/gpu-checks/<check>/ in
one commit with a MANIFEST.json ({path, bytes, sha256} per file); prints the manifest's sha256.
The records are session records, not harness-schema ones, so ``bench upload`` (which aggregates
``results.parquet`` from harness records) does not apply.

    python upload.py /scratch/gpu-checks/out/check1 check1 [extra files...]
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


def main(src: str, check: str, *extra: str) -> None:
    root = Path(src)
    named = {
        str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()
    }
    named |= {f"logs/{Path(x).name}": Path(x) for x in extra}
    prefix = f"artifacts/gpu-checks/{check}"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    files = list(named.values())
    entries = [
        {"path": k, "bytes": p.stat().st_size, "sha256": sha(p)}
        for k, p in named.items()
    ]
    manifest = {"plan": "gpu-checks", "check": check, "git_head": head, "citable": False,
                "files": entries}  # fmt: skip
    body = (json.dumps(manifest, indent=1) + "\n").encode()
    ops = [
        CommitOperationAdd(f"{prefix}/{e['path']}", str(p))
        for e, p in zip(entries, files)
    ]
    ops.append(CommitOperationAdd(f"{prefix}/MANIFEST.json", body))
    api = HfApi()
    api.create_commit(
        REPO, ops, repo_type="dataset", commit_message=f"gpu-checks: {check}"
    )
    with tempfile.TemporaryDirectory(dir="/scratch/tmp") as d:
        for e in entries:
            got = hf_hub_download(
                REPO, f"{prefix}/{e['path']}", repo_type="dataset", local_dir=d
            )
            assert sha(Path(got)) == e["sha256"], e["path"]
    print(
        f"{prefix} files={len(entries) + 1} bytes={sum(e['bytes'] for e in entries) + len(body)}"
    )
    print("manifest sha256", hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:])
