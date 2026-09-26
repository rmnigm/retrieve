import sys
from pathlib import Path

from huggingface_hub import HfApi

JOBS = [
    ("pinkmeme/eval-goodreads-work-id", Path("/data/goodreads-work-id/trainer"), None),
    ("pinkmeme/eval-yambda-500m", Path("/data/yambda-500m/trainer"), None),
    ("pinkmeme/eval-kuairand", Path("/data/kuairand"), ["train.parquet", "val.parquet", "prep_log.json"]),
]
dry = "--dry-run" in sys.argv
api = HfApi()
for repo, src, allow in JOBS:
    if api.dataset_info(repo).private is not True:
        sys.exit(f"STOP: {repo} is not private")
    files = [src / f for f in allow] if allow else sorted(p for p in src.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{repo} trainer/ <- {src} ({len(files)} files, {total} bytes, private)")
    for p in files:
        print(f"  + {p.name} {p.stat().st_size}")
    if dry:
        continue
    api.upload_folder(
        folder_path=str(src), repo_id=repo, repo_type="dataset", path_in_repo="trainer",
        allow_patterns=allow, commit_message="Upload trainer inputs",
    )
    print(f"  uploaded {repo}/trainer")
