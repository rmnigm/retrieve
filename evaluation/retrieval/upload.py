"""``upload-results`` — mirror ``results/`` to a HuggingFace dataset repo (H §3.1: a mirror,
no staging layout). ``_logs/`` and ``_parity/`` (campaign scratch) are skipped; everything
else under ``--results`` goes up as-is, so the JSONL + samples sidecars keep their names.

    uv run upload-results --repo-id <user/repo> [--results results] [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
IGNORE = ["_*", "_*/**", "**/_*", "**/_*/**"]


def files(results: Path) -> list[Path]:
    return sorted(
        p
        for p in results.rglob("*")
        if p.is_file() and not any(part.startswith("_") for part in p.relative_to(results).parts)
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="upload-results", description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--repo-id", required=True, help="target HF dataset repo, e.g. user/retrieval-evals"
    )
    ap.add_argument("--results", type=Path, default=EVAL_DIR / "results")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = ap.parse_args()
    results = args.results if args.results.is_absolute() else EVAL_DIR / args.results
    listing = files(results)
    total = sum(p.stat().st_size for p in listing)
    for p in listing:
        print(f"  {p.relative_to(results)} ({p.stat().st_size:,} bytes)")
    print(f"{len(listing)} files, {total:,} bytes -> {args.repo_id} (private={args.private})")
    if args.dry_run:
        return 0
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(results),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=IGNORE,
        commit_message=f"results mirror {dt.date.today().isoformat()}",
    )
    print(f"done: https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
