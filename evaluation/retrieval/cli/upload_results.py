"""Upload all retrieval eval results to HuggingFace.

Mirrors the local ``results/`` layout (staged by ``stage_results``):
  <dataset>/<name>.json        (concat of per-algo rows for that config)
  <dataset>/<name>.yaml        (the YAML config used for that run)
  README.md                    (per-kind tables + optional campaign notes)

Skips ``*.perkernel/`` (per-algo originals, redundant with combined) and
``_runlogs/`` (campaign-internal). Idempotent: re-runs replace files.

Campaign-specific prose (section descriptions, caveats) belongs in a
markdown file passed via ``--notes-file``; it is appended verbatim to the
generated README.

Usage:
  uv run upload-results --repo-id <user/repo> [--notes-file notes.md]
                        [--private | --public] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = EVAL_DIR / "results"
STAGE_DIR = EVAL_DIR / "upload_staging"

REPO_TYPE = "dataset"


def _find_combined_jsons() -> list[Path]:
    """Combined JSON results: top-level files under <dataset>/<name>.json.

    Excludes per-kernel subdirs (``*.perkernel/*``) and runlog dirs (``_*``).
    """
    out: list[Path] = []
    for p in RESULTS_DIR.rglob("*.json"):
        if any(part.endswith(".perkernel") for part in p.parts):
            continue
        if any(part.startswith("_") for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def _classify(rel_path: Path) -> str:
    """Bucket results by campaign type for README grouping."""
    parts = rel_path.parts
    name = parts[-1]
    if parts[0] == "deep_sweeps":
        return "deep_sweeps"
    if parts[0] == "yambda":
        return "yambda"
    if "filter" in name:
        return "filter"
    if "quality" in name:
        return "quality"
    return "other"


def stage_all() -> list[dict]:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    stats: list[dict] = []
    for json_path in _find_combined_jsons():
        rel = json_path.relative_to(RESULTS_DIR)
        dst_json = STAGE_DIR / rel
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dst_json)

        yaml_src = json_path.with_suffix(".yaml")
        yaml_copied = False
        if yaml_src.exists():
            shutil.copy2(yaml_src, dst_json.with_suffix(".yaml"))
            yaml_copied = True

        with open(json_path) as f:
            rows = json.load(f)
        impls = collections.Counter(
            r.get("impl") for r in rows if isinstance(r, dict)
        )
        stats.append({
            "path": str(rel),
            "kind": _classify(rel),
            "rows": len(rows),
            "impls": dict(impls),
            "yaml": yaml_copied,
        })
        print(f"  staged {rel}  rows={len(rows)}  impls={sorted(impls.keys())}  yaml={yaml_copied}")
    return stats


def write_readme(stats: list[dict], notes_file: Path | None) -> None:
    """Generated part: front matter + per-kind tables derived from ``stats``.

    Campaign-specific prose comes from ``notes_file`` (appended verbatim).
    """
    by_kind: dict[str, list[dict]] = collections.defaultdict(list)
    for s in stats:
        by_kind[s["kind"]].append(s)

    lines = [
        "---",
        "license: mit",
        "tags:",
        "- retrieval",
        "- benchmark",
        "- filtered-knn",
        "---",
        "",
        f"# Retrieval eval results ({dt.date.today().isoformat()})",
        "",
        "JSON outputs from the retrieval benchmark in `retrieve/evaluation/`.",
        "Each row is one (filter_kind, sweep, impl, backend, k, batch_size) cell.",
        "",
        "Each combined `<config>.json` has its YAML alongside (`<config>.yaml`) describing exactly the run that produced it.",
        "",
    ]

    for kind in sorted(by_kind):
        lines.append(f"## {kind}")
        lines.append("")
        lines.append("| Path | Rows | Impls |")
        lines.append("|---|---:|---|")
        for s in sorted(by_kind[kind], key=lambda x: x["path"]):
            impls = ", ".join(sorted(s["impls"].keys()))
            lines.append(f"| `{s['path']}` | {s['rows']} | {impls} |")
        lines.append("")

    if notes_file is not None:
        lines.append(notes_file.read_text())
        lines.append("")

    (STAGE_DIR / "README.md").write_text("\n".join(lines))


def upload(*, repo_id: str, private: bool, dry_run: bool) -> None:
    if dry_run:
        print("\n[dry-run] would upload:")
        total_bytes = 0
        for p in sorted(STAGE_DIR.rglob("*")):
            if p.is_file():
                rel = p.relative_to(STAGE_DIR)
                sz = p.stat().st_size
                total_bytes += sz
                print(f"  {rel} ({sz:,} bytes)")
        print(f"\n  total: {total_bytes:,} bytes")
        print(f"  → repo: {repo_id} (private={private})")
        return

    from huggingface_hub import HfApi  # lazy import

    api = HfApi()
    print(f"\ncreate_repo (exist_ok): {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    print(f"upload_folder: {STAGE_DIR} → {repo_id}")
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        commit_message=f"Upload results ({dt.date.today().isoformat()})",
    )
    print(f"\ndone: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-id", required=True, help="target HF dataset repo, e.g. user/retrieval-evals"
    )
    ap.add_argument(
        "--notes-file",
        type=Path,
        default=None,
        help="markdown appended verbatim to the generated README (campaign notes)",
    )
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument(
        "--private", dest="private", action="store_true", help="create the repo as private"
    )
    vis.add_argument(
        "--public", dest="private", action="store_false", help="create the repo as public (default)"
    )
    ap.set_defaults(private=False)
    ap.add_argument("--dry-run", action="store_true", help="stage files but don't upload")
    args = ap.parse_args()

    if args.notes_file is not None and not args.notes_file.exists():
        ap.error(f"--notes-file {args.notes_file} does not exist")

    stats = stage_all()
    write_readme(stats, args.notes_file)
    print(f"\nwrote {STAGE_DIR}/README.md")
    upload(repo_id=args.repo_id, private=args.private, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
