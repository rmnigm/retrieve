"""Upload all retrieval eval results to HuggingFace.

Mirrors the local ``results/`` layout (staged by ``stage_results``):
  <dataset>/<name>.json        (concat of per-algo rows for that config)
  <dataset>/<name>.yaml        (the YAML config used for that run)
  README.md                    (campaign tables)

Skips ``*.perkernel/`` (per-algo originals, redundant with combined) and
``_runlogs/`` (campaign-internal). Idempotent: re-runs replace files.

Usage:
  uv run upload-results [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = EVAL_DIR / "results"
STAGE_DIR = EVAL_DIR / "upload_staging"

REPO_ID = "pinkmeme/retrieval-filter-evals-2026-05-23"
REPO_TYPE = "dataset"
PRIVATE = False


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


def write_readme(stats: list[dict]) -> None:
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
        "size_categories:",
        "- 10K<n<100K",
        "---",
        "",
        "# Retrieval eval results (campaign of 2026-05-23)",
        "",
        "JSON outputs from the retrieval benchmark in `/workspace/retrieve/evaluation/`.",
        "Each row is one (filter_kind, sweep, impl, backend, k, batch_size) cell.",
        "Schema reference: [pinkmeme/retrieval-filter-evals-2026-05-20](https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-20).",
        "",
        "Each combined `<config>.json` has its YAML alongside (`<config>.yaml`) describing exactly the run that produced it.",
        "",
    ]

    sections = [
        ("filter", "Filter evals", "Clause + bloom filter sweeps × 5 algos. `ks=[100, 500, 1000]`, `bs=[1, 8, 16]`, `users_limit=10000`."),
        ("quality", "Quality evals (arxiv + goodreads)", "No filter; recall vs full-catalog top-K. 4 algos. `ks=[100, 200, 400]`, `bs=[1]`, `backends=[triton, torch]`. arxiv uses pre-encoded queries (~10k); goodreads uses full test split (313k users) with no `users_limit`."),
        ("yambda", "Yambda quality evals", "No filter; full test users (yambda-500m: ~46k; yambda-5b: 459k). 4 algos. `ks=[100, 200, 400]`, `bs=[1]`, `backends=[triton, torch]`."),
        ("deep_sweeps", "Deep parameter sweeps", "Single-algo recall-vs-latency curves with filters on. silvertorch n_lists × n_probe layouts and linr_v3 `candidate_pool` sweep on d=128."),
    ]
    for key, title, desc in sections:
        if not by_kind.get(key):
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append("| Path | Rows | Impls |")
        lines.append("|---|---:|---|")
        for s in sorted(by_kind[key], key=lambda x: x["path"]):
            impls = ", ".join(sorted(s["impls"].keys()))
            lines.append(f"| `{s['path']}` | {s['rows']} | {impls} |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `linr_v2` only appears in filter results (it requires a filter context).")
    lines.append("- `torch_knn` was dropped from quality YAMLs in favor of `linr_v1_filter_mask` (also exact) + `linr_v4` (exact int8).")
    lines.append("- yambda-5b cache writes are skipped at runtime when they'd exceed disk budget; algos re-encode queries per-subprocess (~1–2 min overhead, negligible vs total).")
    lines.append("")

    (STAGE_DIR / "README.md").write_text("\n".join(lines))


def upload(dry_run: bool) -> None:
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
        print(f"  → repo: {REPO_ID} (private={PRIVATE})")
        return

    from huggingface_hub import HfApi  # lazy import

    api = HfApi()
    print(f"\ncreate_repo (exist_ok): {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)
    print(f"upload_folder: {STAGE_DIR} → {REPO_ID}")
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Extend with quality + deep_sweeps + yambda results (2026-05-23 campaign)",
    )
    print(f"\ndone: https://huggingface.co/datasets/{REPO_ID}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stage files but don't upload")
    args = ap.parse_args()

    stats = stage_all()
    write_readme(stats)
    print(f"\nwrote {STAGE_DIR}/README.md")
    upload(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
