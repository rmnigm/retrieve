"""Upload filter-eval results to HuggingFace.

Mirrors the structure of pinkmeme/retrieval-filter-evals-2026-05-20:
  <dataset>/<dim>-filter.json         (concat of per-algo result lists)
  configs/<dataset>/<dim>-filter.yaml (the YAML used for this campaign)
  README.md                            (table: config, rows, dropped algos)

Idempotent: safe to re-run if upload is interrupted (uses HfApi.upload_folder
with exist_ok=True on create_repo). Run via:

  uv run upload-results [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

from retrieval.results_io import load_rows

EVAL_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = EVAL_DIR / "results"
CONF_DIR = EVAL_DIR / "config"
STAGE_DIR = EVAL_DIR / "upload_staging"

REPO_ID = "pinkmeme/retrieval-filter-evals-2026-05-23"
REPO_TYPE = "dataset"
PRIVATE = False  # matches the 2026-05-20 campaign visibility

EXPECTED_ALGOS = [
    "silvertorch",
    "linr_v1_filter_mask",
    "linr_v2",
    "linr_v3",
    "linr_v4",
]
CONFIGS = [
    ("arxiv", "d64"),
    ("arxiv", "d128"),
    ("arxiv", "d256"),
    ("goodreads", "d64"),
    ("goodreads", "d128"),
    ("goodreads", "d256"),
]


def stage_one_config(dataset: str, dim: str) -> dict:
    """Concat per-algo JSONs for one config; copy YAML; return stats."""
    src_dir = RESULTS_DIR / dataset / f"{dim}-filter"
    out_path = STAGE_DIR / dataset / f"{dim}-filter.json"
    yaml_src = CONF_DIR / dataset / f"{dim}-filter.yaml"
    yaml_dst = STAGE_DIR / "configs" / dataset / f"{dim}-filter.yaml"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_dst.parent.mkdir(parents=True, exist_ok=True)

    present = []
    missing = []
    rows: list[dict] = []
    for algo in EXPECTED_ALGOS:
        p = src_dir / f"{algo}.json"
        data = load_rows(p)
        if data is None:
            if p.exists():
                print(f"  ! {p}: missing or unparseable — treating as missing", file=sys.stderr)
            missing.append(algo)
            continue
        rows.extend(data)
        present.append(algo)

    json.dump(rows, open(out_path, "w"))
    if yaml_src.exists():
        shutil.copy2(yaml_src, yaml_dst)
        yaml_copied = True
    else:
        yaml_copied = False

    impl_counts = collections.Counter(r.get("impl") for r in rows)
    return {
        "config": f"{dataset}/{dim}-filter",
        "rows": len(rows),
        "present": present,
        "missing": missing,
        "impl_counts": dict(impl_counts),
        "yaml_copied": yaml_copied,
    }


def write_readme(stats: list[dict]) -> None:
    lines = [
        "---",
        "license: mit",
        "tags:",
        "- retrieval",
        "- benchmark",
        "- filtered-knn",
        "size_categories:",
        "- 1K<n<10K",
        "---",
        "",
        "# Retrieval filter-eval results (campaign of 2026-05-23)",
        "",
        "JSON outputs from the retrieval benchmark in `/workspace/retrieve/evaluation/`.",
        "Each row is one (filter_kind, sweep, impl, backend, k, batch_size) cell;",
        "see [pinkmeme/retrieval-filter-evals-2026-05-20](https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-20) for the schema reference.",
        "",
        "Campaign vs. 2026-05-20: adds `linr_v4`, `ks=[100, 500, 1000]` (was `[100, 200, 400]`).",
        "",
        "| Config | Rows | Algos present | Algos dropped |",
        "|---|---:|---|---|",
    ]
    for s in stats:
        present = ", ".join(s["present"]) or "—"
        missing = ", ".join(s["missing"]) or "—"
        lines.append(f"| `{s['config']}` | {s['rows']} | {present} | {missing} |")
    lines.append("")
    lines.append("Configs used for the runs are mirrored under `configs/<dataset>/<dim>-filter.yaml`.")
    lines.append("")
    (STAGE_DIR / "README.md").write_text("\n".join(lines))


def upload(dry_run: bool) -> None:
    if dry_run:
        print("\n[dry-run] would upload:")
        for p in sorted(STAGE_DIR.rglob("*")):
            if p.is_file():
                rel = p.relative_to(STAGE_DIR)
                print(f"  {rel} ({p.stat().st_size:,} bytes)")
        print(f"  → repo: {REPO_ID} (private={PRIVATE})")
        return

    from huggingface_hub import HfApi  # lazy import so --dry-run works without network

    api = HfApi()
    print(f"\ncreate_repo (exist_ok): {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)
    print(f"upload_folder: {STAGE_DIR} → {REPO_ID}")
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Upload filter-eval results (campaign 2026-05-23)",
    )
    print(f"\ndone: https://huggingface.co/datasets/{REPO_ID}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="stage files but don't upload")
    ap.add_argument("--clean-stage", action="store_true", help="wipe upload_staging/ before staging")
    args = ap.parse_args()

    if args.clean_stage and STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    for dataset, dim in CONFIGS:
        print(f"\nstaging {dataset}/{dim}-filter ...")
        s = stage_one_config(dataset, dim)
        stats.append(s)
        print(f"  rows={s['rows']}  present={s['present']}  missing={s['missing']}  yaml_copied={s['yaml_copied']}")
        print(f"  impl_counts={s['impl_counts']}")

    write_readme(stats)
    print(f"\nwrote {STAGE_DIR}/README.md")

    upload(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())