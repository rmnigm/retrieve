"""Bench orchestrator: one subprocess per algorithm group.

Each ``bench_*.py`` file runs in a fresh ``pytest`` subprocess, ensuring the
CUDA context, allocator pool, and Triton autotune state are isolated between
algos. Within a subprocess, the autouse ``_isolate_cuda`` fixture in
``conftest.py`` clears state between cells.

Output: ``bench_results/<run_id>/{algo}__{cell_slug}__{impl}.json`` plus a
``manifest.json`` listing every record produced and any failures.

Usage:
    uv run python tests/bench/run.py
    uv run python tests/bench/run.py --algos silvertorch linr
    uv run python tests/bench/run.py --select "1048576"   # passes -k to pytest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "tests" / "bench"

ALGO_FILES = {
    "kernels":     "bench_kernels.py",
    "linr":        "bench_linr.py",
    "silvertorch": "bench_silvertorch.py",
}


def run(algos: list[str], select: str | None, rep_ms: float, render: bool) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "bench_results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "refs").mkdir(exist_ok=True)

    print(f"[bench] run_id={run_id}", flush=True)
    print(f"[bench] out_dir={out_dir}", flush=True)

    failures: list[dict] = []
    started = datetime.now(timezone.utc)

    for algo in algos:
        rel = ALGO_FILES[algo]
        path = BENCH_DIR / rel
        if not path.exists():
            print(f"[bench] skipping missing {rel}", flush=True)
            continue
        print(f"\n[bench] === {algo} ({rel}) ===", flush=True)

        cmd = [
            "uv", "run", "pytest", str(path),
            "--bench",
            f"--bench-run-id={run_id}",
            f"--bench-out-dir={out_dir}",
            f"--bench-rep={rep_ms}",
            "-v",
        ]
        if select:
            cmd += ["-k", select]

        env = {**os.environ, "BENCH_RUN_ID": run_id, "BENCH_OUT_DIR": str(out_dir)}
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            failures.append({"algo": algo, "returncode": proc.returncode})
            print(f"[bench] {algo} subprocess exited {proc.returncode}", flush=True)

    # Manifest
    records = sorted(p.name for p in out_dir.glob("*.json") if p.name != "manifest.json")
    manifest = {
        "run_id": run_id,
        "started":  started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "algos": algos,
        "select": select,
        "rep_ms": rep_ms,
        "records": records,
        "failures": failures,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\n[bench] wrote manifest: {len(records)} records, {len(failures)} algo failures", flush=True)

    if render:
        # Import from sibling file by path; avoids needing tests/ on sys.path.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bench_render", Path(__file__).parent / "render.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        md = mod.render(out_dir)
        md_path = out_dir / "bench.md"
        md_path.write_text(md)
        print(f"[bench] rendered markdown: {md_path}", flush=True)

    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--algos", nargs="+", default=list(ALGO_FILES),
        choices=list(ALGO_FILES),
        help="which algorithm groups to run (default: all)",
    )
    p.add_argument("--select", default=None, help="pytest -k expression to filter cells")
    p.add_argument("--rep-ms", type=float, default=200.0)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args(argv)
    return run(args.algos, args.select, args.rep_ms, render=not args.no_render)


if __name__ == "__main__":
    raise SystemExit(main())
