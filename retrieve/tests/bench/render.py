"""Render a directory of bench JSON records to a markdown table.

Usage:
    python -m tests.bench.render <run_dir> [--out bench.md]

Produces tables grouped by ``algo``, with one row per (cell, impl). The "vs"
verdict column is computed at render time by comparing each impl's median
against the cell's first non-Triton baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _verdict(impl_ms: float, base_ms: float, threshold: float = 0.95) -> str:
    if base_ms <= 0:
        return "?"
    ratio = impl_ms / base_ms
    if ratio <= threshold:
        return "✅"
    if ratio >= 1 / threshold:
        return "❌"
    return "➖"


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}"


def _baseline_impl(impls: list[str]) -> str | None:
    """Pick a baseline impl among a cell's records: prefer 'torch', then any 'composed_*'."""
    for cand in ("torch", "composed_ivf_bloom", "composed_ivf"):
        if cand in impls:
            return cand
    for impl in impls:
        if impl.startswith("composed"):
            return impl
    return None


def render(run_dir: Path) -> str:
    records: list[dict] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        with path.open() as f:
            records.append(json.load(f))

    if not records:
        return "# Benchmark results\n\n_(no records)_\n"

    by_algo: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_algo[r["algo"]].append(r)

    out = ["# Benchmark results", ""]

    # Header summary
    first = records[0]
    out.append(f"_run id_: `{first['run_id']}`")
    out.append(
        f"_device_: `{first['device'].get('name', '?')}` "
        f"({first['device'].get('memory_total_mib', '?')} MiB)"
    )
    out.append(
        f"_versions_: torch=`{first['versions'].get('torch', '?')}` "
        f"triton=`{first['versions'].get('triton', '?')}` "
        f"git=`{first['versions'].get('git_sha', '?')}`"
    )
    out.append("")

    for algo in sorted(by_algo):
        out.append(f"## {algo}")
        out.append("")
        out.append(
            "| cell | impl | median (ms) | p20 | p80 "
            "| index (MiB) | fwd peak (MiB) | transient (MiB) "
            "| vs baseline | extra |"
        )
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")

        cells: dict[str, list[dict]] = defaultdict(list)
        for r in by_algo[algo]:
            cells[r["cell"]].append(r)

        for cell in sorted(cells):
            rows = cells[cell]
            impls = [r["impl"] for r in rows]
            base_impl = _baseline_impl(impls)
            base_ms = next(
                (r["timing"]["median_ms"] for r in rows if r["impl"] == base_impl),
                None,
            )
            for r in rows:
                t = r["timing"]
                m = r["memory"]
                if r["impl"] == base_impl or base_ms is None:
                    vs = "baseline"
                else:
                    vs = _verdict(t["median_ms"], base_ms)
                extra_pairs = []
                for k, v in r.get("extra", {}).items():
                    extra_pairs.append(f"{k}={v}")
                if r.get("correctness", {}).get("correct") is False:
                    extra_pairs.append("**INCORRECT**")
                extra_str = " ".join(extra_pairs)
                out.append(
                    f"| {r['cell']} | {r['impl']} "
                    f"| {t['median_ms']:.3f} | {t['p20_ms']:.3f} | {t['p80_ms']:.3f} "
                    f"| {_mib(m['index_bytes'])} | {_mib(m['forward_peak_bytes'])} "
                    f"| {_mib(m['transient_peak_bytes'])} "
                    f"| {vs} | {extra_str} |"
                )
        out.append("")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None, help="output markdown file (default: stdout)")
    args = p.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"error: not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    md = render(args.run_dir)
    if args.out:
        args.out.write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
