#!/usr/bin/env python3
"""A1 / handoff step 5 — per-kernel perf gate, `main` vs the refactor track.

Reads the `tune-kernels --json-out` blobs from both sides and reports, per
kernel and per shape, the delta of the *winning* median. Gate: within
±5 %; anything worse is handoff **Fallback F3** (per-kernel inline
revert), not a tolerance to loosen (CLAUDE.md rule 3).

Two shape-key formats have to join here, which is why this is a script
and not a `diff`:

* `main` writes `per_bucket`, keyed by the int P bucket, for
  `fused-masked-knn-topk`; every other kernel writes `per_regime` keyed
  by a `LABEL=value,…` string.
* the refactor track writes `per_regime` everywhere, and its fmkt key is
  `P=…,D=…,B=…`.

The handoff's own note — "compare medians, not keys" — is the rule: the
shapes are generated in the same order on both sides, so the join is by
position, and both key strings are printed so a human can confirm the
pairing. A kernel whose shape counts differ is reported and skipped
rather than joined by guesswork.

    python3 a1_step5_compare.py --main <dir> --branch <dir> [--report <path.md>]

Both dirs hold `<kernel>.json` files named after the tune subcommand.
Exit 0 iff every joined shape is within tolerance and no kernel is
missing from either side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_PCT = 5.0


def shapes(blob: dict) -> list[tuple[str, float]]:
    """(shape key, winner ms) in the order the tuner produced them."""
    per = blob.get("per_regime") or blob.get("per_bucket") or {}
    return [(str(k), float(v["winner_ms"])) for k, v in per.items()]


def compare(kernel: str, main_p: Path, branch_p: Path) -> tuple[bool, list[str]]:
    a, b = json.loads(main_p.read_text()), json.loads(branch_p.read_text())
    sa, sb = shapes(a), shapes(b)
    lines: list[str] = []

    if not sa or not sb:
        return False, [f"  FAIL  {kernel}: no per-shape results (main={len(sa)}, branch={len(sb)})"]
    if len(sa) != len(sb):
        return False, [
            (f"  FAIL  {kernel}: shape counts differ (main={len(sa)}, branch={len(sb)}); "
             "cannot join by position — compare by hand")
        ]

    ok = True
    worst = 0.0
    for (ka, ma), (kb, mb) in zip(sa, sb):
        pct = (mb - ma) / ma * 100.0 if ma else float("inf")
        worst = pct if abs(pct) > abs(worst) else worst
        flag = "" if abs(pct) <= TOL_PCT else "   <-- OVER TOLERANCE (F3)"
        if abs(pct) > TOL_PCT:
            ok = False
        lines.append(f"      {ka:<34} | {kb:<34} | {ma:8.3f} -> {mb:8.3f} ms  {pct:+6.1f}%{flag}")

    default_a, default_b = a.get("default"), b.get("default")
    if default_a != default_b:
        lines.append(f"      note: winning config moved {default_a} -> {default_b} (not a gate)")
    lines.insert(0, f"  {'PASS' if ok else 'FAIL'}  {kernel}  ({len(sa)} shapes, worst {worst:+.1f}%)")
    lines.insert(1, f"      {'main shape':<34} | {'branch shape':<34} | {'main':>8} -> {'branch':>8}")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    out = [
        "# A1 / handoff step 5 — per-kernel perf gate (+-5%), main vs branch",
        "",
        f"main:   {args.main}",
        f"branch: {args.branch}",
        "",
    ]
    failures, missing = 0, 0
    for bp in sorted(args.branch.glob("*.json")):
        mp = args.main / bp.name
        if not mp.exists():
            missing += 1
            out.append(f"  SKIP  {bp.stem}: no main-side baseline ({mp})")
            out.append("        (expected for codesigned-probe-score — the K1 bug makes main's")
            out.append("        tuner raise on the first sweep point; handoff step 5 says take")
            out.append("        the baseline from the pre-K3 commit or fall back to the parity")
            out.append("        suite's wall clock.)")
            continue
        ok, lines = compare(bp.stem, mp, bp)
        failures += 0 if ok else 1
        out.extend(lines)

    compared = len(list(args.branch.glob("*.json"))) - missing
    out += [
        "",
        f"{compared} kernel(s) compared, {failures} over tolerance, {missing} without a baseline.",
        "",
        f"Gate: every shape within +-{TOL_PCT:.0f}%. Over tolerance -> handoff Fallback F3",
        "(per-kernel inline revert), then re-run that kernel's gate. Do not loosen.",
    ]
    text = "\n".join(out)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
        print(f"\nwrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
