#!/usr/bin/env python3
"""A1 / handoff step 6 — golden-run diff, `main` vs the refactor track.

Compares the 11 golden cells produced from the `main` worktree against the
same 11 produced on this branch, per the pass criteria of
`docs/plans/refactor-validation-handoff.md` step 6:

1. quality columns (`recall@k`, `ndcg@k`) **byte-identical** on every row,
   joined on the explicit key `(filter_kind, sweep, impl, backend,
   batch_size, k)` — never on the `cell` string (H §1 verdict 4: it
   collides);
2. **additive columns only** — the branch may add `precision@k`, `mrr@k`
   and `extra.gpu` / `extra.torch` / `extra.commit`; nothing renamed or
   dropped, and `device == "cuda"` on both sides;
3. the `cell` key sets are identical;
4. latency within ~5 % — reported, never fatal: it is a timing
   observation on a shared box, and H §1 verdicts 1–2 explain why these
   particular numbers are not a target.

Exit 0 iff every cell passes 1–3. Criterion 4 is printed and summarised
but does not change the exit code; the run record quotes it.

    python3 a1_step6_diff.py --branch evaluation/golden \\
                             --main   evaluation/golden/_main \\
                             [--report <path.md>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUALITY_PREFIXES = ("recall@", "ndcg@")
ADDITIVE_PREFIXES = ("precision@", "mrr@")
LATENCY_COLS = ("median_ms", "p20_ms", "p80_ms")
LATENCY_TOL_PCT = 5.0

JOIN_KEY = ("filter_kind", "sweep", "impl", "backend", "batch_size", "k")


def row_key(r: dict) -> tuple:
    return tuple(r[c] for c in JOIN_KEY)


def load(path: Path) -> dict[tuple, dict]:
    rows = json.loads(path.read_text())
    out: dict[tuple, dict] = {}
    for r in rows:
        k = row_key(r)
        if k in out:
            raise SystemExit(f"{path}: duplicate join key {k} — the key is not unique")
        out[k] = r
    return out


def quality_cols(r: dict) -> dict[str, object]:
    return {c: v for c, v in r.items() if c.startswith(QUALITY_PREFIXES)}


def compare_cell(name: str, main_path: Path, branch_path: Path) -> tuple[bool, list[str]]:
    """Returns (passed, lines). `lines` are the human-readable findings."""
    lines: list[str] = []
    a = load(main_path)   # main
    b = load(branch_path)  # branch
    ok = True

    # --- criterion 1 prerequisite: same set of rows -------------------------
    if a.keys() != b.keys():
        only_main = sorted(a.keys() - b.keys())
        only_branch = sorted(b.keys() - a.keys())
        ok = False
        lines.append(f"    row sets differ: {len(only_main)} only-main, {len(only_branch)} only-branch")
        for k in only_main[:5]:
            lines.append(f"      only main:   {k}")
        for k in only_branch[:5]:
            lines.append(f"      only branch: {k}")

    shared = sorted(a.keys() & b.keys(), key=str)

    # --- criterion 1: quality byte-identical --------------------------------
    drifted = 0
    for k in shared:
        qa, qb = quality_cols(a[k]), quality_cols(b[k])
        missing = set(qa) - set(qb)
        if missing:
            ok = False
            lines.append(f"    {k}: branch dropped quality columns {sorted(missing)}")
            continue
        for col, va in qa.items():
            vb = qb[col]
            # Byte-identical, not approximate: these are the same computation
            # on the same oracle. `repr` catches a float that differs in the
            # last bit but prints the same at default precision.
            if repr(va) != repr(vb):
                ok = False
                drifted += 1
                if drifted <= 5:
                    lines.append(f"    {k}: {col} main={va!r} branch={vb!r}")
    if drifted > 5:
        lines.append(f"    ... and {drifted - 5} more quality drifts")

    # --- criterion 2: additive columns only ---------------------------------
    for k in shared[:1]:  # the schema is per-file, one row is enough
        new = set(b[k]) - set(a[k])
        gone = set(a[k]) - set(b[k])
        bad_new = {c for c in new if not c.startswith(ADDITIVE_PREFIXES)}
        if gone:
            ok = False
            lines.append(f"    columns dropped by the branch: {sorted(gone)}")
        if bad_new:
            ok = False
            lines.append(f"    unexpected new columns: {sorted(bad_new)}")
        elif new:
            lines.append(f"    additive columns (expected): {sorted(new)}")
        for side, r in (("main", a[k]), ("branch", b[k])):
            if r.get("device") != "cuda":
                ok = False
                lines.append(f"    {side} device={r.get('device')!r}, expected 'cuda'")
        extra_new = set(b[k].get("extra", {})) - set(a[k].get("extra", {}))
        if extra_new:
            lines.append(f"    extra.* added (expected gpu/torch/commit): {sorted(extra_new)}")

    # --- criterion 3: cell key sets identical -------------------------------
    cells_a = {r.get("cell") for r in a.values()}
    cells_b = {r.get("cell") for r in b.values()}
    if cells_a != cells_b:
        ok = False
        lines.append(f"    `cell` key sets differ: {sorted(cells_a ^ cells_b)[:6]}")

    # --- criterion 4: latency, informational --------------------------------
    worst = ("", 0.0)
    for k in shared:
        for col in LATENCY_COLS:
            va, vb = a[k].get(col), b[k].get(col)
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)) or va == 0:
                continue
            pct = (vb - va) / va * 100.0
            if abs(pct) > abs(worst[1]):
                worst = (f"{k} {col}: main={va:.4f} branch={vb:.4f}", pct)
    if worst[0]:
        flag = "" if abs(worst[1]) <= LATENCY_TOL_PCT else f"  >{LATENCY_TOL_PCT:.0f}% (informational)"
        lines.append(f"    worst latency delta {worst[1]:+.1f}%  [{worst[0]}]{flag}")

    lines.insert(0, f"  {'PASS' if ok else 'FAIL'}  {name}  ({len(shared)} rows joined)")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", type=Path, required=True, help="dir of this branch's golden JSONs")
    ap.add_argument("--main", type=Path, required=True, help="dir of the main worktree's JSONs")
    ap.add_argument("--report", type=Path, default=None, help="also write the findings here")
    args = ap.parse_args()

    branch_files = sorted(p for p in args.branch.glob("*.json"))
    if not branch_files:
        return _fail(f"no *.json under {args.branch}")

    out: list[str] = [
        "# A1 / handoff step 6 — golden-run diff, main vs branch",
        "",
        f"branch: {args.branch}",
        f"main:   {args.main}",
        "",
    ]
    failures = 0
    missing = 0
    for bp in branch_files:
        mp = args.main / bp.name
        if not mp.exists():
            missing += 1
            out.append(f"  SKIP  {bp.name}  (no main-side counterpart at {mp})")
            continue
        ok, lines = compare_cell(bp.name, mp, bp)
        failures += 0 if ok else 1
        out.extend(lines)

    n = len(branch_files) - missing
    out += [
        "",
        f"{n} cell(s) compared, {failures} FAIL, {missing} without a main-side counterpart.",
        "",
        "Criteria: 1 quality byte-identical, 2 additive columns only, 3 `cell` key",
        "sets identical (all fatal); 4 latency within ~5% (informational only).",
    ]
    text = "\n".join(out)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
        print(f"\nwrote {args.report}")

    if missing:
        print(f"\nincomplete: {missing} cell(s) have no main-side JSON", file=sys.stderr)
        return 1
    return 1 if failures else 0


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
