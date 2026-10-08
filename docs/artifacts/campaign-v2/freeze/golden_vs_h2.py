"""Gate 3 of the campaign-v2 freeze: the golden cells rerun on the integrated tree against the H2
rerun's records (Hub ``artifacts/h2``, code_version ``c0e42d1``), every quality metric.

    python3 golden_vs_h2.py NEW_DIR H2_DIR [--seed 0]

Both dirs hold ``filter/{goodreads,arxiv}-d128.jsonl``. Records match on the key block without
``code_version``; the last record per key wins. Every numeric leaf of ``quality`` (oracle and
held-out, at every k both carry) is compared; the gate is max |diff| == 0 on every matched cell.
An H2 cell with no new record, or a new record not ``ok``/``partial``, is a FAIL. The residuals
against the golden JSON (``--golden evaluation/golden``) reuse c4_gate.py's quality check on the
new records; c4_gate.py itself reads ``ok`` records only, and quality-only records are ``partial``.
They are printed with their |diff|, for comparison with the standing residuals. Exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEY = ("dataset", "dim", "inputs", "suite", "filter_kind", "sweep", "algo", "backend", "seed")


def load(root: Path, seed: int) -> dict[str, dict]:
    out = {}
    for f in sorted((root / "filter").glob("*-d128.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["seed"] == seed:
                    k = [r.get(x) for x in KEY] + [json.dumps(r.get("params") or {}, sort_keys=True)]
                    out[json.dumps(k)] = r
    return out


def leaves(q: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in (q or {}).items():
        if isinstance(v, dict):
            out |= leaves(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def vs_golden(new: dict[str, dict], golden_dir: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation-harness-v2"))
    import c4_gate

    golden = c4_gate.load_golden(golden_dir)
    rows: list = []
    seen = set()
    for r in new.values():
        key = (r["dataset"], r["dim"], r["filter_kind"], r["sweep"], r["algo"], r["backend"], r["seed"])
        if key in golden and c4_gate.at_defaults(r["algo"], r.get("params") or {}):
            seen.add(key)
            c4_gate.check_quality(rows, c4_gate.cell_name(r), r, golden[key], 1e-6)
    print("\n## vs golden JSON (c4_gate quality check, tol 1e-6)\n")
    for row in rows:
        print(f"{row['verdict']:5s} {row['cell']} {row['check']}: {row['detail']}")
    for key in sorted(set(golden) - seen):
        print(f"none  {golden[key]['file']}: no record at the algo defaults")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("h2", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--golden", type=Path)
    a = ap.parse_args()
    new, old = load(a.new, a.seed), load(a.h2, a.seed)
    fails = 0
    print("| cell | status | metrics | max abs diff | at |\n|---|---|---|---|---|")
    for key, o in sorted(old.items()):
        k = json.loads(key)
        cell = f"{k[0]} {k[6]} {k[7]} {k[9]}"
        n = new.get(key)
        if n is None:
            print(f"| {cell} | missing | | | FAIL |")
            fails += 1
            continue
        lo, ln = leaves(o["quality"]), leaves(n["quality"])
        common = sorted(set(lo) & set(ln))
        diffs = {m: abs(lo[m] - ln[m]) for m in common}
        worst = max(diffs, key=diffs.get) if diffs else "-"
        mx = diffs.get(worst, float("nan"))
        bad = n["status"] not in ("ok", "partial") or not common or mx != 0.0
        fails += bad
        print(
            f"| {cell} | {n['status']} | {len(common)} (only old {len(set(lo) - set(ln))}, "
            f"only new {len(set(ln) - set(lo))}) | {mx:.1e} | {worst}{' FAIL' if bad else ''} |"
        )
    extra = sorted(set(new) - set(old))
    for key in extra:
        print(f"| {json.loads(key)} | new only | | | INFO |")
    print(f"\n{len(old)} H2 cells, {fails} FAIL, {len(extra)} new-only")
    if a.golden:
        vs_golden(new, a.golden)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
