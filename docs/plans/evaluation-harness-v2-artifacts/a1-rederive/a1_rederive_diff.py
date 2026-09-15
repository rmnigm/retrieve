#!/usr/bin/env python3
"""Quality diff: A1's golden cells vs the re-derived ones.

Both sides are the *same* harness (`origin/dev/a1-golden`); only the library
differs (A1: `70bafc4`'s `retrieve/`; re-derive: `4f52972`'s). So every row
joins on `(cell file, k, batch_size)` and any delta is a library delta.

    python3 a1_rederive_diff.py --old <dir> --new <dir> --report <md>

Exit 1 if a cell or row is missing on either side. A quality delta is NOT a
failure here — the point of the run is to measure it — so deltas are reported,
never gated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUALITY = ("recall", "ndcg", "precision", "mrr")
LATENCY = ("median_ms", "p20_ms", "p80_ms")
MEM = ("peak_mem_mib", "fwd_scratch_mib", "index_mem_mib")


def rows(d: Path) -> dict[tuple[str, int, int], dict]:
    out: dict[tuple[str, int, int], dict] = {}
    for f in sorted(d.glob("*.json")):
        for r in json.loads(f.read_text()):
            out[(f.stem, r["k"], r["batch_size"])] = r
    return out


def qcols(r: dict) -> dict[str, float]:
    return {c: v for c, v in r.items() if c.split("@")[0] in QUALITY}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    old, new = rows(a.old), rows(a.new)
    missing = sorted(set(old) ^ set(new))
    lines: list[str] = []
    lines.append("# A1 golden re-derive — quality diff\n")
    lines.append(f"old: `{a.old}` ({len(old)} rows)  \nnew: `{a.new}` ({len(new)} rows)\n")
    if missing:
        lines.append("## MISSING ROWS\n")
        for m in missing:
            lines.append(f"- `{m}` — only in {'old' if m in old else 'new'}")
        lines.append("")

    lines.append("## Quality deltas (new - old), rows where any column moved\n")
    lines.append("| cell | k | bs | column | old | new | delta |")
    lines.append("|---|---|---|---|---|---|---|")
    n_moved = 0
    moved_cells: dict[str, int] = {}
    for key in sorted(set(old) & set(new)):
        o, n = qcols(old[key]), qcols(new[key])
        assert set(o) == set(n), f"schema differs at {key}: {set(o) ^ set(n)}"
        for c in sorted(o):
            d = n[c] - o[c]
            if d != 0.0:
                n_moved += 1
                moved_cells[key[0]] = moved_cells.get(key[0], 0) + 1
                lines.append(
                    f"| {key[0]} | {key[1]} | {key[2]} | {c} | "
                    f"{o[c]:.9f} | {n[c]:.9f} | {d:+.3e} |"
                )
    if n_moved == 0:
        lines.append("| — | — | — | — | — | — | (none: every quality column bit-identical) |")
    lines.append("")

    lines.append("## Per-cell summary\n")
    lines.append("| cell | rows | quality cols moved | max |delta| | direction | n_users_kept old/new |")
    lines.append("|---|---|---|---|---|---|")
    for stem in sorted({k[0] for k in set(old) & set(new)}):
        ks = [k for k in sorted(set(old) & set(new)) if k[0] == stem]
        deltas = [
            qcols(new[k])[c] - qcols(old[k])[c] for k in ks for c in sorted(qcols(old[k]))
        ]
        nz = [d for d in deltas if d != 0.0]
        direction = (
            "—" if not nz else "down" if all(d < 0 for d in nz)
            else "up" if all(d > 0 for d in nz) else "mixed"
        )
        mx = max((abs(d) for d in nz), default=0.0)
        uk_o = {old[k].get("n_users_kept") for k in ks}
        uk_n = {new[k].get("n_users_kept") for k in ks}
        lines.append(
            f"| {stem} | {len(ks)} | {len(nz)}/{len(deltas)} | {mx:.3e} | {direction} | "
            f"{sorted(uk_o)} / {sorted(uk_n)} |"
        )
    lines.append("")

    lines.append("## Latency, for context only (not clock-controlled, not a gate)\n")
    lines.append("| cell | k | bs | median_ms old | new | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(set(old) & set(new)):
        o, n = old[key]["median_ms"], new[key]["median_ms"]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {o:.4f} | {n:.4f} | {n / o:.3f}× |")
    lines.append("")

    lines.append("## Memory columns\n")
    lines.append("| cell | k | bs | column | old | new |")
    lines.append("|---|---|---|---|---|---|")
    any_mem = False
    for key in sorted(set(old) & set(new)):
        for c in MEM:
            if old[key].get(c) != new[key].get(c):
                any_mem = True
                lines.append(
                    f"| {key[0]} | {key[1]} | {key[2]} | {c} | {old[key].get(c)} | {new[key].get(c)} |"
                )
    if not any_mem:
        lines.append("| — | — | — | — | — | (identical) |")
    lines.append("")

    a.report.write_text("\n".join(lines) + "\n")
    print(f"rows joined: {len(set(old) & set(new))}, quality columns moved: {n_moved}")
    for stem, c in sorted(moved_cells.items()):
        print(f"  {stem}: {c} column(s) moved")
    print(f"report -> {a.report}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
