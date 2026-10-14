"""Profile table: per point, device us per forward of the filter-payload kernel, the scorer and
the total, exact before (final-freeze) / after (CLAUSE-OPT) and bloom (recheck off).

  python summarize.py OUT_DIR TAG [TAG ...]
"""

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return {r["key"]: r for r in json.loads(path.read_text())} if path.exists() else {}


out = Path(sys.argv[1])
print(
    "| point | exact filter before → after | exact total before → after | bloom filter | bloom total | scorer (exact after) |"
)
print("|---|---|---|---|---|---|")
for tag in sys.argv[2:]:
    base, opt = load(out / f"{tag}-prof-base.json"), load(out / f"{tag}-prof-opt.json")
    for key in opt or base:
        if not key.endswith("/exact"):
            continue
        b, o = base.get(key, {}), opt.get(key, {})
        bl = opt.get(key[: -len("exact")] + "bloom") or base.get(
            key[: -len("exact")] + "bloom", {}
        )
        f = lambda r, g: f"{r[g]:.0f}" if g in r else "–"  # noqa: E731
        print(
            f"| {tag} {key[:-6]} | {f(b, 'filter')} → {f(o, 'filter')} | {f(b, 'total')} → "
            f"{f(o, 'total')} | {f(bl, 'filter')} | {f(bl, 'total')} | {f(o, 'scorer')} |"
        )
