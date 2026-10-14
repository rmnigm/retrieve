"""Profile tables from clause_bench.py --mode profile JSONs.

  python summarize.py OUT_DIR TAG [TAG ...]    # time: exact before → after, bloom
  python summarize.py --bytes OUT_DIR TAG ...  # bytes per probed doc and achieved GB/s

Device us per forward of the filter-payload kernel ("filter") and the whole search; exact
before = final-freeze (``TAG-prof-base.json``), after = CLAUSE-OPT (``TAG-prof-opt.json``);
bloom = Meta's index, recheck off, from the after run. Bytes from bytes_model.py: exact = the
32-byte sectors the warps request (after), int64 rows (before); bloom = k words of 8 bytes per
64 docs and term. GB/s = those bytes / the filter kernel's time.
"""

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return {r["key"]: r for r in json.loads(path.read_text())} if path.exists() else {}


def fmt(r: dict, g: str) -> str:
    return f"{r[g]:.0f}" if g in r else "–"


def times(out: Path, tags: list[str]) -> None:
    print("| point | exact filter before → after | exact total before → after | bloom filter "
          "| bloom total |")  # fmt: skip
    print("|---|---|---|---|---|")
    for tag in tags:
        base, opt = (
            load(out / f"{tag}-prof-base.json"),
            load(out / f"{tag}-prof-opt.json"),
        )
        for key in opt:
            if not key.endswith("/exact"):
                continue
            b, o, bl = base.get(key, {}), opt[key], opt.get(key[:-5] + "bloom", {})
            print(f"| {tag} {key[:-6]} | {fmt(b, 'filter')} → {fmt(o, 'filter')} | "
                  f"{fmt(b, 'total')} → {fmt(o, 'total')} | {fmt(bl, 'filter')} | "
                  f"{fmt(bl, 'total')} |")  # fmt: skip


def bytes_table(out: Path, tags: list[str]) -> None:
    print("| point | probed docs | B/doc exact before / after / bloom | filter us exact / bloom "
          "| GB/s exact / bloom |")  # fmt: skip
    print("|---|---|---|---|---|")
    for tag in tags:
        opt = load(out / f"{tag}-prof-opt.json")
        if "peak" in opt:
            print(f"| {tag} read peak | | | | {opt['peak']['read_GBps']:.0f} |")
        for key, o in opt.items():
            if not key.endswith("/exact") or "docs" not in o:
                continue
            bl = opt[key[:-5] + "bloom"]
            d = o["docs"]
            gbs_e = o["exact_sector_B"] / (o["filter"] * 1e3)
            gbs_b = o["bloom_B"] / (bl["filter"] * 1e3)
            print(
                f"| {tag} {key[:-6]} | {d:,.0f} | {o['exact_before_B'] / d:.1f} / "
                f"{o['exact_sector_B'] / d:.1f} / {o['bloom_B'] / d:.2f} | "
                f"{o['filter']:.0f} / {bl['filter']:.0f} | {gbs_e:.0f} / {gbs_b:.0f} |"
            )


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--bytes":
        bytes_table(Path(args[1]), args[2:])
    else:
        times(Path(args[0]), args[1:])
