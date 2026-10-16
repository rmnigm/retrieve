"""Two ``capture.py`` output directories, cell by cell: the sha256 of every cell's ids and scores
must match; a cell whose hash differs is loaded and reported as scores-equal / ids-equal per
pool batch. Exit status 1 on any difference or a cell present on one side only.

    python compare.py BEFORE_DIR AFTER_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    mb = json.loads((before / "manifest.json").read_text())["cells"]
    ma = json.loads((after / "manifest.json").read_text())["cells"]
    tags = sorted(set(mb) | set(ma))
    n_cells = n_bad = 0
    for tag in tags:
        if tag not in mb or tag not in ma:
            print(f"{tag}: only in {'before' if tag in mb else 'after'}")
            n_bad += 1
            continue
        names = sorted(set(mb[tag]) | set(ma[tag]))
        bad = [n for n in names if mb[tag].get(n) != ma[tag].get(n)]
        n_cells += len(names)
        n_bad += len(bad)
        if not bad:
            continue
        cb = torch.load(before / f"{tag}.pt")
        ca = torch.load(after / f"{tag}.pt")
        for n in bad:
            pairs = list(zip(cb.get(n, []), ca.get(n, []), strict=False))
            s_eq = sum(torch.equal(x[1], y[1]) for x, y in pairs)
            i_eq = sum(torch.equal(x[0], y[0]) for x, y in pairs)
            print(
                f"{tag} {n}: scores equal {s_eq}/{len(pairs)}, ids equal {i_eq}/{len(pairs)}"
            )
    print(f"{len(tags)} lines, {n_cells} cells, {n_bad} differ")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
