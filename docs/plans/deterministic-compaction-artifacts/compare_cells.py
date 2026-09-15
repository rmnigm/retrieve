"""Compare golden-cell JSONs row by row on their quality columns.

    python compare_cells.py A.json B.json      # exit 1 if any quality column differs

Rows are matched on ``cell``; the columns compared are every ``recall@``, ``ndcg@``,
``precision@`` and ``mrr@`` key plus ``n_users_kept`` — the columns C4 gates. Latency columns
are printed for context only."""

import json
import sys

QUALITY = ("recall@", "ndcg@", "precision@", "mrr@", "n_users_kept")


def rows(path):
    return {r["cell"]: r for r in json.load(open(path))}


a, b = rows(sys.argv[1]), rows(sys.argv[2])
assert a.keys() == b.keys(), (a.keys(), b.keys())
bad = 0
for cell in a:
    ra, rb = a[cell], b[cell]
    for k in ra:
        if any(k.startswith(q) for q in QUALITY) and ra[k] != rb[k]:
            bad += 1
            print(f"{cell:40s} {k:16s} {ra[k]!r} != {rb[k]!r}  delta {abs(ra[k] - rb[k]):.3e}")
    print(f"{cell:40s} recall@k {ra['recall@' + str(ra['k'])]:.12f}  median_ms {ra['median_ms']:.4f} / {rb['median_ms']:.4f}")
print("quality columns:", "IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
sys.exit(1 if bad else 0)
