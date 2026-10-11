"""R-RES: the two re-derived golden cells — run 1 against run 2, against the cells they
replace, and V2 against V1's golden at @100 / @500 (both exact, both fp32 now).

    python golden_check.py OUT GOLDEN_DIR

OUT holds ``run1/`` and ``run2/`` from golden_rederive.sh; GOLDEN_DIR is the committed
``evaluation/golden`` *before* the two files are replaced.
"""

import json
import sys
from pathlib import Path

QUALITY = ("recall", "ndcg", "precision", "mrr")
CELLS = ("linr_v2", "linr_v3")


def rows(path: Path) -> dict[tuple[int, int], dict]:
    return {(r["k"], r["batch_size"]): r for r in json.loads(path.read_text())}


def cols(r: dict) -> dict[str, float]:
    return {f"{m}@{r['k']}": r[f"{m}@{r['k']}"] for m in QUALITY} | {
        "n_users_kept": r["n_users_kept"]
    }


out, gdir = Path(sys.argv[1]), Path(sys.argv[2])
v1 = rows(gdir / "goodreads-d128-c0_genre-linr_v1_filter_mask-triton.json")
for algo in CELLS:
    name = f"goodreads-d128-c0_genre-{algo}-triton.json"
    r1, r2, old = (
        rows(out / "run1" / name),
        rows(out / "run2" / name),
        rows(gdir / name),
    )
    same = all(cols(r1[key]) == cols(r2[key]) for key in r1) and r1.keys() == r2.keys()
    print(
        f"\n## {algo}: {len(r1)} rows; run 1 == run 2 on every quality column: {same}"
    )
    keys = {key: sorted(r1[key]) for key in r1}
    print(
        f"   columns equal to the old file's: {all(keys[key] == sorted(old[key]) for key in old)}"
    )
    print(
        "| k | metric | old golden | new (run 1) | new − old | V1 golden | new − V1 |"
    )
    print("|---|---|---|---|---|---|---|")
    for k in (100, 500, 1000):
        for m in QUALITY[:3]:
            c = f"{m}@{k}"
            o, n, w = old[(k, 1)][c], r1[(k, 1)][c], v1[(k, 1)][c]
            vs_v1 = f"{w:.10f} | {n - w:+.1e}" if algo == "linr_v2" else "- | -"
            print(f"| {k} | {m} | {o:.10f} | {n:.10f} | {n - o:+.1e} | {vs_v1} |")
    spread = {
        f"{c}@{k}": {r1[(k, bs)][f"{c}@{k}"] for bs in (1, 8, 16)}
        for k in (100, 500, 1000)
        for c in QUALITY
    }
    print(
        f"   quality identical across batch sizes: {all(len(v) == 1 for v in spread.values())}"
    )
