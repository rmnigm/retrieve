"""R-RES: confirm.sh's records against the golden JSONs, every oracle metric at the record's k.

python compare_confirm.py OUT GOLDEN_DIR
"""

import json
import sys
from pathlib import Path

GOLDEN = {
    ("arxiv", "silvertorch"): "arxiv-d128-c0_maincat-silvertorch-triton.json",
    ("goodreads", "linr_v2"): "goodreads-d128-c0_genre-linr_v2-triton.json",
    ("goodreads", "linr_v3"): "goodreads-d128-c0_genre-linr_v3-triton.json",
}

out, gdir = Path(sys.argv[1]), Path(sys.argv[2])
print(
    "| run | cell | metric | golden | rerun | rerun − golden |\n|---|---|---|---|---|---|"
)
for f in sorted(out.glob("*/filter/*.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        g = {
            row["k"]: row
            for row in json.loads(
                (gdir / GOLDEN[(r["dataset"], r["algo"])]).read_text()
            )
        }
        for k in r["ks"]:
            for m in ("recall", "ndcg", "precision"):
                a, b = g[k][f"{m}@{k}"], r["quality"]["oracle"][f"{m}@{k}"]
                print(
                    f"| {f.parts[-3]} | {r['dataset']} {r['algo']} | {m}@{k} | {a:.10f} | {b:.10f} | {b - a:+.1e} |"
                )
