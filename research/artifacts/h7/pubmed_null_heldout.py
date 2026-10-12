"""H7 gate on the real records: every d1/pubmed record whose held-out side scored no row
(``n == 0``, written as 0.0 before H7) aggregates to null held-out metrics, and no other
record changes.

    python pubmed_null_heldout.py EVALUATION_DIR RESULTS_DIR
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from bench import records  # noqa: E402

results = Path(sys.argv[2])
recs = records.latest(results)
empty = {
    records.record_key(r)
    for r in recs
    if ((r.get("quality") or {}).get("heldout") or {}).get("n") == 0
}
rows = records.read_table(records.aggregate(results, results / "h7.parquet"))
null_rows = [
    r for r in rows if r.get("heldout_recall@100") is None and r["status"] != "failed"
]
cells = sorted({(r["sweep"], r["algo"], r["backend"], r["params"]) for r in null_rows})
for c in cells:
    print(*c)
print(
    f"{len(recs)} records, {len(empty)} with an empty held-out side; null cells: {len(cells)}"
)
assert len(cells) == len(empty) > 0 and all(r["heldout_n"] == 0 for r in null_rows)
