"""Every record of a rerun tree against the FR record with the same key block: `quality` equal (the parity note aside,
it depends on which backend ran first) and the per-query sidecar arrays equal. Exit 1 on any difference or missing cell.

usage: compare_quality.py FR NEW   (FR holds the fetched fr/* subtrees, NEW is quality_subset.sh's --out)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
from bench import records  # noqa: E402

FR, NEW = Path(sys.argv[1]), Path(sys.argv[2])


def key(rec):
    return json.dumps(records.key_block(rec), sort_keys=True)


def quality(rec):
    return {k: v for k, v in (rec.get("quality") or {}).items() if k != "parity"}


def main():
    fr = {}
    for sub in sorted(FR.iterdir()):
        if sub.is_dir():
            for p in records.record_files(sub):
                for r in records.read_records(p):
                    if r["status"] in ("ok", "partial") and r.get("quality"):
                        fr[key(r)] = (sub, r)
    new = [r for p in records.record_files(NEW) for r in records.read_records(p)]
    bad = 0
    for r in new:
        k = key(r)
        if r["status"] == "failed" or k not in fr:
            print("FAILED" if r["status"] == "failed" else "NO FR RECORD", k[:160])
            bad += 1
            continue
        sub, f = fr[k]
        if quality(r) != quality(f):
            print(
                "QUALITY DIFFERS",
                sub.name,
                k[:120],
                "\n  fr ",
                quality(f),
                "\n  new",
                quality(r),
            )
            bad += 1
        a, b = np.load(NEW / r["per_query"]), np.load(sub / f["per_query"])
        if set(a.files) != set(b.files) or any(
            not np.array_equal(a[n], b[n], equal_nan=True) for n in a.files
        ):
            print("SIDECAR DIFFERS", sub.name, k[:120])
            bad += 1
    print(f"{len(new)} records compared, {bad} differ or missing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
