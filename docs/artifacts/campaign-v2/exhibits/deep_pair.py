"""Same-box re-time of a `deep` leg: per clause cell at seed 0, recall and p50 of NEW over OLD (README.md § deep_pair).

usage: deep_pair.py OUT.csv OLD.jsonl NEW.jsonl

`wide` marks B · n_probe >= 512, the cells ST-WIDE (v2.9) changed.
"""

import csv
import json
import sys

from load import perf, recall


def cells(path):
    return {
        (r["sweep"], r["params"]["n_lists"], r["params"]["n_probe"]): r
        for r in map(json.loads, open(path))
        if r["seed"] == 0 and r["filter_kind"] == "clause"
    }


def main():
    old, new = cells(sys.argv[2]), cells(sys.argv[3])
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sweep",
                "n_lists",
                "n_probe",
                "recall_old",
                "recall_new",
                "bs",
                "mode",
                "p50_old",
                "p50_new",
                "ratio",
                "wide",
            ]
        )
        for k in sorted(new):
            for bs in (1, 16):
                for m in ("graph", "eager"):
                    eo, en = perf(old[k], bs, 100, m), perf(new[k], bs, 100, m)
                    w.writerow(
                        [
                            *k,
                            recall(old[k]),
                            recall(new[k]),
                            bs,
                            m,
                            eo["median_ms"],
                            en["median_ms"],
                            round(en["median_ms"] / eo["median_ms"], 4),
                            bs * k[2] >= 512,
                        ]
                    )


if __name__ == "__main__":
    main()
