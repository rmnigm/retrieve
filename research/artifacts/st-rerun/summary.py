"""Counts before and after the rerun, and the settings the paper names.

usage: python summary.py RESELECT_CSV > summary.md

"after" takes a setting's new outcome and verdict where the rerun covered it (reselect.py), its final-run
ones elsewhere: the safe settings, which the selection left out as below 0.7x with margin.
"""

import csv
import sys
from collections import Counter
from statistics import median

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    run = r["new_verdict"] != "not_run"
    r["after_outcome"] = r["new_outcome"] if run else r["old_outcome"]
    r["after_verdict"] = r["new_verdict"] if run else r["old_verdict"]
    r["after_ratio"] = r["new_ratio"] if run else r["old_ratio"]

W = {"silvertorch_faster": "W", "tie": "T", "linr_faster": "L"}


def counts(rs, side):
    out = []
    for name, keep in (
        ("within a quarter", lambda o: o == "within_quarter"),
        ("deeper only", lambda o: o == "deeper"),
        ("any depth", lambda o: o != "not_reached"),
    ):
        sel = [r for r in rs if keep(r[f"{side}_outcome"])]
        c = Counter(W[r[f"{side}_verdict"]] for r in sel)
        ratios = [float(r[f"{side}_ratio"]) for r in sel]
        out.append(f"{len(sel)}; {c['W']} / {c['T']} / {c['L']}; {median(ratios):.3f}" if sel else "0")
    out.append(str(sum(r[f"{side}_outcome"] == "not_reached" for r in rs)))
    return out


print("| population | side | within a quarter: n; W / T / L; median | deeper only | any depth | not reached |")
print("|---|---|---|---|---|---|")
for label, rs in (
    ("all 312 (Yambda excluded)", rows),
    ("the 112 at risk", [r for r in rows if r["at_risk"] == "True"]),
    ("k 100", [r for r in rows if r["k"] == "100"]),
    ("real filters", [r for r in rows if "-synth" not in r["dataset"]]),
):
    for side in ("old", "after"):
        print(f"| {label} | {'final run' if side == 'old' else 'rerun'} | " + " | ".join(counts(rs, side)) + " |")

flips = [r for r in rows if r["new_verdict"] not in ("not_run", r["old_verdict"])]
moved = [r for r in rows if r["new_outcome"] not in ("not_run", r["old_outcome"])]
print(f"\n{len(flips)} verdicts change, {len(moved)} outcomes change; "
      f"{sum(r['new_verdict'] != 'not_run' for r in rows)} settings rerun "
      f"({sum(r['new_verdict'] != 'not_run' and r['at_risk'] == 'True' for r in rows)} at risk).\n")
print("| setting | outcome | pick | recall | ratio | verdict | unstable |")
print("|---|---|---|---|---|---|---|")
for r in flips + [r for r in moved if r not in flips]:
    print(
        f"| {r['dataset']} {r['filter']} b{r['batch']} k{r['k']} {r['recall_target']} | {r['old_outcome']} → {r['new_outcome']} "
        f"| {r['old_pick'] or '—'} → {r['new_pick'] or '—'} | {r['old_recall'] or '—'} → {r['new_recall'] or '—'} "
        f"| {r['old_ratio'] or '—'} → {r['new_ratio'] or '—'} | {r['old_verdict']} → {r['new_verdict']} | {r['new_unstable']} |"
    )
NAMED = [  # the paper's three strict losses and four ties within its tie bands (NUMBERS.md)
    ("laion10m", "tags4", "64", "100", "0.95"),
    ("arxiv-synth", "p1", "64", "100", "0.95"),
    ("laion10m-synth", "p05", "64", "100", "0.95"),
    ("arxiv", "c3_nversions", "64", "1000", "0.95"),
    ("laion10m", "c0_domain", "1", "100", "0.9"),
    ("laion3m", "c0_domain", "16", "100", "0.95"),
    ("laion3m", "c0_domain", "16", "1000", "0.95"),
]
print("\nThe paper's three losses and four ties:\n")
print("| setting | old: outcome, ratio, verdict | rerun: outcome, pick, ratio, verdict | unstable |")
print("|---|---|---|---|")
for key in NAMED:
    r = next(r for r in rows if (r["dataset"], r["filter"], r["batch"], r["k"], r["recall_target"]) == key)
    print(
        f"| {' '.join(key)} | {r['old_outcome']}, {r['old_ratio']}, {r['old_verdict']} "
        f"| {r['new_outcome']}, {r['new_pick'] or '—'}, {r['new_ratio'] or '—'}, {r['new_verdict']} | {r['new_unstable']} |"
    )
