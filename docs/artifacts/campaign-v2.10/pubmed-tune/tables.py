"""Readout of `tune-q` (PubMed 10 M d768, v2.10): recall_oracle@100 per (filter kind, sweep, n_lists) along n_probe; per n_lists the
smallest n_probe reaching 0.80 / 0.90 / 0.95 / 0.99 (`-` = not reached by 4096 or the n_lists / 4 cap) and the knee (smallest
n_probe within 0.01 of that n_lists' best). With --json it also writes the candidates [[sweep, filter_kind, n_lists, n_probe], ...].

    python tables.py [RESULTS] [--json OUT]   (default /scratch/campaign-v2.10/results)
"""

import collections
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
R = args[0] if args and not sys.argv[sys.argv.index(args[0]) - 1] == "--json" else "/scratch/campaign-v2.10/results"
out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
TARGETS = (0.80, 0.90, 0.95, 0.99)

curve = collections.defaultdict(dict)
pass_rate = {}
for line in open(f"{R}/tune-q/pubmed-d768.jsonl"):
    r = json.loads(line)
    p = r["params"]
    curve[r["filter_kind"], r["sweep"], p["n_lists"]][p["n_probe"]] = r["quality"]["oracle"]["recall@100"]
    pass_rate[r["sweep"]] = r["pass_rate"]

cands = set()
print("| filter | sweep (pass) | n_lists | recall@100 along n_probe | best | 0.80 | 0.90 | 0.95 | 0.99 | knee |")
print("|---|---|---|---|---|---|---|---|---|---|")
for fk, sw, nl in sorted(curve, key=lambda k: (k[0] != "clause", k[1], k[2])):
    c = dict(sorted(curve[fk, sw, nl].items()))
    best = max(c.values())
    hit = [next((n for n, v in c.items() if v >= t), None) for t in TARGETS]
    knee = next(n for n, v in c.items() if v >= best - 0.01)
    cands |= {(sw, fk, nl, n) for n in [*hit, knee] if n}
    pts = " ".join(f"{n}:{v:.3f}" for n, v in c.items())
    print(f"| {fk} | {sw} ({pass_rate[sw]:.4f}) | {nl} | {pts} | {best:.4f} | " + " | ".join(str(h or "-") for h in hit) + f" | {knee} |")
print(len(cands), "candidate cells")
if out:
    open(out, "w").write(json.dumps(sorted(cands)))
