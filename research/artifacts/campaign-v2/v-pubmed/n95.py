"""n95 from an `n95` suite JSONL: recall_oracle@100 per n_probe, per seed and on the seed mean,
the smallest n_probe reaching 0.95 each way, and whether the grid brackets 0.95 on every seed.
`bench report`'s matched table cannot give it: it emits curves per timed batch size and
`perf: false` records have none.

    python n95.py RESULTS/n95/<dataset>-d<dim>.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

TARGET = 0.95
rec = defaultdict(dict)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["status"] == "ok":
        rec[r["params"]["n_probe"]][r["seed"]] = r["quality"]["oracle"]["recall@100"]
seeds = sorted({s for v in rec.values() for s in v})
print("pass_rate", r["pass_rate"], "sweep", r["sweep"], "n_queries_oracle", r["n_queries_oracle"])
print("n_probe", *[f"seed{s}" for s in seeds], "mean", sep="\t")
for x in sorted(rec):
    print(x, *[f"{rec[x][s]:.4f}" for s in seeds], f"{statistics.mean(rec[x].values()):.4f}", sep="\t")
xs = sorted(rec)
per_seed = {s: next((x for x in xs if rec[x][s] >= TARGET), None) for s in seeds}
mean = next((x for x in xs if statistics.mean(rec[x].values()) >= TARGET), None)
bracketed = all(rec[xs[0]][s] < TARGET and per_seed[s] is not None for s in seeds)
print(json.dumps({"n95_seed_mean": mean, "n95_per_seed": per_seed, "bracketed": bracketed}))
