"""D3 PubMed bloomwidth (C4) and C7 seed tables from a v2.9 results tree: mean ± sd over seeds per width, and the
official / Triton eager ratio per (k, n_probe, bs) with a t 95 % CI over seeds.

    python tables.py [RESULTS]   (default /scratch/campaign-v2.9/results)
"""

import collections
import json
import statistics as st
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "/scratch/campaign-v2.9/results"
T95 = {2: 12.706, 3: 4.303}


def load(suite):
    return [json.loads(line) for line in open(f"{R}/{suite}/pubmed-d768.jsonl")]


def ms(v):
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


q = collections.defaultdict(list)
for r in load("bloomwidth"):
    p, o = r["params"], r["quality"]["oracle"]
    q[r["backend"], p.get("m_bits", 0), p["k_hash"]].append((o["recall@100"], o["recall@1000"], r["bloom_fp_rate"], r["index_mib"]))
print("bloomwidth: backend m_bits k_hash | n | recall_oracle@100 | @1000 | bloom_fp_rate | index_mib")
for key in sorted(q):
    v = q[key]
    print(key, len(v), ms([x[0] for x in v]), ms([x[1] for x in v]), f"{st.mean(x[2] for x in v):.2e}", round(v[0][3], 1))

t = collections.defaultdict(list)
for r in load("bloomwidth-timed"):
    d = {(x["mode"], x["bs"]): x["median_ms"] for x in r["perf"] if x["k"] == 100}
    t[r["backend"], r["params"].get("m_bits", 0)].append((r["quality"]["oracle"]["recall@100"], d["eager", 16], d.get(("graph", 16)), bool(r["unstable"])))
print("bloomwidth-timed k 100 bs 16 k_hash 5: backend m_bits | n | recall | eager median ms | graph median ms | unstable")
for key in sorted(t):
    v = t[key]
    print(key, len(v), ms([x[0] for x in v]), ms([x[1] for x in v]), ms([x[2] for x in v]) if v[0][2] else "-", sum(x[3] for x in v))

c = collections.defaultdict(dict)
for r in load("filter"):
    if r["algo"] != "silvertorch":
        continue
    for x in r["perf"]:
        if x["mode"] == "eager":
            c[x["k"], r["params"]["n_probe"], x["bs"], r["seed"]][r["backend"]] = x["median_ms"]
rat = collections.defaultdict(list)
for (k, n_probe, bs, _), d in sorted(c.items()):
    rat[k, n_probe, bs].append(d["official"] / d["triton"])
print("C7 eager official / triton: k n_probe bs | per seed | mean ± t95 half-width")
for key in sorted(rat):
    v = rat[key]
    print(key, [round(x, 3) for x in v], f"{st.mean(v):.3f} ± {T95[len(v)] * st.stdev(v) / len(v) ** 0.5:.3f}" if len(v) > 1 else "")
