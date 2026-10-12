"""PubMed diagnostic (b), CPU only: the saturated bloom recall read as 0.6699 at 408b1188 and 0.6727 at v2.9 on the same `bloomwidth`
cells. For every cell (backend, m_bits, k_hash, seed) in both trees: the per-query sidecar arrays compared with np.array_equal, the
per-seed recall_oracle@100, and the mean over seeds; plus the default width's seed-0 recall in every other PubMed bloom record found.

    python bloom_seeds.py OLD_RESULTS NEW_RESULTS [MORE_RESULTS ...]
"""

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np


def load(root, suite="bloomwidth"):
    out = {}
    path = Path(root) / suite / "pubmed-d768.jsonl"
    for line in open(path) if path.exists() else []:
        r = json.loads(line)
        p = r["params"]
        out[r["backend"], p.get("m_bits", 0), p["k_hash"], r["seed"]] = r
    return out


old, new = load(sys.argv[1]), load(sys.argv[2])
same = [k for k in old if all(np.array_equal(a, b, equal_nan=True) for a, b in zip(
    np.load(Path(sys.argv[1]) / old[k]["per_query"]).values(), np.load(Path(sys.argv[2]) / new[k]["per_query"]).values()))]  # fmt: skip
print(f"cells {len(old)} / {len(new)}; per-query arrays identical in {len(same)}")
print("cell | code_version old / new | recall_oracle@100 seeds 0 / 1 / 2 (old) | (new) | mean")
for b, m, kh in sorted({k[:3] for k in old}):
    o = [old[b, m, kh, s]["quality"]["oracle"]["recall@100"] for s in (0, 1, 2)]
    n = [new[b, m, kh, s]["quality"]["oracle"]["recall@100"] for s in (0, 1, 2)]
    cv = old[b, m, kh, 0]["env"]["code_version"][:8], new[b, m, kh, 0]["env"]["code_version"][:8]
    print(f"{b} {m} {kh} | {cv[0]} / {cv[1]} | {' / '.join(f'{x:.4f}' for x in o)} | {' / '.join(f'{x:.4f}' for x in n)} | {st.mean(n):.4f}")
for root in sys.argv[3:]:
    for k, r in sorted(load(root, "bloomwidth-timed").items()):
        if k[1] in (0, 1024) and k[3] == 0:
            print(root, "bloomwidth-timed", k, r["env"]["code_version"][:8], f"{r['quality']['oracle']['recall@100']:.4f}")
