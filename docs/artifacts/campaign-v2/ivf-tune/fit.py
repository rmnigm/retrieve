"""IVF-TUNE size rule from the two tuned datasets: n_lists as a multiple of sqrt(N), n95 as a
fraction of n_lists, each interpolated log-linearly in log N between the two picks (held at
the nearer pick outside them), then rounded to the nearest power of two. The tuned datasets
keep their measured picks exactly.

    python fit.py N_SMALL N_LISTS N_PROBE [N_BIG N_LISTS N_PROBE]

With the small dataset alone (part A, PubMed not yet tuned) both ratios are goodreads' and PubMed
is left out (its slot stays open).
"""

import json
import math
import sys

N = {
    "goodreads": 797_084,
    "arxiv": 2_988_996,
    "yfcc10m": 10_000_000,
    "pubmed": 10_000_000,
}
a = [float(x) for x in sys.argv[1:7]]
if len(a) == 3:  # one point: a flat rule; n1 only keeps interp defined
    a += [a[0] * 10, a[1] * 10**0.5, a[2] * 10**0.5]
    del N["pubmed"]
(n0, l0, p0), (n1, l1, p1) = a[:3], a[3:]
m = (l0 / math.sqrt(n0), l1 / math.sqrt(n1))
f = (p0 / l0, p1 / l1)


def interp(y, n):
    t = min(max((math.log(n) - math.log(n0)) / (math.log(n1) - math.log(n0)), 0.0), 1.0)
    return math.exp((1 - t) * math.log(y[0]) + t * math.log(y[1]))


def pow2(x):
    return 2 ** round(math.log2(x))


out = {}
for ds, n in N.items():
    tuned = (
        {n0: (l0, p0), n1: (l1, p1)}.get(n) if ds in ("goodreads", "pubmed") else None
    )
    if tuned:
        nl, npr, src = int(tuned[0]), int(tuned[1]), "tuned"
    else:
        nl = pow2(interp(m, n) * math.sqrt(n))
        npr, src = pow2(interp(f, n) * nl), "rule"
    out[ds] = {"N": n, "n_lists": nl, "n95": npr, "n_lists_per_sqrtN": round(nl / math.sqrt(n), 3),
               "n95_per_n_lists": round(npr / nl, 5), "source": src}  # fmt: skip
print(json.dumps({"multiple_of_sqrtN": [round(x, 3) for x in m],
                  "n95_fraction": [round(x, 5) for x in f], "values": out}, indent=1))  # fmt: skip
