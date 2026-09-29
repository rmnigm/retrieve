"""Fix 3: compare two `_parity` spills of one cell (triton, official) rank by rank. Per pair: the
score diff where both are finite, rows with a divergence > THRESH, their finite-candidate
counts on each side, any finite score on an id == -1 slot (a masking leak), and whether the
diverging row's id sets differ (a candidate-set difference) or only its order does.

    python bloom_divergence.py TRITON.npz OFFICIAL.npz
"""

import sys

import numpy as np

THRESH = 0.01
t, o = np.load(sys.argv[1]), np.load(sys.argv[2])
ti, ts, oi, os_ = t["ids"], t["scores"], o["ids"], o["scores"]
print("backends", t["backend"], o["backend"], "shape", ti.shape)
for name, ids, sc in (("triton", ti, ts), ("official", oi, os_)):
    leak = np.isfinite(sc) & (ids < 0)
    hole = ~np.isfinite(sc) & (ids >= 0)
    print(f"{name}: finite/row min {np.isfinite(sc).sum(1).min()} median "
          f"{np.median(np.isfinite(sc).sum(1)):.0f}; finite score on id -1: {leak.sum()}; "
          f"-inf on a real id: {hole.sum()}")
both = np.isfinite(ts) & np.isfinite(os_)
diff = np.where(both, np.abs(ts - os_), 0)
print("score_max_abs_diff", diff.max(), "floor (99.9 pct)", np.quantile(diff[both], 0.999))
rows = np.where(diff.max(1) > THRESH)[0]
print(f"rows with a diff > {THRESH}: {len(rows)} of {len(ti)}")
for r in rows[:12]:
    ft, fo = np.isfinite(ts[r]).sum(), np.isfinite(os_[r]).sum()
    st, so = set(ti[r][np.isfinite(ts[r])]), set(oi[r][np.isfinite(os_[r])])
    first = np.argmax(diff[r] > THRESH)
    only_o, only_t = so - st, st - so
    # where official's extra ids would rank on triton's side
    extra_rank = [int(np.where(oi[r] == x)[0][0]) for x in list(only_o)[:5]]
    print(f" row {r}: finite t/o {ft}/{fo}; first rank > thresh {first} "
          f"(t {ts[r, first]:.4f} o {os_[r, first]:.4f}); ids only in official {len(only_o)} "
          f"(ranks {extra_rank}), only in triton {len(only_t)}")
