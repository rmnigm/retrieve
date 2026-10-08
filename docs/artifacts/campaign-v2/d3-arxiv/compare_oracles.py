"""Compare each oracle blob in NEW (rebuilt at v2.1) with the same-named blob in OLD (408b1188):
torch.equal on every tensor, and the non-tensor fields that differ. Usage: compare_oracles.py NEW OLD"""

import sys
from pathlib import Path

import torch

new, old = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for o in sorted(old.glob("oracle_v4_*.pt")):
    n = new / o.name
    if not n.exists():
        print(f"{o.name}: not rebuilt (not read by these suites)")
        continue
    a, b = torch.load(o, weights_only=False), torch.load(n, weights_only=False)
    tens = [k for k, v in a.items() if torch.is_tensor(v)]
    eq = {k: torch.equal(a[k], b[k]) for k in tens}
    meta = {
        k: (a[k], b.get(k)) for k in a if not torch.is_tensor(a[k]) and a[k] != b.get(k)
    }
    bad += not all(eq.values())
    print(
        f"{o.name}: tensors {'EQUAL' if all(eq.values()) else 'DIFFER ' + str(eq)} ({', '.join(tens)}); meta changed {sorted(meta)}"
    )
print("ALL EQUAL" if not bad else f"{bad} blob(s) DIFFER")
sys.exit(1 if bad else 0)
