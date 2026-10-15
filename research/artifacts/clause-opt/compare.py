"""The bit-exact gate: two clause_bench.py dumps (final-freeze vs CLAUSE-OPT), torch.equal on
scores and ids per cell and batch.  python compare.py base.pt new.pt"""

import sys

import torch

base, new = (torch.load(p, weights_only=True) for p in sys.argv[1:3])
assert base.keys() == new.keys(), sorted(set(base) ^ set(new))
bad = 0
for key in base:
    for i, ((bi, bs), (ni, ns)) in enumerate(zip(base[key], new[key], strict=True)):
        if not (torch.equal(bs, ns) and torch.equal(bi, ni)):
            bad += 1
            print(
                f"MISMATCH {key} batch {i}: scores equal {torch.equal(bs, ns)}, "
                f"ids differ at {int((bi != ni).sum())} of {bi.numel()}"
            )
cells = len(base)
batches = sum(len(v) for v in base.values())
print(
    f"{cells} cells, {batches} batches, {bad} mismatching: {'PASS' if bad == 0 else 'FAIL'}"
)
sys.exit(bad != 0)
