"""CLAUSE-SKIP: registers / spills / shared memory of the clause kernels in one tree, at goodreads' attr shape
(C, A_MAX from the data, `all4` = every clause active) and the 10-clause synth table (A_MAX 1), bs 1 and 16.

    PYTHONPATH=<tree>/retrieve/src python regs.py <goodreads item_attrs_narrow.pt>
"""

import sys

import importlib

import torch

cc = importlib.import_module("retrieve.ops.triton.clause_compact")
cm = importlib.import_module("retrieve.ops.triton.clause_mask")

g = torch.load(sys.argv[1]).long()
if g.shape[0] > 1 and (g[0] == 0).all():  # legacy padding row
    g = g[1:]
shapes = {
    "goodreads": g[:200_000].cuda(),
    "synth10": torch.randint(0, 2, (200_000, 10, 1), device="cuda"),
}
for name, attrs in shapes.items():
    n, c, a = attrs.shape
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    for bs in (1, 16):
        qa = attrs[:bs, :, 0].clone()  # every clause active
        lc = cc._clause_compact_prep(attrs, rev, qa, cfg=cc.DEFAULT_CONFIG)
        kc = cc._clause_compact_kernel[lc.grid](**lc.kwargs)
        lm = cm._clause_mask_prep(attrs, rev, qa, cfg=cm.DEFAULT_CONFIG)
        km = cm._clause_mask_kernel[lm.grid](**lm.kwargs)
        for kn, k in (("clause_compact", kc), ("clause_mask", km)):
            print(
                f"{name} C={c} A={a} bs={bs} {kn}: regs {k.n_regs} spills {k.n_spills} shared {k.metadata.shared}"
            )
