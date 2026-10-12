"""META-FORK C0: Meta's pinned build and the fork in one process. Both op namespaces register, each
``fused_kmean_ann`` runs on the same random problem (int32 and fp16 paths, a full-N mask and none), and the
two outputs are ``torch.equal`` (the fork's upstream-shaped op against upstream's). Prints one JSON line.

    PYTHONPATH=<build tree>/retrieve/src python c0_coexist.py
"""

import json

import torch

from retrieve.ops import official as of

st = of.ensure_loaded("official")
stfork = of.ensure_loaded("official-fork")
g = torch.Generator(device="cuda").manual_seed(0)
sizes = torch.randint(0, 300, (128,), generator=g, device="cuda")
offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
n, b, p, d = int(offsets[-1]), 16, 24, 128
codes = torch.randint(-127, 128, (n, d), dtype=torch.int8, generator=g, device="cuda")
q = torch.randint(-127, 128, (b, d), dtype=torch.int8, generator=g, device="cuda")
probe = torch.rand(b, 128, generator=g, device="cuda").argsort(1)[:, :p]
width = int(sizes.sort(descending=True).values[:p].sum())
mask = of.pack_mask(torch.rand(b, n, generator=g, device="cuda") < 0.3)
out = {
    "namespaces": [hasattr(torch.ops, "st"), hasattr(torch.ops, "stfork")],
    "equal": {},
}
for divisor in (-1, 32):
    for m in (None, mask):
        args = (q, probe, offsets, sizes, codes, width)
        up = of.fused_scores(
            *args, divisor=divisor, filtering_bit_mask=m, backend="official"
        )
        fk = of.fused_scores(
            *args, divisor=divisor, filtering_bit_mask=m, backend="official-fork"
        )
        key = f"divisor={divisor},mask={m is not None}"
        out["equal"][key] = torch.equal(up[0], fk[0]) and torch.equal(up[1], fk[1])
print(json.dumps(out))
