"""V2-FILL: SASS of fill_variants' bodies at one width (one launch each at p 0.1, bs 16), written to <out>/<variant>.sass,
with the count of each instruction class between the first and last branch target (the loops).

    PYTHONPATH=<pkgs>:retrieve/src:. python sass_variants.py N D OUT_DIR --variants before,Sq,Sq32
"""

import collections
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
variants = sys.argv[sys.argv.index("--variants") + 1].split(",")
sys.argv = [
    sys.argv[0],
    sys.argv[1],
    sys.argv[2],
    "/dev/null",
    "--rates",
    "",
    "--variants",
    "A",
]
import fill_variants as fv  # noqa: E402  (its sweep is empty with --rates "")
import torch  # noqa: E402

n, d = int(sys.argv[1]), int(sys.argv[2])
g = torch.Generator(device=fv.DEV).manual_seed(0)
items = torch.randn(n, d, device=fv.DEV, generator=g).to(torch.float16)
q = torch.randn(16, d, device=fv.DEV, generator=g).to(torch.float16)
cand, counts = fv.candidates(n, 16, 0.1, seed=1)
for v in variants:
    kern, la = fv.launch(v, q, items, cand, counts)
    ck = kern[la.grid](**la.kwargs)
    sass = ck.asm["sass"]
    (out / f"{v}.sass").write_text(sass)
    ops = collections.Counter(
        re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)", sass)
    )
    top = ", ".join(f"{k} {c}" for k, c in ops.most_common(14))
    print(
        f"{v}: regs {ck.n_regs} spills {ck.n_spills} total {sum(ops.values())} | {top}",
        flush=True,
    )
