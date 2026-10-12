"""V2-FILL: compile `fused_masked_knn_topk` at D = 768 and 1024 (SPLIT, the body that must not change), through
the public op and the bucketed `_impl`, into a fresh TRITON_CACHE_DIR, and print one sha256 per cubin's SASS
(addresses and comments stripped; pubmed-fixes/sass_identity.py's method). Run once per tree with `--src`;
identical output means the same machine code.

    TRITON_CACHE_DIR=$(mktemp -d) python sass_fmkt.py --src <tree>/retrieve/src
"""

import hashlib
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, sys.argv[sys.argv.index("--src") + 1])

import torch  # noqa: E402
import triton  # noqa: E402

from retrieve.ops.triton import fused_masked_knn_topk  # noqa: E402
from retrieve.ops.triton.fused_masked_knn_topk import _fused_masked_knn_topk_impl  # noqa: E402

print("retrieve from", fused_masked_knn_topk.__module__, file=sys.stderr)
N, B, P = 4096, 2, 1024
for d in (768, 1024):
    items = torch.randn(N, d, device="cuda").half()
    q = torch.randn(B, d, device="cuda").half()
    cand = torch.randint(0, N, (B, P), device="cuda")
    counts = torch.tensor([P // 2, P], device="cuda")
    fused_masked_knn_topk(q, items, cand, counts, 8)
    _fused_masked_knn_topk_impl(q, items, cand, counts, 8)
torch.cuda.synchronize()
cuobjdump = pathlib.Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
rows = []
for p in pathlib.Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.cubin"):
    sass = subprocess.run(
        [cuobjdump, "-sass", p], capture_output=True, text=True
    ).stdout
    body = [
        re.sub(r"/\*[0-9a-f]{4}\*/", "", ln).split(";")[0].strip()
        for ln in sass.splitlines()
    ]
    body = [
        ln
        for ln in body
        if ln and not ln.startswith(("code for", "Function", ".", "//"))
    ]
    rows.append(
        f"{p.stem} {hashlib.sha256(chr(10).join(body).encode()).hexdigest()[:16]}"
    )
print("\n".join(sorted(rows)))
print(len(rows), "cubins", file=sys.stderr)
