"""C5-OURS: run the SilverTorch triton forward (none / bloom default path / exact) at D = 128 and 768 into a fresh
TRITON_CACHE_DIR and print one sha256 per cubin's SASS (pubmed-fixes/sass_identity.py's method). Run once per tree with
`--src`; identical output means the default path launches the same machine code.

    TRITON_CACHE_DIR=$(mktemp -d) python sass_layer.py --src <tree>/retrieve/src
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

from retrieve.modules.silvertorch import SilverTorch  # noqa: E402

print("retrieve from", SilverTorch.__module__, file=sys.stderr)
N = 20_000
g = torch.Generator(device="cuda").manual_seed(0)
for d in (128, 768):
    embs = torch.randn(N, d, device="cuda", generator=g)
    attrs = torch.randint(0, 5, (N, 2, 1), device="cuda", generator=g)
    for mode in ("none", "bloom", "exact"):
        for bs in (1, 16):
            kw = {"m_bits": 1024, "k_hash": 5} if mode == "bloom" else {}
            m = SilverTorch(
                k=100, n_lists=64, n_probe=8, filter_mode=mode, n_iter=2, **kw
            )
            m.register_index(embs, None if mode == "none" else attrs)
            q = torch.randn(bs, d, device="cuda", generator=g)
            m(
                q,
                None
                if mode == "none"
                else torch.randint(0, 5, (bs, 2), device="cuda", generator=g),
            )
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
