"""Fix 1: compile the three probe-scorer ops at D = 128 and 192 (the widths whose tile did not
change) into a fresh TRITON_CACHE_DIR and print one sha256 per cubin's SASS (`cuobjdump -sass`,
addresses and comments stripped; l4-pow2-pad/ptx_identity.py's method). Run once per tree with
`--src`; identical output means the same machine code.

    TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=retrieve python sass_identity.py [--src SRC]
"""

import hashlib
import os
import pathlib
import re
import subprocess
import sys

if "--src" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--src") + 1])

import torch
import triton

import retrieve
from retrieve.ops import triton as T

print("retrieve", retrieve.__file__, file=sys.stderr)
N, B = 512, 2


def _i64(*shape):
    return torch.randint(0, 8, shape, device="cuda")


probe = torch.tensor([[0, 1]] * B, device="cuda")
offs = torch.tensor([0, N // 2, N], device="cuda")
perm = torch.arange(N, device="cuda")
for d in (128, 192):
    qf, codes = torch.randn(B, d, device="cuda"), torch.zeros(N, d, dtype=torch.int8, device="cuda")
    rev, qa, attrs = torch.zeros(2, dtype=torch.bool, device="cuda"), _i64(B, 2), _i64(N, 2, 2)
    T.codesigned_probe_score(qf, probe, offs, codes, perm, 0.1, 4, N)
    T.codesigned_probe_score_bloom(
        qf, probe, offs, codes, perm, _i64(B, 10), _i64(512, N // 64), 0.1, 4, N
    )
    T.codesigned_probe_score_exact(qf, probe, offs, codes, perm, attrs, rev, qa, 0.1, 4, N)
torch.cuda.synchronize()
cuobjdump = pathlib.Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
rows = []
for p in pathlib.Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.cubin"):
    sass = subprocess.run([cuobjdump, "-sass", p], capture_output=True, text=True).stdout
    body = [re.sub(r"/\*[0-9a-f]{4}\*/", "", ln).split(";")[0].strip() for ln in sass.splitlines()]
    body = [ln for ln in body if ln and not ln.startswith(("code for", "Function", ".", "//"))]
    rows.append(f"{p.stem} {hashlib.sha256(chr(10).join(body).encode()).hexdigest()[:16]}")
print("\n".join(sorted(rows)))
print(len(rows), "cubins", file=sys.stderr)
