"""ST-SKIP128: compile the three probe-scorer variants through `_impl` without the pass-rate tables (GATED off)
at D = 128, 192 and 768 into a fresh TRITON_CACHE_DIR and print one sha256 per cubin's SASS (pubmed-fixes/
sass_identity.py's method). Run once per tree with `--src`; identical output means the ungated code is unchanged,
so the gate is the only codegen difference.

    TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=retrieve python sass_ungated.py --src <tree>/retrieve/src
"""

import hashlib
import importlib
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, sys.argv[sys.argv.index("--src") + 1])

import torch  # noqa: E402
import triton  # noqa: E402

cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
cpse = importlib.import_module("retrieve.ops.triton.codesigned_probe_score_exact")
print("retrieve from", cps.__file__, file=sys.stderr)
N, B = 512, 2


def _i64(*shape):
    return torch.randint(0, 8, shape, device="cuda")


probe = torch.tensor([[0, 1]] * B, device="cuda")
offs = torch.tensor([0, N // 2, N], device="cuda")
perm = torch.arange(N, device="cuda")
for d in (128, 192, 768):
    qf, codes = (
        torch.randn(B, d, device="cuda"),
        torch.zeros(N, d, dtype=torch.int8, device="cuda"),
    )
    rev, qa, attrs = (
        torch.zeros(2, dtype=torch.bool, device="cuda"),
        _i64(B, 2),
        _i64(N, 2, 2),
    )
    lay = (qf, probe, offs, codes, perm, 0.1, 4, N)
    cps._codesigned_probe_score_impl(*lay)
    cps._codesigned_probe_score_impl(
        *lay, query_bit_positions=_i64(B, 10), bloom_transposed=_i64(512, N // 64)
    )
    cpse._codesigned_probe_score_exact_impl(
        *lay, item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=qa
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
