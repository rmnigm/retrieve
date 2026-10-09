"""CLAUSE-SKIP: call every `retrieve::` Triton op once (the op-registry test's input builders,
tests/compile/test_export_kernel_ref.py::_op_args, of the same tree) into a fresh TRITON_CACHE_DIR and print one sha256
per cubin's SASS (pubmed-fixes/sass_identity.py's method). Run once per tree; only the clause kernels may differ.

    TRITON_CACHE_DIR=$(mktemp -d) python sass_ops.py --tree <tree>
"""

import hashlib
import os
import pathlib
import re
import subprocess
import sys

tree = sys.argv[sys.argv.index("--tree") + 1]
sys.path[:0] = [f"{tree}/retrieve/src", f"{tree}/retrieve"]

import torch  # noqa: E402
import triton  # noqa: E402

import retrieve.ops.triton as T  # noqa: E402
from tests.compile.test_export_kernel_ref import _op_args  # noqa: E402

print("retrieve from", T.__file__, file=sys.stderr)
for name, args in _op_args().items():
    getattr(T, name)(*args)
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
