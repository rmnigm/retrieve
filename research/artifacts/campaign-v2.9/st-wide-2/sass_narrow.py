"""ST-WIDE: the probe scorers below the table threshold (n_probe 6, bs {1, 16}) at D {128, 192, 768}, none / bloom /
exact, compiled into a fresh TRITON_CACHE_DIR from one tree; per cubin its register count and two SASS hashes: the
instructions, and the opcodes alone (constant-bank offsets move with an added kernel argument). Run once per tree.

    TRITON_CACHE_DIR=$(mktemp -d) python sass_narrow.py --tree <tree>
"""

import hashlib
import json
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
from retrieve.indexing.selectivity import bloom_bit_freq  # noqa: E402
from tests.parity.conftest import (  # noqa: E402
    make_attrs,
    make_bloom,
    make_probe_family,
    make_query_attrs,
)

for d in (128, 192, 768):
    for b in (1, 16):
        lay = make_probe_family(b, 64, 300, 6)
        g = torch.Generator(device="cuda").manual_seed(0)
        query = torch.randn(b, d, device="cuda", generator=g)
        codes = torch.randint(
            -127, 128, (lay.n, d), dtype=torch.int8, device="cuda", generator=g
        )
        base = (query, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm)
        T.codesigned_probe_score(*base, 0.01, 32, lay.width)
        qpos, bt, _, _ = make_bloom(lay.n, b)
        T.codesigned_probe_score_bloom(
            *base, qpos, bt, bloom_bit_freq(bt, lay.n), 0.01, 32, lay.width
        )
        attrs = make_attrs(lay.n, c=2, a_max=2, seed=1).contiguous()
        rev = torch.tensor([False, True], device="cuda")
        T.codesigned_probe_score_exact(
            *base, attrs, rev, make_query_attrs(b, c=2, seed=2), 0.01, 32, lay.width
        )
torch.cuda.synchronize()
cuobjdump = pathlib.Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
rows = []
for p in pathlib.Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.cubin"):
    meta = json.loads(p.with_suffix(".json").read_text())
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
    ops = [re.sub(r"^@!?U?P\w+\s+", "", ln).split()[0] for ln in body]
    h = lambda xs: hashlib.sha256("\n".join(xs).encode()).hexdigest()[:16]  # noqa: E731
    key = {k: meta.get(k) for k in ("name", "num_warps", "num_stages")} | {
        k: v
        for k, v in (meta.get("constants") or {}).items()
        if k in ("D", "BLOCK_P", "HAS_QB", "SKIP", "GATED")
    }
    rows.append(
        f"{json.dumps(key, sort_keys=True)} sass={h(body)} ops={h(ops)} n={len(body)}"
    )
print("\n".join(sorted(rows)))
print(len(rows), "cubins", file=sys.stderr)
