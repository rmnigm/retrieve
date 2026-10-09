"""ST-WIDE-2: host time of one eager `codesigned_probe_score_exact` call (GPU work small, no sync inside the timed
loop) with the per-row table on and off, on a v2.9 tree, where the table is its own launch (`probe_table_kernel`) and
the query is quantized by `quantize_int8` (torch ops); plus the table launch alone and `torch.empty` for scale.

    PYTHONPATH=<v2.9 tree>/retrieve/src:<v2.9 tree>/retrieve python host_cost.py OUT.json
"""

import importlib
import json
import sys
import time

import torch

from tests.conftest import make_attrs, make_query_attrs
from tests.parity.conftest import make_probe_family

H = importlib.import_module("retrieve.ops.triton._host")
from retrieve.ops.triton.common import probe_table_kernel  # noqa: E402

b, d, n_probe = 16, 128, 128
lay = make_probe_family(b, 1024, 60, n_probe)
q = torch.randn(b, d, device="cuda")
codes = torch.randint(-127, 128, (lay.n, d), dtype=torch.int8, device="cuda")
attrs = make_attrs(lay.n, c=2, a_max=2).contiguous()
rev = torch.tensor([False, True], device="cuda")
qa = make_query_attrs(b, c=2)
args = (
    q,
    lay.probe_ids,
    lay.cluster_offsets,
    codes,
    lay.sort_perm,
    attrs,
    rev,
    qa,
    0.01,
    100,
    lay.width,
)
op = torch.ops.retrieve.codesigned_probe_score_exact


def host_us(fn, n=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    el = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    return el


out = {}
for label, pairs in (("table", 0), ("no_table", 1 << 60)):
    H.TABLE_MIN_PAIRS = pairs
    out[f"op_{label}_us"] = host_us(lambda: op(*args))
H.TABLE_MIN_PAIRS = 0
la = H.probe_prep(q, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, 0.01, lay.width, block_p=256,
                  num_warps=4, num_stages=3, block_d=256, skip=False)  # fmt: skip
out["table_launch_us"] = host_us(
    lambda: probe_table_kernel[la.table.grid](**la.table.kwargs)
)
out["torch_empty_us"] = host_us(
    lambda: torch.empty((b, 3, n_probe), dtype=torch.int64, device="cuda")
)
print(json.dumps(out))
open(sys.argv[1], "w").write(json.dumps(out, indent=1))
