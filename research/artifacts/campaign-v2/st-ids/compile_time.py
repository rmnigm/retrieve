"""ST-IDS: cold compile time of one `probe_ids_kernel` variant, CPU only (an explicit sm_80 target, no
GPU, no launch), in a fresh `TRITON_CACHE_DIR`. One process per variant, so variants can run side by side
on cores the timed jobs do not use. `--pkg retrieve_v22` is the campaign-v2.2 kernel (make_pkg.sh), whose
tile is [next_pow2(k), next_pow2(n_probe)]; `--pkg retrieve` is this tree's, looped over k and probes.

    TRITON_CACHE_DIR=$(mktemp -d) taskset -c 96 python compile_time.py --pkg retrieve --k 1000 --n-probe 1024
"""

import argparse
import importlib
import json
import pathlib
import re
import subprocess
import tempfile
import time

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--n-probe", type=int, required=True)
args = ap.parse_args()

common = importlib.import_module(f"{args.pkg}.ops.triton.common")
host = importlib.import_module(f"{args.pkg}.ops.triton._host")
ptrs = {"slots_ptr": "*i64", "scores_ptr": "*fp32", "probe_ids_ptr": "*i64", "offsets_ptr": "*i64",
        "sort_perm_ptr": "*i64", "ids_ptr": "*i64", "n_probe": "i32", "k": "i32"}  # fmt: skip
if hasattr(host, "IDS_BLOCK_K"):
    cx = {"BLOCK_K": min(triton.next_power_of_2(args.k), host.IDS_BLOCK_K),
          "BLOCK_N": min(triton.next_power_of_2(args.n_probe), host.IDS_BLOCK_N)}  # fmt: skip
else:
    cx = {
        "NPP": triton.next_power_of_2(args.n_probe),
        "KP": triton.next_power_of_2(args.k),
    }
signature = {**ptrs, **dict.fromkeys(cx, "constexpr")}
src = ASTSource(common.probe_ids_kernel, signature, constexprs=cx)
t0 = time.perf_counter()
kernel = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
dt = time.perf_counter() - t0
cuobjdump = pathlib.Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(kernel.asm["cubin"])
    f.flush()
    res = subprocess.run(
        [cuobjdump, "-res-usage", f.name], capture_output=True, text=True
    ).stdout
reg, stack = map(int, re.search(r"REG:(\d+) STACK:(\d+)", res).groups())
row = {"pkg": args.pkg, "k": args.k, "n_probe": args.n_probe, **cx, "compile_s": round(dt, 2),
       "regs": reg, "stack_bytes": stack, "cubin_bytes": len(kernel.asm["cubin"])}  # fmt: skip
print(json.dumps(row), flush=True)
