"""One-off SASS sanity for the CuTe DSL scorer (CUTE_DSL_KEEP disables the JIT cache, so this
is never part of the normal path). Dumps the D=128 / HAS_MASK / UNROLL=1 scorer (plus the
UNROLL=4 and no-mask variants for the register table), counts the instructions the plan
asks about, and reads the register count off the loaded cubin via the driver API.

Run: cd /workspace/retrieve/retrieve && uv run python <this file>
"""

import collections
import importlib
import os
import re
import sys

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump")
os.makedirs(DUMP, exist_ok=True)
os.environ["CUTE_DSL_KEEP"] = "ptx,sass,cubin"
os.environ["CUTE_DSL_DUMP_DIR"] = DUMP

import torch  # noqa: E402

torch.zeros(1, device="cuda")  # primary context for the driver-API register query below
from cuda.bindings import driver as cuda  # noqa: E402

dev = importlib.import_module("retrieve.kernels.silvertorch.cute.codesigned_probe_score")

WANT = ("IDP.4A", "LDG.E.EF.128", "LDG.E.128", "LDG.E.64", "LDG.E ", "LDG.E.CONSTANT", "REDUX",
        "FMUL", "FFMA", "I2F", "STG.E", "BAR.SYNC", "LDL", "STL", "BRA", "EXIT", "SHFL", "BSSY",
        "REDUX.OR", "VOTE")


def kernel_body(sass: str) -> str:
    """The SASS of the device kernel only (the dump also holds the host launcher stub)."""
    m = re.search(r"\.text\.(\S*cps_score_kernel\S*):(.*?)(?=\n\s+\.text\.|\Z)", sass, re.S)
    return m.group(2) if m else sass


def count_ops(sass: str) -> dict:
    c = collections.Counter()
    for line in sass.splitlines():
        code = line.split(";")[0]
        for w in WANT:
            if w in code:
                c[w.strip()] += 1
    return dict(sorted(c.items()))


def num_regs(cubin: bytes, entry: str) -> int:
    err, mod = cuda.cuModuleLoadData(cubin)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    err, fn = cuda.cuModuleGetFunction(mod, entry.encode())
    assert err == cuda.CUresult.CUDA_SUCCESS, (err, entry)
    err, n = cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    err, spill = cuda.cuFuncGetAttribute(
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn
    )
    return n, spill


def report(name, compiled):
    sass = compiled.__sass__ or ""
    ptx = compiled.__ptx__ or ""
    cubin = compiled.__cubin__
    if isinstance(cubin, str):
        cubin = open(cubin, "rb").read()
    entries = re.findall(r"\.entry\s+(\S+)\(", ptx)
    kentries = [e for e in entries if "kernel" in e]
    body = kernel_body(sass)
    print(f"=== {name}: ptx entries {entries}")
    print("   ops (kernel body):", count_ops(body))
    for e in kentries:
        n, spill = num_regs(cubin, e)
        print(f"   {e}: {n} registers/thread, {spill} B local (spill) memory")
    # per-item instruction ordering check: the dp4a chain and the reduction
    seq = [l.split(";")[0].strip() for l in body.splitlines() if re.search(r"IDP|REDUX|FMUL|I2F|LDG", l)]
    print("   sequence:", " -> ".join(s.split()[0] if not s.startswith("/*") else s.split()[1] for s in seq)[:600])
    with open(os.path.join(DUMP, f"{name}.sass"), "w") as f:
        f.write(sass)
    with open(os.path.join(DUMP, f"{name}.ptx"), "w") as f:
        f.write(ptx)
    return count_ops(body)


ops = report("score_seg8_mask_u1", dev.compile_score(8, True, 1))
report("score_seg8_mask_u4", dev.compile_score(8, True, 4))
report("score_seg8_nomask_u1", dev.compile_score(8, False, 1))
report("score_generic_mask", dev.compile_score_generic(True))
report("bloom_mask", dev.compile_bloom_mask())
report("clause_mask_2_2", dev.compile_clause_mask(2, 2))

print("\nchecks on score<8, true, 1>:")
checks = {
    "IDP.4A x4": ops.get("IDP.4A", 0) == 4,
    "LDG.E.EF.128 row load": ops.get("LDG.E.EF.128", 0) >= 1,
    "REDUX.SUM": ops.get("REDUX", 0) >= 1,
    "2 FMUL": ops.get("FMUL", 0) == 2,
    "no FFMA": ops.get("FFMA", 0) == 0,
    "no BAR.SYNC": ops.get("BAR.SYNC", 0) == 0,
    "no LDL/STL": ops.get("LDL", 0) == 0 and ops.get("STL", 0) == 0,
}
for k, v in checks.items():
    print(f"   {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
