"""WP-4 step 7: re-dump SASS + registers for the scorer builds after the step-6 change
(WP-1's sass_dump.py method; `_compiled[key]` now holds the JitCompiledFunction)."""

import collections
import importlib
import os
import re
import sys

W = "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad"
DUMP = f"{W}/wp1/dump"  # the existing dir
os.environ["CUTE_DSL_KEEP"] = "ptx,sass,cubin"
os.environ["CUTE_DSL_DUMP_DIR"] = DUMP
import torch  # noqa: E402

torch.zeros(1, device="cuda")
from cuda.bindings import driver as cuda  # noqa: E402

dev = importlib.import_module("retrieve.kernels.silvertorch.cute.codesigned_probe_score")
WANT = ("IDP.4A", "LDG.E.EF.128", "LDG.E.128", "LDG.E.64", "LDG.E ", "LDG.E.CONSTANT", "REDUX", "FMUL", "FFMA", "I2F",
        "STG.E", "BAR.SYNC", "LDL", "STL", "BRA", "EXIT", "SHFL", "BSSY", "REDUX.OR", "VOTE")


def kernel_body(sass):
    m = re.search(r"\.text\.(\S*cps_score_kernel\S*):(.*?)(?=\n\s+\.text\.|\Z)", sass, re.S)
    return m.group(2) if m else sass


def count_ops(sass):
    c = collections.Counter()
    for line in sass.splitlines():
        code = line.split(";")[0]
        for w in WANT:
            if w in code:
                c[w.strip()] += 1
    return dict(sorted(c.items()))


def num_regs(cubin, entry):
    err, mod = cuda.cuModuleLoadData(cubin); assert err == cuda.CUresult.CUDA_SUCCESS, err
    err, fn = cuda.cuModuleGetFunction(mod, entry.encode()); assert err == cuda.CUresult.CUDA_SUCCESS, (err, entry)
    err, n = cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
    err, spill = cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)
    return n, spill


def report(name, key):
    compiled = dev._compiled[key]
    sass, ptx, cubin = compiled.__sass__ or "", compiled.__ptx__ or "", compiled.__cubin__
    if isinstance(cubin, str):
        cubin = open(cubin, "rb").read()
    entries = [e for e in re.findall(r"\.entry\s+(\S+)\(", ptx) if "kernel" in e]
    body = kernel_body(sass)
    ops = count_ops(body)
    regs = [num_regs(cubin, e) for e in entries]
    print(f"=== {name}: ops {ops}; regs/spill {regs}")
    open(f"{DUMP}/{name}.sass", "w").write(sass)
    open(f"{DUMP}/{name}.ptx", "w").write(ptx)
    return ops, regs


rows = {}
for name, args in (("score_seg8_nomask_u1", (8, False, 1)), ("score_seg8_nomask_u4", (8, False, 4)),
                   ("score_seg8_mask_u1", (8, True, 1)), ("score_seg8_mask_u4", (8, True, 4))):
    dev.compile_score(*args, device=0)
    rows[name] = report(name, ("score",) + args)
ops, regs = rows["score_seg8_nomask_u1"]
checks = {"IDP.4A x4": ops.get("IDP.4A", 0) == 4, "LDG.E.EF.128 x1": ops.get("LDG.E.EF.128", 0) == 1,
          "REDUX.SUM present": ops.get("REDUX", 0) >= 1, "2 FMUL": ops.get("FMUL", 0) == 2, "no FFMA": ops.get("FFMA", 0) == 0,
          "no BAR.SYNC": ops.get("BAR.SYNC", 0) == 0, "no LDL/STL": ops.get("LDL", 0) == 0 and ops.get("STL", 0) == 0}
for k, v in checks.items():
    print(f"   {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
