"""WP-4 step 7: cold cute.compile time per specialization and first-call latency, fresh process."""

from __future__ import annotations

import importlib
import time

import torch

t0 = time.perf_counter()
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
print(f"import host module: {time.perf_counter() - t0:.3f}s")
t0 = time.perf_counter()
ct.ensure_built(verbose=True)  # DSL init + first compile (score <8,false,1>)
print(f"ensure_built total (DSL import + first compile): {time.perf_counter() - t0:.2f}s")
dev = ct._load_dev()
specs = [
    ("score <8,true,1>", lambda: dev.compile_score(8, True, 1, device=0)),
    ("score <8,false,4>", lambda: dev.compile_score(8, False, 4, device=0)),
    ("score <8,true,4>", lambda: dev.compile_score(8, True, 4, device=0)),
    ("score <4,false,1>", lambda: dev.compile_score(4, False, 1, device=0)),
    ("score <16,false,1>", lambda: dev.compile_score(16, False, 1, device=0)),
    ("bloom mask", lambda: dev.compile_bloom_mask(device=0)),
    ("clause mask <2,2>", lambda: dev.compile_clause_mask(2, 2, device=0)),
    ("clause mask <0,0>", lambda: dev.compile_clause_mask(0, 0, device=0)),
    ("generic <false>", lambda: dev.compile_score_generic(False, device=0)),
    ("generic <true>", lambda: dev.compile_score_generic(True, device=0)),
]
for name, fn in specs:
    t0 = time.perf_counter()
    fn()
    print(f"cold cute.compile {name:<20} {1e3 * (time.perf_counter() - t0):7.1f} ms")
t0 = time.perf_counter()
dev.compile_score(8, True, 1, device=0)
print(f"warm cache lookup: {1e6 * (time.perf_counter() - t0):.1f} us")

# first-call latency per filter mode in this (already-initialised) process: time the first
# _impl call of each mode incl. any compile it triggers, then a second (warm) call.
import sys  # noqa: E402

sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import bench_common as bc  # noqa: E402

for mode in ("none", "bloom", "exact"):
    ct_dev = ct._load_dev()
    n_before = len(ct_dev._compiled)
    p, fns = bc.build("S", 1, mode)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fns["cute"]()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    fns["cute"]()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"{mode:<6}: first call {1e3 * (t1 - t0):7.2f} ms, second {1e3 * (t2 - t1):6.2f} ms; "
          f"specializations touched: {len(ct_dev._compiled) - n_before} new, keys={sorted(k for k in ct_dev._compiled)}")
