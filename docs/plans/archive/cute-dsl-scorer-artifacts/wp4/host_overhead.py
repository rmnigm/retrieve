"""WP-4 step 4b/5: host enqueue overhead per call (GPU kept busy, reviewer's host.py
method) for the three backends' _impls at B=16 layout A, all modes, plus the phase-3
launcher alone (cute vs cuda) and its breakdown."""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import torch  # noqa: E402
from bench_common import build, check_equal, cu, ct  # noqa: E402
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global  # noqa: E402

DEV = "cuda"
cu.ensure_built()
ct.ensure_built()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60  # ~10 launches per _impl call: keep well under the launch-queue depth
PHASE3_ONLY = len(sys.argv) > 3 and sys.argv[3] == "phase3"
big = torch.randn(8192, 8192, device=DEV)


def enqueue_us(fn, n=N, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
        big @ big  # ~60 ms each: the queue never drains during the timed loop
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return 1e6 * (t1 - t0) / n


out = {}
# NOTE: the full _impls cannot be measured enqueue-only: something in the shared prep /
# topk tail blocks the host (~GPU time per call), for all three backends alike. The
# per-launcher costs below are the host-side numbers; the wall-vs-GPU split at the
# head-to-head regimes comes from h2h (do_bench) minus kernel_only (profiler).
print("--- phase-2 mask launchers (eager _impl), B=16, n_probe=32, max_size=1824 (layout A), enqueue-only us/call")
import bench_common as bc  # noqa: E402
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds  # noqa: E402

torch.manual_seed(0)
n_lists, max_size, n_probe = bc.LAYOUTS["A"]
n = n_lists * max_size
padded = torch.randperm(n, device=DEV).reshape(n_lists, max_size)
probe_ids = torch.randint(0, n_lists, (16, n_probe), device=DEV)
flat_a = padded[probe_ids].reshape(16, -1)
attrs = torch.randint(0, 50, (n, 2, 2), device=DEV)
q_attrs = torch.randint(0, 50, (16, 2), device=DEV)
seeds = generate_seeds(5, device=DEV)
sigs = build_signatures(attrs, seeds, 1024, 5, 16)
qb = build_signatures(q_attrs.unsqueeze(-1), seeds, 1024, 5, 16)
sigs_t = cu.build_transposed_sigs(sigs, padded)
rev = torch.zeros(2, dtype=torch.bool, device=DEV); rev[0] = True
qc = torch.randint(0, 50, (16, 2), device=DEV); qc[:, -1] = -1
mask_parts = {
    "cuda _bloom_partial_mask_cuda_impl": lambda: cu._bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size),
    "cute _bloom_partial_mask_cute_impl": lambda: ct._bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size),
    "cuda _clause_partial_mask_cuda_impl": lambda: cu._clause_partial_mask_cuda_impl(flat_a, attrs, rev, qc, max_size),
    "cute _clause_partial_mask_cute_impl": lambda: ct._clause_partial_mask_cute_impl(flat_a, attrs, rev, qc, max_size),
}
assert torch.equal(mask_parts["cuda _bloom_partial_mask_cuda_impl"](), mask_parts["cute _bloom_partial_mask_cute_impl"]())
assert torch.equal(mask_parts["cuda _clause_partial_mask_cuda_impl"](), mask_parts["cute _clause_partial_mask_cute_impl"]())
out["mask"] = {}
for name, fn in mask_parts.items():
    us = enqueue_us(fn, n=300)
    out["mask"][name] = us
    print(f"{name:55s} {us:7.1f} us")

print("--- phase-3 launcher only (cps_scores / _cps_cute_scores), B=16 P=8192")
b, p, n, d = 16, 8192, 131072, 128
codes, gs = quantize_int8_global(torch.randn(n, d, device=DEV))
q_codes, q_scales = quantize_int8(torch.randn(b, d, device=DEV))
q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
flat = torch.randint(0, n, (b, p), device=DEV)
o = torch.empty(b, p, dtype=torch.float32, device=DEV)
dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
cfg_cu, cfg_ct = cu.DEFAULT_CONFIG, ct.DEFAULT_CONFIG
ext = cu._load_ext()
dev = ct._load_dev()
run_cuda = lambda: ext.cps_scores(q_codes, q_scales, dummy, flat, codes, o, float(gs), False, 0,
                                  cfg_cu.block_p, cfg_cu.num_warps, cfg_cu.unroll)
run_cute = lambda: ct._cps_cute_scores(q_codes, q_scales, dummy, flat, codes, o, global_scale=float(gs),
                                       has_mask=False, max_size=0, cfg=cfg_ct)
launch = dev.compile_score(8, False, 1, device=0)  # _Launch (pre-packed cells)
addrs = [q_codes.data_ptr(), q_scales.data_ptr(), dummy.data_ptr(), flat.data_ptr(), codes.data_ptr(), o.data_ptr()]
ptrs = [
    dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
    dev.gmem_ptr(dev.Int64, dummy.data_ptr(), 8),
    dev.gmem_ptr(dev.Int64, flat.data_ptr(), 8),
    dev.gmem_ptr(dev.Int32, codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, o.data_ptr(), 4),
]
stream = dev.cu_stream(torch.cuda.current_stream().cuda_stream)
scalars = (float(gs), p, 0, 1, 1, b, cfg_ct.block_p, cfg_ct.num_warps)
handle = torch.cuda.current_stream().cuda_stream


def with_device():
    with torch.cuda.device(q_codes.device):
        pass


parts = {
    "cuda cps_scores (C++ launcher)": run_cuda,
    "cute _cps_cute_scores (full)": run_cute,
    "cute _Launch (pre-packed cells), args prebuilt": lambda: launch(*addrs, *scalars, handle),
    "cute DSL JitExecutor.__call__, args prebuilt": lambda: launch._executor(*ptrs, *scalars, stream),
    "6x gmem_ptr": lambda: [dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), 16), dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
                            dev.gmem_ptr(dev.Int64, dummy.data_ptr(), 8), dev.gmem_ptr(dev.Int64, flat.data_ptr(), 8),
                            dev.gmem_ptr(dev.Int32, codes.data_ptr(), 16), dev.gmem_ptr(dev.Float32, o.data_ptr(), 4)],
    "with torch.cuda.device(...)": with_device,
    "torch.cuda.current_stream().cuda_stream + cu_stream": lambda: dev.cu_stream(torch.cuda.current_stream().cuda_stream),
    "torch._C._cuda_getCurrentRawStream(0) + cu_stream": lambda: dev.cu_stream(torch._C._cuda_getCurrentRawStream(0)),
    "torch.cuda.current_device()": torch.cuda.current_device,
    "compile_score cache lookup": lambda: dev.compile_score(8, False, 1, device=0),
    "torch baseline o.fill_(0)": lambda: o.fill_(0.0),
}
out["phase3"] = {}
for name, fn in parts.items():
    us = enqueue_us(fn, n=500)
    out["phase3"][name] = us
    print(f"{name:55s} {us:7.1f} us")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
