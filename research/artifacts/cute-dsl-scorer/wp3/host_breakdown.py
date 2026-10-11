"""WP-3: where the per-call host time of _cps_cute_scores goes."""

from __future__ import annotations

import importlib
import sys
import time

sys.path.insert(0, "/workspace/retrieve/retrieve")
import torch  # noqa: E402

ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global  # noqa: E402

DEV = "cuda"
ct.ensure_built()
dev = ct._load_dev()

b, p, n, d = 16, 8192, 131072, 128
codes, gs = quantize_int8_global(torch.randn(n, d, device=DEV))
q_codes, q_scales = quantize_int8(torch.randn(b, d, device=DEV))
q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
flat = torch.randint(0, n, (b, p), device=DEV)
out = torch.empty(b, p, dtype=torch.float32, device=DEV)
dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
cfg = ct.DEFAULT_CONFIG
N = 500


def timeit(name, fn, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    big = torch.randn(8192, 8192, device=DEV)
    for _ in range(3):
        big @ big  # keep the GPU busy so enqueue never blocks
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"{name:55s} {1e6 * (t1 - t0) / N:7.1f} us")


launch = dev.compile_score(8, False, 1)
ptrs = [
    dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
    dev.gmem_ptr(dev.Int64, dummy.data_ptr(), 8),
    dev.gmem_ptr(dev.Int64, flat.data_ptr(), 8),
    dev.gmem_ptr(dev.Int32, codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, out.data_ptr(), 4),
]
stream = dev.cu_stream(torch.cuda.current_stream().cuda_stream)
scalars = (float(gs), p, 0, 1, 1, b, cfg.block_p, cfg.num_warps)

timeit("full _cps_cute_scores", lambda: ct._cps_cute_scores(
    q_codes, q_scales, dummy, flat, codes, out, global_scale=float(gs), has_mask=False,
    max_size=0, cfg=cfg))
timeit("compiled callable, all args prebuilt", lambda: launch(*ptrs, *scalars, stream))
timeit("6x gmem_ptr only", lambda: [
    dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
    dev.gmem_ptr(dev.Int64, dummy.data_ptr(), 8),
    dev.gmem_ptr(dev.Int64, flat.data_ptr(), 8),
    dev.gmem_ptr(dev.Int32, codes.data_ptr(), 16),
    dev.gmem_ptr(dev.Float32, out.data_ptr(), 4),
])


def with_device():
    with torch.cuda.device(q_codes.device):
        pass


timeit("with torch.cuda.device(...) only", with_device)
timeit("torch.cuda.current_stream().cuda_stream + cu_stream", lambda: dev.cu_stream(torch.cuda.current_stream().cuda_stream))
timeit("torch baseline out.fill_(0)", lambda: out.fill_(0.0))

# what does the DSL do per call? profile generate_execution_args
import cProfile  # noqa: E402
import pstats  # noqa: E402

pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    launch(*ptrs, *scalars, stream)
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(18)
