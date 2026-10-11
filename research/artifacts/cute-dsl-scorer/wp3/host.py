"""WP-3: B=0 launch behaviour, per-call host overhead (cute vs cuda), stream-handle facts."""

from __future__ import annotations

import importlib
import sys
import time

sys.path.insert(0, "/workspace/retrieve/retrieve")
import torch  # noqa: E402

cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global  # noqa: E402

DEV = "cuda"
cu.ensure_built()
ct.ensure_built()

# --- B = 0 ---------------------------------------------------------------------------------
d, n, p = 128, 256, 64
codes, gs = quantize_int8_global(torch.randn(n, d, device=DEV))
for be, fn in (("cuda", lambda *a, **k: cu._load_ext().cps_scores(*a[:6], float(gs), False, 0, 128, 8, 1)),
               ("cute", lambda *a, **k: ct._cps_cute_scores(*a[:6], global_scale=float(gs), has_mask=False, max_size=0, cfg=ct.DEFAULT_CONFIG))):
    q = torch.empty(0, d, dtype=torch.int8, device=DEV)
    qs = torch.empty(0, dtype=torch.float32, device=DEV)
    flat = torch.empty(0, p, dtype=torch.int64, device=DEV)
    out = torch.empty(0, p, dtype=torch.float32, device=DEV)
    dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
    try:
        fn(q, qs, dummy, flat, codes, out)
        torch.cuda.synchronize()
        print(f"B=0 {be}: no error (grid.y = 0)")
    except Exception as e:
        print(f"B=0 {be}: {type(e).__name__}: {str(e)[:100]}")
# context still alive?
torch.ones(1, device=DEV).sum().item()
print("context OK after B=0")

# --- per-call host overhead, phase 3 only, no sync inside the loop --------------------------
b, p, n = 16, 8192, 131072
codes, gs = quantize_int8_global(torch.randn(n, d, device=DEV))
q_codes, q_scales = quantize_int8(torch.randn(b, d, device=DEV))
q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
flat = torch.randint(0, n, (b, p), device=DEV)
out = torch.empty(b, p, dtype=torch.float32, device=DEV)
dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
cfg_cu = cu.DEFAULT_CONFIG
cfg_ct = ct.DEFAULT_CONFIG


def run_cuda():
    cu._load_ext().cps_scores(q_codes, q_scales, dummy, flat, codes, out, float(gs), False, 0,
                              cfg_cu.block_p, cfg_cu.num_warps, cfg_cu.unroll)


def run_cute():
    ct._cps_cute_scores(q_codes, q_scales, dummy, flat, codes, out, global_scale=float(gs),
                        has_mask=False, max_size=0, cfg=cfg_ct)


for name, fn in (("cuda", run_cuda), ("cute", run_cute)):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    # host time per call: enqueue-only, measured while the GPU is busy with a long op so
    # the queue never drains (host overhead, not kernel time).
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        pass
    big = torch.randn(8192, 8192, device=DEV)
    for _ in range(3):
        big @ big
    t0 = time.perf_counter()
    for _ in range(500):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"{name}: host enqueue {1e6 * (t1 - t0) / 500:.1f} us/call")

# --- torch stream handles: pooled? ------------------------------------------------------------
hs = [torch.cuda.Stream().cuda_stream for _ in range(40)]
print("distinct torch stream handles out of 40 Stream() objects:", len(set(hs)))
