"""Measured roofline anchors on this box: D2D copy bandwidth and the empty-kernel launch floor."""

import json
import statistics
import subprocess

import torch
import triton
import triton.language as tl


@triton.jit
def _empty(x):
    pass


def sm_mhz():
    q = ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader,nounits"]
    return [int(v) for v in subprocess.check_output(q, text=True).split(",")]


def windows(fn, iters, n_windows=20):
    fn()
    times, mhz = [], []
    for _ in range(n_windows):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / iters * 1e-3)
        mhz.append(sm_mhz()[0])  # under load: right after the window's sync
    return statistics.median(times), min(times), max(times), mhz


nbytes = 4 << 30  # 4 GiB per buffer
src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
dst = torch.empty_like(src)
t, t_min, t_max, mhz = windows(lambda: dst.copy_(src), iters=10)
dummy = torch.empty(1, device="cuda")
lt, lt_min, lt_max, lmhz = windows(lambda: _empty[(1,)](dummy), iters=1000)
print(json.dumps({
    "device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
    "copy_buffer_gib": nbytes / 2**30, "copy_s_median": t,
    "copy_gbps_rw_median": 2 * nbytes / t / 1e9, "copy_gbps_rw_best": 2 * nbytes / t_min / 1e9,
    "copy_gbps_rw_worst": 2 * nbytes / t_max / 1e9,
    "launch_us_median": lt * 1e6, "launch_us_min": lt_min * 1e6, "launch_us_max": lt_max * 1e6,
    "sm_mhz_copy": [min(mhz), max(mhz)], "sm_mhz_launch": [min(lmhz), max(lmhz)],
    "mem_mhz": sm_mhz()[1], "unstable": max(mhz + lmhz) - min(mhz + lmhz) > 50,
}, indent=1))
