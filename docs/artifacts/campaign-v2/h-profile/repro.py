"""H-PROFILE repro: repeated profiler sessions (the pre-fix `profile_once` body) on one Triton SilverTorch
(bloom and none) in one process, with a timing burst between sessions as `run.perf` has.
Per session: CUDA kernels in key_averages (what profile_once keeps) vs raw kineto device events.
With `graph`, each session also captures both arms through `bench.measure.graph_callable`
(torch.compile reduce-overhead, its own profiler session) after the eager profile, as
`run.perf` does per (bs, k). With `fixed`, each session calls the fixed `measure.profile_once`
(all kernels) and prints its kernel count and wall seconds (> 0.02 s means a padded retry).
Usage: python repro.py [sessions] [burst_calls] [graph] [fixed]"""

import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

from bench import measure
from retrieve import SilverTorch

sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 40
burst = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
graph = "graph" in sys.argv[3:]
fixed = "fixed" in sys.argv[3:]
N, D, B = 200_000, 128, 16
g = torch.Generator(device="cuda").manual_seed(0)
embs = torch.nn.functional.normalize(torch.randn(N, D, generator=g, device="cuda"), dim=1)
q = torch.nn.functional.normalize(torch.randn(B, D, generator=g, device="cuda"), dim=1)
attrs = torch.randint(0, 50, (N, 1, 2), generator=g, device="cuda")
q_attrs = torch.randint(0, 50, (B, 1), generator=g, device="cuda")

bloom = SilverTorch(k=100, n_lists=1024, n_probe=24, filter_mode="bloom", m_bits=512, k_hash=5,
                    n_iter=3, backend="triton")
bloom.register_index(embs, attrs)
plain = SilverTorch(k=100, n_lists=1024, n_probe=24, n_iter=3, backend="triton")
plain.register_index(embs)
arms = {"bloom": lambda: bloom(q, q_attrs), "none": lambda: plain(q)}


def one(fn):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.time_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.time_ns()
    ka = [e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs = prof.profiler.kineto_results.events()
    dev = [e for e in evs if e.device_type() == torch.autograd.DeviceType.CUDA]
    launch = {e.correlation_id(): e.start_ns() for e in evs
              if e.device_type() == torch.autograd.DeviceType.CPU and "Launch" in e.name()}
    lag = sorted((e.start_ns() - launch[e.correlation_id()]) / 1e3 for e in dev
                 if e.correlation_id() in launch)
    cpu = [e for e in evs if e.device_type() == torch.autograd.DeviceType.CPU]
    # event clock vs wall clock: >0 = the profiler stamps events later than they happened
    skew = {"cpu_first_us": (min(e.start_ns() for e in cpu) - t0) / 1e3,
            "sync_end_us": (max(e.end_ns() for e in cpu if "Synchronize" in e.name()) - t1) / 1e3}
    return ka, [e.name() for e in dev], lag, skew


print(torch.__version__, __import__("triton").__version__, torch.cuda.get_device_name())
T0 = time.perf_counter()
with torch.inference_mode():
    for s in range(sessions):
        for name, fn in arms.items():
            if fixed:
                t = time.perf_counter()
                ks = measure.profile_once(fn, top=1000)
                print(f"session {s:3d} {name:5s} fixed_kernels={len(ks):3d} "
                      f"calls={sum(k['calls'] for k in ks):3d} s={time.perf_counter() - t:.3f} "
                      f"t={time.perf_counter() - T0:.0f}s", flush=True)
                for _ in range(burst):
                    fn()
                torch.cuda.synchronize()
                continue
            ka, raw, lag, skew = one(fn)
            trit = sorted({k for k in raw if not k.startswith(("void", "at::", "Memcpy", "Memset"))})
            print(f"session {s:3d} {name:5s} key_averages_cuda={len(ka):3d} raw_device={len(raw):3d} "
                  f"lag_us min={lag[0] if lag else float('nan'):.1f} med={lag[len(lag) // 2] if lag else float('nan'):.1f} "
                  f"skew_us cpu_first={skew['cpu_first_us']:.1f} sync_end={skew['sync_end_us']:.1f} "
                  f"t={time.perf_counter() - T0:.0f}s triton={trit}", flush=True)
            for _ in range(burst):
                fn()
            torch.cuda.synchronize()
        if graph:
            torch._dynamo.reset()
            measure.graph_callable(bloom, q, q_attrs)
            measure.graph_callable(plain, q)
