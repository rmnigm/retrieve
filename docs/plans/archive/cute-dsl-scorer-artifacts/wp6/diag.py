"""Where does the eager/compiled forward's host time go? CPU-side op breakdown per backend."""
import sys, importlib
sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp6")
import torch, torch._dynamo
import triton.testing as tt
from torch.profiler import profile, ProfilerActivity
import graphs_lib as gl

layout, mode, b = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = gl.make_data(layout)
q, qa = gl.make_query(b, mode)
for backend in ("triton", "cuda", "cute"):
    m = gl.build_module(data, mode, backend)
    for variant in ("eager", "compile"):
        torch._dynamo.reset()
        f = m if variant == "eager" else torch.compile(m, fullgraph=True)
        fn = lambda: f(q, qa)
        for _ in range(5): fn()
        torch.cuda.synchronize()
        ms = [tt.do_bench(fn, rep=300, warmup=50, return_mode="median") for _ in range(3)]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20): fn()
            torch.cuda.synchronize()
        print(f"=== {backend} {variant} {mode} {layout} B={b}: do_bench {['%.4f' % x for x in ms]} ms")
        ka = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)
        for e in ka[:14]:
            print(f"   {e.key[:60]:<60} cpu_total {e.cpu_time_total/20:8.1f} us  self {e.self_cpu_time_total/20:8.1f} us  n={e.count//20}  dev={e.device_time_total/20:7.1f} us")
    del m; torch.cuda.empty_cache()
