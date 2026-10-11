"""Why is compile-default slower than eager for the opaque two-kernel ops? Micro-timings."""
import sys
sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp6")
import torch, torch._dynamo
import triton.testing as tt
import graphs_lib as gl
from graphs_lib import cu, ct

layout, mode, b = "S", "none", 1
data = gl.make_data(layout)
q, qa = gl.make_query(b, mode)
K = gl.K
def db(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    return tt.do_bench(fn, rep=300, warmup=50, return_mode="median")

for backend, mod, opname in (("cuda", cu, "codesigned_probe_score_cuda"), ("cute", ct, "codesigned_probe_score_cute")):
    m = gl.build_module(data, mode, backend)
    flat = m._phase1_probe(q)
    gs = m._global_scale_f
    impl = getattr(mod, f"_{opname}_impl")
    opobj = getattr(mod, opname)
    opdef = getattr(torch.ops.retrieve, opname).default
    torch._dynamo.reset()
    cm = torch.compile(m, fullgraph=True)
    torch._dynamo.reset()
    fns = {
        "_impl": lambda: impl(q, flat, m.item_codes, gs, K),
        "op object (python)": lambda: opobj(q, flat, m.item_codes, gs, K),
        "torch.ops...default": lambda: opdef(q, flat, m.item_codes, gs, K),
        "forward eager": lambda: m(q, qa),
        "forward eager no_grad": lambda: torch.no_grad()(m)(q, qa),
        "forward compiled": lambda: cm(q, qa),
        "forward compiled no_grad": lambda: torch.no_grad()(cm)(q, qa),
    }
    with torch.inference_mode():
        fns["forward compiled inference_mode"] = lambda: cm(q, qa)
        fns["forward eager inference_mode"] = lambda: m(q, qa)
    print(f"=== {backend}")
    for name, fn in fns.items():
        if "inference_mode" in name:
            with torch.inference_mode():
                torch._dynamo.reset(); cm2 = torch.compile(m, fullgraph=True)
                f2 = (lambda: cm2(q, qa)) if "compiled" in name else (lambda: m(q, qa))
                print(f"   {name:<36} {db(f2):.4f} ms")
        else:
            print(f"   {name:<36} {db(fn):.4f} ms")
    del m
