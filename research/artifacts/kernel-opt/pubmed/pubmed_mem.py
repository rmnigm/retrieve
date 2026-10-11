import time, torch
from pathlib import Path
from bench import config, inputs, algos
from retrieve import SilverTorch
from retrieve.modules import silvertorch as st_mod
G = 2**30
dev = torch.device("cuda")
ds = config.load_dataset(Path("config/pubmed.yaml"), 768)
inp = inputs.load_inputs(ds, dev, with_filters=True)
e = inp["item_embs"]
print(f"inputs: item_embs {tuple(e.shape)} {e.dtype}; allocated {torch.cuda.memory_allocated()/G:.1f} GiB", flush=True)
m = SilverTorch(k=100, filter_mode="exact", seed=0, backend="triton", **{**algos.SILVERTORCH_DEFAULTS, "n_probe": 24})
phases = {}
def wrap(name):
    orig = getattr(m, name)
    def f(*a, **kw):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t = time.perf_counter(); out = orig(*a, **kw); torch.cuda.synchronize()
        phases[name] = (torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base, time.perf_counter() - t)
        return out
    setattr(m, name, f)
for n in ("_build_ivf", "_quantize_items", "_register_filter_buffers"):
    wrap(n)
m.register_index(e, inp["item_attrs"], inp["clause_is_reverse"])
for n, (peak, kept, s) in phases.items():
    print(f"{n:26s} transient peak +{peak/G:5.1f} GiB, kept +{kept/G:5.1f} GiB, {s:5.1f} s")
print(f"after build: allocated {torch.cuda.memory_allocated()/G:.1f} GiB, reserved {torch.cuda.memory_reserved()/G:.1f} GiB, width {m._probe_width}")
