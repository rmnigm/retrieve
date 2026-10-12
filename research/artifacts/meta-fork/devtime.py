"""Device time per kernel of one forward (bs, n_probe from argv) on goodreads bloom c0_genre, fork vs triton v2.12."""
import sys, importlib, collections, torch
from pathlib import Path
from bench import config, inputs
from retrieve import OfficialConfig, SilverTorch
from torch.profiler import profile, ProfilerActivity
DEV = torch.device("cuda"); bs, n_probe, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
DS, DIM, NL, SW = (sys.argv[4:8] + ["goodreads", "128", "4096", "c0_genre"][len(sys.argv[4:8]):])
ds = config.load_dataset(Path(f"config/{DS}.yaml"), int(DIM)); inp = inputs.load_inputs(ds, DEV, with_filters=True)
kw = dict(k=int(__import__("os").environ.get("K", "100")), n_lists=int(NL), n_probe=n_probe, n_iter=10, seed=0)
base = SilverTorch(**kw); base.register_index(inp["item_embs"])
LAY = ("centroids","item_codes","global_scale","cluster_offsets","cluster_sizes","sort_perm","inv_perm")
sw = None if mode == "none" else SW
fm = {"bloom": "bloom", "exact": "exact", "none": "none"}[mode]
import yaml; filt = yaml.safe_load(Path(f"config/{DS}.yaml").read_text())["filters"]
qa_s, skip = (None, None) if sw is None else inputs.sweep_qa(inp["qa"], tuple(filt["clause" if mode=="exact" else "bloom"][sw]))
pool, qa = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, n_pool=4, device=DEV)
for name, cls, backend in (("fork", SilverTorch, "official-fork"), ("triton", importlib.import_module("retrieve_v212.modules.silvertorch").SilverTorch, "triton")):
    extra = {"m_bits": 1024, "k_hash": 5} if mode == "bloom" else {}
    m = cls(filter_mode=fm, backend=backend, **({"official": OfficialConfig(score_path="int32", bloom_path=__import__("os").environ.get("BLOOM_PATH", "partial"), **({"bloom_width": "upstream", "recheck": False} if __import__("os").environ.get("FORK_BLOOM") == "meta" else {}))} if backend != "triton" else {}), **kw, **extra)
    for b in LAY: m.register_buffer(b, getattr(base, b))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    if mode != "none": m._register_filter_buffers(inp["n_items"], inp["item_attrs"].to(DEV), None, base.sort_perm)
    p = None if mode == "none" else m.prepare_queries(qa[0])
    with torch.inference_mode():
        for _ in range(5): m(pool[0], p)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20): m(pool[0], p)
            torch.cuda.synchronize()
    t = collections.Counter(); c = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            t[e.name[:90]] += e.device_time / 20; c[e.name[:90]] += 1/20
    print(f"== {name} bs{bs} np{n_probe} {mode}: total {sum(t.values()):.1f} us, {sum(c.values()):.0f} kernels")
    for n, v in t.most_common(14): print(f"  {v:8.1f} us x{c[n]:.0f} {n}")
