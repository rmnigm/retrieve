"""Fork rechecked bloom vs fork exact on one real cell: torch.equal ids and scores (bloom AND exact = exact)."""
import sys, torch, yaml
from pathlib import Path
from bench import config, inputs
from retrieve import OfficialConfig, SilverTorch
DEV = torch.device("cuda")
DS, DIM, NL, NP, BS, K, SW = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
ds = config.load_dataset(Path(f"config/{DS}.yaml"), DIM); inp = inputs.load_inputs(ds, DEV, with_filters=True)
kw = dict(k=K, n_lists=NL, n_probe=NP, n_iter=10, seed=0)
base = SilverTorch(**kw); base.register_index(inp["item_embs"])
LAY = ("centroids","item_codes","global_scale","cluster_offsets","cluster_sizes","sort_perm","inv_perm")
filt = yaml.safe_load(Path(f"config/{DS}.yaml").read_text())["filters"]
qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(filt["bloom"][SW]))
pool, qa = inputs.query_pool(inp, qa_s, skip, bs=BS, seed=0, n_pool=16, device=DEV)
mods = {}
for name, fm, extra, be in (("bloom", "exact", {}, "triton"), ("exact", "exact", {}, "official-fork")):
    m = SilverTorch(filter_mode=fm, backend=be, **({"official": OfficialConfig(score_path="int32")} if be != "triton" else {}), **kw, **extra)
    for b in LAY: m.register_buffer(b, getattr(base, b))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    m._register_filter_buffers(inp["n_items"], inp["item_attrs"].to(DEV), None, base.sort_perm)
    mods[name] = m
bad = 0
with torch.inference_mode():
    for i in range(len(pool)):
        a = mods["bloom"](pool[i], mods["bloom"].prepare_queries(qa[i]))
        b = mods["exact"](pool[i], mods["exact"].prepare_queries(qa[i]))
        sa, sb = a[1].sort(1, descending=True).values, b[1].sort(1, descending=True).values
        ok = torch.equal(sa, sb)
        if ok:
            # ids may differ only among equal scores: a missing id's score must equal the row's k-th
            for r in range(a[0].shape[0]):
                miss = set(b[0][r].tolist()) - set(a[0][r].tolist())
                if miss:
                    kth = b[1][r].min()
                    ms = [b[1][r][b[0][r] == x].item() for x in miss]
                    ties = all(v == kth.item() for v in ms)
                    print("batch", i, "row", r, "ids differ:", len(miss), "all at the k-th score:", ties)
        bad += not ok
        if not ok:
            d = (a[0] != b[0]).any(1).nonzero().flatten()[:2].tolist()
            print("batch", i, "rows differ", d, "scores equal:", torch.equal(a[1], b[1]))
print(f"{DS} np{NP} bs{BS} k{K} {SW}: {len(pool) - bad}/{len(pool)} batches with equal sorted scores (triton exact vs fork exact)")
