"""CLAUSE-SKIP gates, one dataset per process: this tree (clause_pass skips inactive clauses' item loads) against staging
c354bb5 (package `retrieve_stg`, campaign-v2/st-ids/make_pkg.sh with the quoted-path fix), on the 10-clause arxiv-synth
table or goodreads' real attrs. Modules: LiNR V1 and V2 on an ExactAttributeFilter, SilverTorch exact on triton
(n_lists 1024, n_probe 24; one index, the after module's state dict loaded into the before one), and SilverTorch exact
on official (bit-exact only). Per sweep:
  exact  ids + scores `torch.equal` on 8 pool batches at bs 16 and 4 at bs 1, k 100
  time   interleaved ABAB windows (eager, and graph = bench.measure.graph_callable), bs {1, 16}, k 100: after / before,
         median [min, max] over windows, the SM clock

    cd evaluation && PYTHONPATH=<pkgs>:.:../retrieve/src:../retrieve python cs_gate.py {arxiv-synth|goodreads} out.json
"""

import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from bench import config, inputs, measure

DEV = torch.device("cuda")
SWEEPS = {
    "arxiv-synth": ["p0001", "p01", "p1"],
    "goodreads": ["c0_genre", "c1_lang_reverse", "all4"],
}
PKG = {"before": "retrieve_stg", "after": "retrieve"}
MOD = {a: importlib.import_module(f"{p}.modules") for a, p in PKG.items()}
CALLS, WINDOWS, N_POOL = 100, 8, 32

ds, out = sys.argv[1], Path(sys.argv[2])
cfg_path = Path(f"config/{ds}.yaml")
data = config.load_dataset(cfg_path, 128)
inp = inputs.load_inputs(data, DEV, with_filters=True)
item_embs = inp["item_embs"].to(DEV)
attrs = inp["item_attrs"].to(DEV)
rev = inp["clause_is_reverse"]
rev = None if rev is None else rev.to(DEV)
clause_map = yaml.safe_load(cfg_path.read_text())["filters"]["clause"]


def sm_mhz():
    q = [
        "nvidia-smi",
        "--query-gpu=clocks.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    return int(subprocess.check_output(q, text=True).split()[0])


def build(arm):
    m = MOD[arm]
    f = m.ExactAttributeFilter(backend="triton")
    f.register_index(attrs, clause_is_reverse=rev)
    v1 = m.LiNRV1(100, filter=f)
    v1.register_index(item_embs)
    v2 = m.LiNRV2(100, filter=f)
    v2.register_index(item_embs)
    st = {}
    for backend in ("triton", "official"):
        s = m.SilverTorch(
            k=100,
            n_lists=1024,
            n_probe=24,
            filter_mode="exact",
            n_iter=10,
            seed=0,
            backend=backend,
        )
        s.register_index(item_embs, item_clause_attrs=attrs, clause_is_reverse=rev)
        st[backend] = s
    return {
        "linr_v1": v1,
        "linr_v2": v2,
        "st_triton": st["triton"],
        "st_official": st["official"],
    }


after = build("after")
before = build("before")
for name in ("st_triton", "st_official"):
    before[name].load_state_dict(after[name].state_dict())


def window(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        fn(i % N_POOL)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CALLS * 1e3


rows = []
with torch.inference_mode():
    for sweep in SWEEPS[ds]:
        qa_s, skip = inputs.sweep_qa(inp["qa"], tuple(clause_map[sweep]))
        for bs in (16, 1):
            pool, qa_pool = inputs.query_pool(
                inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
            )
            for name in after:
                prep = {a: [mods[name].prepare_queries(qa_pool[i]) for i in range(N_POOL)]
                        for a, mods in (("before", before), ("after", after))}  # fmt: skip
                equal = all(
                    all(torch.equal(x, y) for x, y in zip(
                        before[name](pool[i], prep["before"][i]), after[name](pool[i], prep["after"][i]), strict=True))
                    for i in range(8 if bs == 16 else 4)
                )  # fmt: skip
                row = {
                    "dataset": ds,
                    "sweep": sweep,
                    "bs": bs,
                    "module": name,
                    "equal": equal,
                }
                modes = ("eager",) if name == "st_official" else ("eager", "graph")
                for mode in modes:
                    calls = {}
                    if mode == "graph":
                        torch._dynamo.reset()
                    for a, mods in (("before", before), ("after", after)):
                        m = mods[name]
                        if mode == "graph":
                            m = measure.graph_callable(m, pool[0], prep[a][0])
                        calls[a] = lambda i, m=m, a=a: m(pool[i], prep[a][i])
                    for fn in calls.values():
                        window(fn)
                    t = {a: [] for a in calls}
                    mhz = []
                    for _ in range(WINDOWS):
                        for a, fn in calls.items():
                            t[a].append(window(fn))
                        mhz.append(sm_mhz())
                    r = [x / y for x, y in zip(t["after"], t["before"], strict=True)]
                    row[mode] = {"before_ms": statistics.median(t["before"]), "after_ms": statistics.median(t["after"]),
                                 "ratio": [statistics.median(r), min(r), max(r)], "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
                rows.append(row)
                print(json.dumps(row), flush=True)
            torch._dynamo.reset()
out.write_text(json.dumps(rows, indent=1))
print("ALL EQUAL" if all(r["equal"] for r in rows) else "MISMATCH", flush=True)
