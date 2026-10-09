"""OFFICIAL-REWORK M1 keep rule: the SilverTorch forward with its filter prepared outside the call (this tree) against
the pre-M1 forward that encoded the filter inside (staging 5618bb6, `retrieve_pre`; its official plans parsed per call,
as campaign runs set `cache_plans=False`), interleaved ABAB windows of rotating pool batches, host wall per call, one
sync per window, on one shared index per path (m1_gate.py's build + state-dict load). Per path: `after / before` for
the forward alone, `(prep + after) / before` for the same work, and `prep_ms` (one batch's `prepare_queries`), each
the median over windows with min / max.

    cd evaluation && PYTHONPATH=<pkgs>:.:../retrieve/src:../retrieve python m1_time.py {goodreads|arxiv} out.json
"""

import dataclasses
import importlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from bench import config, inputs

DEV = torch.device("cuda")
SWEEP = {"goodreads": "c0_genre", "arxiv": "c0_maincat"}
ST = {
    a: importlib.import_module(f"{p}.modules.silvertorch")
    for a, p in (("before", "retrieve_pre"), ("after", "retrieve"))
}
CALLS, WINDOWS, N_POOL = 200, 8, 32

ds, out = sys.argv[1], Path(sys.argv[2])
cfg_path = Path(f"config/{ds}.yaml")
data = config.load_dataset(cfg_path, 128)
inp = inputs.load_inputs(data, DEV, with_filters=True)
clauses = tuple(yaml.safe_load(cfg_path.read_text())["filters"]["clause"][SWEEP[ds]])
qa_s, skip = inputs.sweep_qa(inp["qa"], clauses)
item_embs, attrs = inp["item_embs"].to(DEV), inp["item_attrs"].to(DEV)
PATHS = [("triton", "none", None), ("triton", "exact", None), ("triton", "bloom", "partial"),
         ("triton", "bloom", "full"), ("official", "none", None), ("official", "exact", None),
         ("official", "bloom", "partial"), ("official", "bloom", "full")]  # fmt: skip


def sm_mhz():
    q = [
        "nvidia-smi",
        "--query-gpu=clocks.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    return int(subprocess.check_output(q, text=True).split()[0])


def build(arm, backend, mode, path):
    kw = {
        "k": 100,
        "n_lists": 1024,
        "n_probe": 32,
        "n_iter": 10,
        "seed": 0,
        "backend": backend,
    }
    if mode != "none":
        kw["filter_mode"] = mode
    if mode == "bloom":
        kw.update(m_bits=1024 if backend != "official" else None, k_hash=5)
    if backend == "official":
        kw["official"] = ST[arm].OfficialConfig(
            score_path="fp16", **({"bloom_path": path} if path else {})
        )
    elif path == "full":
        kw["bloom_path"] = "full"
    m = ST[arm].SilverTorch(**kw)
    m.register_index(item_embs, *(() if mode == "none" else (attrs,)))
    if arm == "before" and backend == "official":
        m.official = dataclasses.replace(
            m.official, cache_plans=False
        )  # the campaign's timed setting
    return m


def window(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        fn(i % N_POOL)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CALLS * 1e3


rows = []
with torch.inference_mode():
    for bs in (16, 1):
        pool, qa_pool = inputs.query_pool(
            inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
        )
        for backend, mode, path in PATHS:
            after, before = (
                build("after", backend, mode, path),
                build("before", backend, mode, path),
            )
            before.load_state_dict(after.state_dict())
            prep = [
                None if mode == "none" else after.prepare_queries(qa_pool[i])
                for i in range(N_POOL)
            ]
            arms = {
                "before": lambda i: before(
                    pool[i], None if mode == "none" else qa_pool[i]
                ),
                "after": lambda i: after(pool[i], prep[i]),
                "prep+after": lambda i: after(
                    pool[i],
                    None if mode == "none" else after.prepare_queries(qa_pool[i]),
                ),
            }
            for fn in arms.values():
                window(fn)
            t = {a: [] for a in arms}
            mhz = []
            for _ in range(WINDOWS):
                for a, fn in arms.items():
                    t[a].append(window(fn))
                mhz.append(sm_mhz())
            prep_ms = (
                None if mode == "none"
                else statistics.median(t["prep+after"]) - statistics.median(t["after"])
            )  # fmt: skip
            r = [a / b for a, b in zip(t["after"], t["before"], strict=True)]
            rt = [a / b for a, b in zip(t["prep+after"], t["before"], strict=True)]
            row = {"dataset": ds, "bs": bs, "backend": backend, "mode": mode, "bloom_path": path,
                   "before_ms": statistics.median(t["before"]), "after_ms": statistics.median(t["after"]),
                   "prep_ms": prep_ms, "after/before": [statistics.median(r), min(r), max(r)],
                   "(prep+after)/before": [statistics.median(rt), min(rt), max(rt)], "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
            rows.append(row)
            print(f"{ds} bs={bs} {backend:8} {mode:5} {path or '':7} before {row['before_ms']:.3f} after "
                  f"{row['after_ms']:.3f} ms  after/before {row['after/before'][0]:.3f} [{min(r):.3f}, {max(r):.3f}]  "
                  f"(prep+after)/before {row['(prep+after)/before'][0]:.3f}  prep {prep_ms if prep_ms is None else round(prep_ms, 3)}"
                  f"  sm {row['sm_mhz']}", flush=True)  # fmt: skip
            torch.cuda.empty_cache()
out.write_text(json.dumps(rows, indent=1))
