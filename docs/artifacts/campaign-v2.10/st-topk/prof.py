"""ST-TOPK, measured first: per cell, one forward's device time split by kernel family (torch.profiler, CALLS calls,
µs per call): our triton SilverTorch and the official arm (Meta's scorer + our adapter's epilogue) on the same index
build. Families: scorer, topk (`at::native` top-k / radix select / sort kernels), ids (our id epilogue / gathers),
phase1 (the centroid gemm), prep, other. Also each family's launch count and the top-k input's dtype / width.

    cd evaluation && PYTHONPATH=.:../retrieve/src python prof.py DATASET DIM KIND SWEEP N_LISTS MODE NPROBES BSS OUT.json [official]
"""

import json
import re
import sys
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from torch.profiler import ProfilerActivity, profile

from retrieve.modules.silvertorch import OfficialConfig, SilverTorch

DEV = torch.device("cuda")
CALLS = 20
OFFICIAL = "official" in sys.argv
ds, dim, kind, sweep, n_lists, mode, nprobes, bss, out = [
    a for a in sys.argv[1:] if a != "official"
]
cfg_path = Path(f"config/{ds}.yaml")
inp = inputs.load_inputs(
    config.load_dataset(cfg_path, int(dim)), DEV, with_filters=mode != "none"
)
kw = {"k": 100, "n_lists": int(n_lists), "n_probe": 24, "n_iter": 10, "seed": 0}
if mode != "none":
    kw["filter_mode"] = mode
if mode == "bloom":
    kw |= {"m_bits": 1024, "k_hash": 5}
arms = {"triton": SilverTorch(**kw)}
if OFFICIAL:
    arms["official"] = SilverTorch(
        **kw, backend="official", official=OfficialConfig(score_path="int32")
    )
for m in arms.values():
    m.register_index(
        inp["item_embs"].to(DEV),
        *(() if mode == "none" else (inp["item_attrs"].to(DEV),)),
    )
del inp["item_embs"]
torch.cuda.empty_cache()
qa_s, skip = (
    (None, None)
    if mode == "none"
    else inputs.sweep_qa(
        inp["qa"], tuple(yaml.safe_load(cfg_path.read_text())["filters"][kind][sweep])
    )
)
FAMILIES = [
    (
        "topk",
        re.compile(
            r"topk|TopK|radix|Radix|DigitCount|DigitCumSum|WithinK|sort|Sort", re.I
        ),
    ),
    (
        "scorer",
        re.compile(r"probe_score|bloom_filter|bloom_dot|fused_kmean|ann|search", re.I),
    ),
    ("ids", re.compile(r"probe_ids|gather|index|Index", re.I)),
    ("phase1", re.compile(r"gemm|sgemm|cutlass|ampere|matmul", re.I)),
    ("prep", re.compile(r"probe_prep|quantiz|abs|amax|round|clamp|copy", re.I)),
]


def family(name):
    return next((f for f, rx in FAMILIES if rx.search(name)), "other")


rows = []
with torch.inference_mode():
    for n_probe in map(int, nprobes.split(",")):
        for m in arms.values():
            m.set_query_params(n_probe=n_probe)
        for bs in map(int, bss.split(",")):
            pool, qa_pool = inputs.query_pool(
                inp, qa_s, skip, bs=bs, seed=0, n_pool=4, device=DEV
            )
            for arm, m in arms.items():
                prep = [
                    None if mode == "none" else m.prepare_queries(qa_pool[i])
                    for i in range(4)
                ]
                for i in range(10):
                    m(pool[i % 4], prep[i % 4])
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    for i in range(CALLS):
                        m(pool[i % 4], prep[i % 4])
                    torch.cuda.synchronize()
                fam, calls, kern = {}, {}, []
                for e in p.key_averages():
                    if (
                        e.device_type != torch.autograd.DeviceType.CUDA
                        or e.key.startswith("##")
                    ):
                        continue
                    f = family(e.key)
                    fam[f] = fam.get(f, 0.0) + e.self_device_time_total / CALLS
                    calls[f] = calls.get(f, 0.0) + e.count / CALLS
                    kern.append(
                        (
                            e.key[:90],
                            round(e.self_device_time_total / CALLS, 1),
                            e.count / CALLS,
                            f,
                        )
                    )
                kern.sort(key=lambda x: -x[1])
                row = {"dataset": ds, "dim": int(dim), "mode": mode, "sweep": sweep, "arm": arm, "n_probe": n_probe,
                       "bs": bs, "width": arms["triton"]._probe_width, "family_us": fam, "family_launches": calls,
                       "total_us": sum(fam.values()), "kernels": kern[:12]}  # fmt: skip
                rows.append(row)
                print(f"{ds} d{dim} {mode} {arm:8s} np{n_probe:5d} bs{bs:3d} total {row['total_us']:8.1f} us | "
                      + " ".join(f"{k} {v:.0f}" for k, v in sorted(fam.items(), key=lambda x: -x[1])), flush=True)  # fmt: skip
Path(out).write_text(json.dumps(rows, indent=1))
