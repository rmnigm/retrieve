"""BLOOM-BUILD-CHUNK fit check on pod d (controller 2026-10-11 02:00): (a) arXiv 3 M item signatures, the chunked
`build_transposed_sigs` (this tree) against staging's unchunked one, imported from STAGING's file, `torch.equal`, peaks above input;
(b) SilverTorch triton bloom builds on laion30m through the harness (inputs loaded, fp32 table resident), codesign-laion30m's
triton jobs (bloom_path partial + full, n_lists 16384): build seconds, peak CUDA memory (absolute and above the pre-build
allocation), one forward of 16 queries (attrs through `prepare_queries`, as the harness does since v2.6).
    cd evaluation && python fit_check_pod_d.py STAGING_BLOOM_HASH_PY out.json
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from bench import config, inputs, run
from retrieve.indexing.bloom_hash import (
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
)

DEV = torch.device("cuda")
spec = importlib.util.spec_from_file_location("staging_bloom_hash", sys.argv[1])
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)
out = {"staging_file": sys.argv[1]}


def peak(fn, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = fn(x)
    torch.cuda.synchronize()
    return t, (torch.cuda.max_memory_allocated() - base) / 2**20


inp = inputs.load_inputs(config.load_dataset(Path("config/arxiv.yaml"), 128), DEV)
attrs = inp["item_attrs"].to(DEV).long()
sigs = build_signatures(
    attrs,
    generate_seeds(5, device=DEV),
    1024,
    5,
    16,
    clause_salt=generate_clause_salt(attrs.shape[1], device=DEV),
)
del inp, attrs
new, new_mib = peak(build_transposed_sigs, sigs)
old, old_mib = peak(staging.build_transposed_sigs, sigs)
out["arxiv"] = {"n": sigs.shape[0], "words": sigs.shape[1], "equal": torch.equal(new, old),
                "chunked_peak_mib": new_mib, "staging_peak_mib": old_mib}  # fmt: skip
print(out["arxiv"], flush=True)
del sigs, new, old
torch.cuda.empty_cache()

jobs = [j for j in config.load_matrix(Path("config/laion30m.yaml"), Path("config/suites.yaml"), "codesign-laion30m",
                                      backends=["triton"], sweeps=["c0_domain"])]  # fmt: skip
inp = inputs.load_inputs(jobs[0].data, DEV)
out["laion30m"] = []
for j in jobs:
    assets = run.sweep_assets(j, inp, max(j.ks), DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    m = run.build_module(j, inp, assets, max(j.ks), {})
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    pk = torch.cuda.max_memory_allocated()
    rows = assets["keep"].nonzero().reshape(-1)[:16]
    with torch.inference_mode():
        ids, _ = m(
            inp["queries"][rows].to(DEV),
            m.prepare_queries(assets["qa_s"][rows].to(DEV)),
        )
    torch.cuda.synchronize()
    r = {"bloom_path": j.build.get("bloom_path"), "build": j.build, "build_s": build_s,
         "peak_gib": pk / 2**30, "peak_above_prebuild_gib": (pk - base) / 2**30,
         "prebuild_allocated_gib": base / 2**30, "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
         "forward_ids_shape": list(ids.shape), "forward_ids_valid": float((ids >= 0).float().mean())}  # fmt: skip
    out["laion30m"].append(r)
    print(r, flush=True)
    del m, assets, ids
    torch.cuda.empty_cache()
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
