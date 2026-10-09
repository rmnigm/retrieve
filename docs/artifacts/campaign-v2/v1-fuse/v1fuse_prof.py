"""V1-FUSE: per-kernel device time of the new LiNRV1 against the old composition (one torch.profiler session each,
20 calls after warm-up), arxiv-synth d128, sweep p001, bs 1 and 16, k 100. Run from evaluation/, GPU 0.
  python v1fuse_prof.py FILTER_KIND OUT.json"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench import algos, inputs
from bench.config import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from v1fuse_gpu import DEV, OldV1  # noqa: E402

fk, out_path = sys.argv[1:]
ds = load_dataset(Path("config/arxiv-synth.yaml"), 128)
inp = inputs.load_inputs(ds, DEV)
filters = inputs.build_filters(fk, inp, ["triton"], bloom=algos.BLOOM_DEFAULTS)
new = algos.build("linr_v1_filter_mask", inp["item_embs"], k=100, backend="triton", filter_kind=fk,
                  filter_mod=filters["triton"])  # fmt: skip
old = OldV1(new)
qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk]["p001"])
res = {}
for bs in (1, 16):
    pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, device=DEV)
    for name, m in (("new", new), ("old", old)):
        with torch.inference_mode():
            for i in range(20):
                m(pool[i], qa_pool[i])
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(20):
                    m(pool[i], qa_pool[i])
                torch.cuda.synchronize()
        k = defaultdict(lambda: [0.0, 0])
        for e in prof.key_averages():
            if e.device_time_total > 0:
                k[e.key][0] += e.device_time_total / 20
                k[e.key][1] += e.count / 20
        res[f"{name}_bs{bs}"] = sorted(([n, round(us, 1), c] for n, (us, c) in k.items()), key=lambda x: -x[1])
        print(name, bs, res[f"{name}_bs{bs}"][:8], flush=True)
Path(out_path).write_text(json.dumps(res, indent=1))
