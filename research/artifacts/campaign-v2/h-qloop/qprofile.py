"""H-QLOOP: cProfile of one cell's `run.quality` on the GPU (py-spy cannot attach in the pod container), the
harness's own build path; prints the top functions by cumulative and self time and the wall of the call.

    python qprofile.py DATASET SUITE ALGO SWEEP OUT_TXT     (cwd evaluation/; PYTHONPATH picks the tree)"""

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import torch
from bench import inputs, measure, run
from bench.config import load_matrix

ds, suite, algo, sweep, out = sys.argv[1:]
dev = torch.device("cuda")
jobs = load_matrix(
    Path(f"config/{ds}.yaml"), Path("config/suites.yaml"), suite, algos=[algo]
)
job = next(
    j
    for j in jobs
    if j.backend == "triton"
    and j.sweep == sweep
    and j.seed == 0
    and j.filter_kind == "clause"
)
params = {**job.build, **job.query[0]}
measure.setup(0)
inp = inputs.load_inputs(job.data, dev, with_filters=True)
assets = run.sweep_assets(job, inp, max(job.ks), dev)
module = run.build_module(
    job, inp, assets, max(job.ks), run.resolve_pool(params, assets)
)
module.k = max(job.ks)
run.quality(module, inp, assets, job.ks, dev)  # warm
torch.cuda.synchronize()
prof = cProfile.Profile()
t0 = time.perf_counter()
prof.enable()
run.quality(module, inp, assets, job.ks, dev)
torch.cuda.synchronize()
prof.disable()
wall = time.perf_counter() - t0
s = io.StringIO()
st = pstats.Stats(prof, stream=s)
st.sort_stats("cumulative").print_stats(30)
st.sort_stats("tottime").print_stats(25)
head = f"{ds}/{suite} {algo} {sweep} params {params}: quality wall {wall:.2f} s, torch threads {torch.get_num_threads()}\n"
Path(out).write_text(head + s.getvalue())
print(head)
