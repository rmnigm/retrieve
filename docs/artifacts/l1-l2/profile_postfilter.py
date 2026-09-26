"""L1: where `PostfilterKNN`'s B=16 forward spends its time, fp16 vs fp32 scores (3M × 128,
the bench_kernels.py shape), by torch.profiler CUDA time over 50 calls.

    python profile_postfilter.py [--src /path/to/retrieve/src]
"""

import sys

if "--src" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--src") + 1])

import torch
from torch.profiler import ProfilerActivity, profile

from retrieve.modules.knn import PostfilterKNN

g = torch.Generator(device="cuda").manual_seed(0)
embs = torch.randn(3_000_000, 128, device="cuda", generator=g).half()
q = torch.randn(16, 128, device="cuda", generator=g)
mask = torch.rand(16, 3_000_000, device="cuda", generator=g) < 0.3
m = PostfilterKNN(k=100)
m.register_index(embs)
for _ in range(10):
    m(q, mask)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(50):
        m(q, mask)
    torch.cuda.synchronize()
print(
    prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=8, max_name_column_width=60
    )
)
