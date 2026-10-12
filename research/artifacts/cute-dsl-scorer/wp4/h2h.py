"""WP-4 step 3: shared-input head-to-head (handoff §6b/§6c + no-filter), three backends."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import torch  # noqa: E402
import triton.testing as ttesting  # noqa: E402
from bench_common import LAYOUTS, build, check_equal, cu, ct  # noqa: E402

cu.ensure_built()
ct.ensure_built()
ROWS = [  # (mode, B, layout) -- every §13 row, plus B=1/P=1024 for the filtered modes
    ("none", 1, "S"), ("none", 1, "A"), ("none", 16, "S"), ("none", 16, "A"),
    ("bloom", 1, "S"), ("bloom", 1, "A"), ("bloom", 16, "S"), ("bloom", 16, "A"), ("bloom", 16, "B"),
    ("exact", 1, "S"), ("exact", 1, "A"), ("exact", 16, "S"), ("exact", 16, "A"), ("exact", 16, "B"),
]
results = []
print(f"{'mode':<6} {'B':>3} {'P':>6} {'pass':>5} {'triton':>8} {'cuda':>8} {'cute':>8} {'tri/cute':>9} {'cuda/cute':>9}")
for mode, b, layout in ROWS:
    p, fns = build(layout, b, mode)
    outs = check_equal(fns)
    # pass rate from the full score buffer: rerun phase 3 through the cuda prep path
    if mode == "none":
        pr = 1.0
    else:
        # top-k scores are finite iff >= k items passed; estimate the pass rate on the
        # full buffer by scoring with k = P through the cute impl (same inputs)
        n_lists, max_size, n_probe = LAYOUTS[layout]
        import bench_common as bc
        bc.K = p
        _, fns_full = build(layout, b, mode)
        _, s_full = fns_full["cuda"]()
        pr = torch.isfinite(s_full).float().mean().item()
        bc.K = 64
    ms = {}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        m, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
        ms[name] = m
    row = dict(mode=mode, B=b, P=p, layout=layout, pass_rate=pr, **ms)
    results.append(row)
    print(f"{mode:<6} {b:>3} {p:>6} {pr:>5.3f} {ms['triton']:>8.3f} {ms['cuda']:>8.3f} {ms['cute']:>8.3f} "
          f"{ms['triton']/ms['cute']:>8.2f}x {ms['cuda']/ms['cute']:>8.2f}x")
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
