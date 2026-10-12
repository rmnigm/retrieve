"""WP-4 step 6: no-filter scorer kernel time, pinned (cutlass.range unroll=1) vs unpinned item
loop, as the module currently stands on disk. Run once per variant. Also counts the dp4a /
ld.global.cs.v4 copies in the <8,False,1> PTX to prove which structure was built."""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

W = "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4"
tag = sys.argv[1]
os.environ["CUTE_DSL_KEEP"] = "ptx,sass"
os.environ["CUTE_DSL_DUMP_DIR"] = f"{W}/dump_{tag}"
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)
sys.path.insert(0, W)
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from bench_common import build, check_equal, ct  # noqa: E402

ct.ensure_built()
dev = ct._load_dev()
N_ITERS = 30


def scorer_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N_ITERS):
            fn()
        torch.cuda.synchronize()
    tot = defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "score" in ev.name and "topk" not in ev.name.lower():
            tot[ev.name] += ev.self_device_time_total
    assert len(tot) == 1, tot
    return next(iter(tot.values())) / N_ITERS


cfgs = {"128.8.1": ct.CodesignedProbeScoreCuteConfig(128, 8, 1), "256.8.4": ct.CodesignedProbeScoreCuteConfig(256, 8, 4),
        "128.8.4": ct.CodesignedProbeScoreCuteConfig(128, 8, 4), "512.4.1": ct.CodesignedProbeScoreCuteConfig(512, 4, 1)}
for mode, b in (("none", 16), ("none", 1), ("bloom", 16)):
    p, fns = build("A", b, mode, cute_cfgs=cfgs)
    check_equal(fns)
    line = f"[{tag}] {mode} B={b}: cuda {scorer_us(fns['cuda']):.1f}"
    for label in cfgs:
        line += f" | cute@{label} {scorer_us(fns['cute@' + label]):.1f}"
    print(line + "  (scorer us)")
for key in (("score", 8, False, 1), ("score", 8, False, 4), ("score", 8, True, 1), ("score", 8, True, 4)):
    fn = dev._compiled[key]
    ptx = fn.__ptx__ or ""
    print(f"[{tag}] {key}: dp4a x{len(re.findall(r'dp4a', ptx))}, ld.global.cs.v4 x{len(re.findall(r'ld.global.cs.v4', ptx))}, "
          f"redux x{len(re.findall(r'redux.sync', ptx))}")
