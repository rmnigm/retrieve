"""WP-4 step 4: per-kernel CUDA times (torch.profiler) at D=128, layout A, three backends."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from bench_common import build, check_equal, cu, ct  # noqa: E402

cu.ensure_built()
ct.ensure_built()
N_ITERS = 20
ROWS = [("none", 1), ("none", 16), ("bloom", 1), ("bloom", 16), ("exact", 1), ("exact", 16)]
layout = sys.argv[2] if len(sys.argv) > 2 else "A"


def classify(name: str) -> str | None:
    n = name.lower()
    if "topk" in n or "gather" in n or "index" in n or "elementwise" in n or "fill" in n or "copy" in n or "reduce_kernel" in n or "vectorized" in n or "sort" in n or "radix" in n or "bitonic" in n:
        return None
    if "bloom_mask" in n:
        return "bloom mask"
    if "clause_mask" in n:
        return "clause mask"
    if "score" in n:
        return "scorer"
    return None


def per_kernel(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(N_ITERS):
            fn()
        torch.cuda.synchronize()
    tot = defaultdict(float)
    cnt = defaultdict(int)
    raw = defaultdict(float)
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        raw[ev.name] += ev.self_device_time_total if hasattr(ev, "self_device_time_total") else ev.cuda_time_total
        cls = classify(ev.name)
        if cls is None:
            continue
        tot[cls] += ev.self_device_time_total if hasattr(ev, "self_device_time_total") else ev.cuda_time_total
        cnt[cls] += 1
    per = {k: tot[k] / N_ITERS for k in tot}
    other = sum(raw.values()) / N_ITERS - sum(tot.values()) / N_ITERS
    return per, {k: v / N_ITERS for k, v in raw.items()}, other


results = []
for mode, b in ROWS:
    p, fns = build(layout, b, mode, cute_cfgs={"128.8.1": ct.CodesignedProbeScoreCuteConfig(128, 8, 1)})
    check_equal(fns)
    row = dict(mode=mode, B=b, P=p)
    for name, fn in fns.items():
        per, raw, other = per_kernel(fn)
        row[name] = per
        row[name + "_other_us"] = other
        row[name + "_raw"] = raw
    results.append(row)
    fmt = lambda d: " + ".join(f"{v:.1f} ({k})" for k, v in sorted(d.items(), key=lambda kv: kv[0] != "scorer"))
    print(f"{mode:<6} B={b:<3} P={p:<6} triton: {fmt(row['triton'])} | cuda: {fmt(row['cuda'])} | cute: {fmt(row['cute'])}"
          f" | cute@128.8.1: {fmt(row['cute@128.8.1'])}   [rest (topk etc.) ~ {row['cuda_other_us']:.1f} us]")
    for name in fns:
        print(f"    {name} raw kernels:", {k: round(v, 1) for k, v in row[name + "_raw"].items()})
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
