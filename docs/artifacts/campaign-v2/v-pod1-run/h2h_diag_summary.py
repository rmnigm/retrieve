"""Per arm and protocol: eager / graph median ms of each repeat, their median and spread, window
sm_mhz and the unstable flags; then official / triton ratios per protocol.
Usage: python h2h_diag_summary.py /scratch/h2h-diag"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
cells: dict = defaultdict(list)
for f in sorted(root.glob("[abc]/r*/h2h/goodreads-d128.jsonl")):
    proto, rep = f.parts[-4], f.parts[-3]
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok":
            print("NOT OK", f, r.get("status"))
            continue
        arm = (
            r["backend"]
            if r["backend"] == "triton"
            else f"official-{r['params']['score_path']}"
        )
        for p in r["perf"]:
            cells[arm, proto, p["mode"]].append(
                (rep, p["median_ms"], p["window_sm_mhz"], p["unstable"])
            )

med = {}
print(
    "arm | protocol | mode | per repeat ms (window sm_mhz, unstable) | median | spread (max-min)/median"
)
for (arm, proto, mode), v in sorted(cells.items()):
    ms = [m for _, m, _, _ in v]
    med[arm, proto, mode] = st.median(ms)
    reps = "; ".join(
        f"{rep} {m:.3f} ({'/'.join(str(int(x)) for x in mhz)}, {'U' if u else '-'})"
        for rep, m, mhz, u in v
    )
    print(
        f"{arm} | {proto} | {mode} | {reps} | {st.median(ms):.3f} | {(max(ms) - min(ms)) / st.median(ms):.3f}"
    )
print("\nofficial / triton (eager medians)")
for proto in "abc":
    t = med.get(("triton", proto, "eager"))
    for arm in ("official-fp16", "official-int32"):
        o = med.get((arm, proto, "eager"))
        if t and o:
            print(f"{proto} {arm} {o / t:.3f}")
