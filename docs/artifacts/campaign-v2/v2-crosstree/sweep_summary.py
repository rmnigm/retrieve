"""programs sweep table per (cell, bs): eager / graph ms, fmkt kernel us, device us, window sm_mhz, and each
variant against the old tree's kernel. Usage: python sweep_summary.py OUT_DIR"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
for cell in ("pm", "ax", "axs"):
    rows = []
    for tree in ("old", "new"):
        f = out / f"sweep-{cell}-{tree}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows += d["rows"]
            print(
                f"\n== {cell} {tree}: {d['code_version'][:8]} pass {d['pass_rate']:.4f} N {d['n_items']}"
            )
    for bs in (1, 16):
        ref = next((r for r in rows if r["bs"] == bs and r["variant"] == "old"), None)
        for r in (r for r in rows if r["bs"] == bs):
            rel = ""
            if ref:
                rel = (
                    f" | /old: eager {r['eager_ms'] / ref['eager_ms']:.3f} "
                    f"graph {r['graph_ms'] / ref['graph_ms']:.3f} fmkt {r['fmkt_us'] / ref['fmkt_us']:.3f}"
                )
            print(
                f"bs {bs:2} {r['variant']:>6}: eager {r['eager_ms']:9.3f} graph {r['graph_ms']:9.3f} ms  "
                f"fmkt {r['fmkt_us']:10.1f} us  device {r['device_us']:10.1f} us  "
                f"sm_mhz e {r['eager_sm_mhz']} g {r['graph_sm_mhz']}{rel}"
            )
