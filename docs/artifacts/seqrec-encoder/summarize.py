"""One results row per E-run from its checkpoint dir, copies the small outputs into the artifact
dir, and prints the row with deltas against the run's bar.

    python3 docs/artifacts/seqrec-encoder/summarize.py RUN_ID BAR_NDCG10 BAR_R100
"""

import json
import shutil
import statistics
import sys
from pathlib import Path

run, bar_n, bar_r = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ckpt = Path("/scratch/ckpt") / run
art = Path(__file__).parent / run
art.mkdir(exist_ok=True)
for f in ("config.json", "train_metrics.json", "eval_quality.json"):
    shutil.copy2(ckpt / f, art / f)

tm = json.loads((ckpt / "train_metrics.json").read_text())
t = tm["test_metrics"]
val = tm["val_metrics_per_epoch"]
best = max(val, key=lambda v: v["ndcg@10"])
row = {
    "run": run,
    "test": {k: round(t[k], 4) for k in ("ndcg@10", "ndcg@100", "recall@10", "recall@100", "coverage@10")},
    "delta_vs_bar": {"ndcg@10": round(t["ndcg@10"] - bar_n, 4), "recall@100": round(t["recall@100"] - bar_r, 4)},
    "bar": {"ndcg@10": bar_n, "recall@100": bar_r},
    "best_epoch": best["epoch"],
    "best_val_ndcg@10": round(best["ndcg@10"], 4),
    "epochs_run": len(tm["epoch_time_sec"]),
    "epoch_time_median_s": round(statistics.median(tm["epoch_time_sec"]), 2),
    "first_epoch_s": round(tm["epoch_time_sec"][0], 1),
    "samples_per_s": round(tm["samples_per_sec"]),
    "peak_gpu_mem_gb": round(tm["peak_gpu_mem_bytes"] / 1024**3, 1),
    "gpu": tm["gpu_name"],
    "sm_mhz": sorted(set(tm["sm_mhz"])),
    "total_wall_s": round(tm["total_time_sec"]),
}
(art / "result.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row, indent=2))
