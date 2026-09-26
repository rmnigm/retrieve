"""Gate A: published checkpoints through load_model_for_eval + evaluate reproduce their
eval_quality.json test ndcg@10 / recall@100 to 4 decimals.

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/gate-a/gate_a.py OUT.json
"""

import json
import sys
from pathlib import Path

import torch

from training.encode import load_model_for_eval
from training.evaluate import evaluate

YDATA = "/data/yambda-500m"
CASES = [
    ("/data/goodreads-work-id/checkpoints/gsasrec-d64-drop0.5-id", "/data/goodreads-work-id"),
    ("/data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id", "/data/goodreads-work-id"),
    ("/data/yambda-500m/checkpoints/gsasrec-d64-drop0.5", YDATA),
    ("/data/yambda-500m/checkpoints/gsasrec-d128-drop0.5", YDATA),
]

device = torch.device("cuda")
out = []
for ckpt, data in CASES:
    num_items = len(json.loads((Path(data) / "item_id_map.json").read_text()))
    model = load_model_for_eval(Path(ckpt) / "best_model.pt", num_items, device)
    got = evaluate(model, f"{data}/test.parquet", num_items=num_items, max_length=200,
                   batch_size=512, ks=(10, 100), device=device)
    ref = json.loads((Path(ckpt) / "eval_quality.json").read_text())["metrics"]
    row = {"ckpt": ckpt, "data": data}
    for m in ("ndcg@10", "recall@100"):
        row[m] = got[m]
        row[f"{m}_ref"] = ref[m]
        row[f"{m}_match4"] = round(got[m], 4) == round(ref[m], 4)
    print(json.dumps(row), flush=True)
    out.append(row)
    del model
    torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
