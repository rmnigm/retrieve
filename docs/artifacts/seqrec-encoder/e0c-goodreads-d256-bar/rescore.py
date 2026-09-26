"""E0c: the published goodreads-work-id d256 checkpoint re-scored on trainer/test.parquet (the bar
for R-g256; W8 flagged the stored value), through gate_a's path (load_model_for_eval + evaluate).

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/e0c-goodreads-d256-bar/rescore.py OUT.json
"""

import json
import sys
from pathlib import Path

import torch

from training.encode import load_model_for_eval
from training.evaluate import evaluate

DATA = "/data/goodreads-work-id"
CKPTS = [f"{DATA}/checkpoints/gsasrec-d256-drop0.5-id"]

device = torch.device("cuda")
num_items = len(json.loads(Path(f"{DATA}/trainer/item_id_map.json").read_text()))
out = []
for ckpt in CKPTS:
    model = load_model_for_eval(Path(ckpt) / "best_model.pt", num_items, device)
    got = evaluate(model, f"{DATA}/trainer/test.parquet", num_items=num_items, max_length=200,
                   batch_size=512, ks=(10, 100), device=device)
    row = {"ckpt": ckpt, "test": f"{DATA}/trainer/test.parquet", "metrics": got,
           "stored": json.loads((Path(ckpt) / "eval_quality.json").read_text())["metrics"]}
    print(json.dumps(row), flush=True)
    out.append(row)
    del model
    torch.cuda.empty_cache()
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
