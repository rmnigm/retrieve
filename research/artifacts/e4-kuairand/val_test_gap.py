import json, torch
from pathlib import Path
from training.encode import load_model_for_eval
from training.evaluate import evaluate
D = Path("data/kuairand"); C = D / "checkpoints/gsasrec-d128-shared/best_model.pt"
n = len(json.load(open(D / "item_id_map.json")))
m = load_model_for_eval(C, num_items=n, device=torch.device("cuda"))
for split, mu in [("test", 4096), ("val", None)]:
    r = evaluate(m, str(D / f"{split}.parquet"), num_items=n, max_length=200, batch_size=512,
                 ks=(10, 100), device="cuda", max_users=mu)
    print(split, mu, json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
