"""E2b was stopped during epoch 15 by user decision, before train() wrote config.json or scored test.
Rebuild config.json from command.sh's overrides, then score the best epoch checkpoint
(hstu-ep13) on val (must reproduce the logged 0.0301) and test through load_model_for_eval +
evaluate with the training run's eval settings.

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/e2b-yambda-d64-hstu-gbce/score_partial.py
"""

import json
from pathlib import Path

import torch

from training.config import TrainConfig
from training.encode import load_model_for_eval
from training.evaluate import evaluate

RUN = "e2b-yambda-d64-hstu-gbce"
CKPT = Path("/scratch/ckpt") / RUN
ART = Path(__file__).parent
cfg = TrainConfig(
    data_dir="/data/yambda-500m/trainer", checkpoint_dir=str(CKPT),
    encoder="hstu", embedding_dim=64, hidden_dim=256, num_blocks=4, num_heads=4, dropout=0.2,
    use_time=True, loss="gbce", normalize=False, num_negatives=256, gbce_t=0.75,
    batch_size=256, max_seq_length=200, learning_rate=1e-3, weight_decay=0,
    warmup_steps=1000, num_epochs=100, patience=10, eval_every=2, eval_batch_size=1024,
    early_stop_metric="ndcg@10", seed=42, compile=True, wandb_enabled=True,
    wandb_project="seqrec-encoder", wandb_run_name=RUN,
)
cfg.save(CKPT / "config.json")
cfg.save(ART / "config.json")

device = torch.device("cuda")
num_items = cfg.num_items
model = load_model_for_eval(CKPT / "hstu-ep13-ndcg100.0301.pt", num_items, device)
kw = dict(num_items=num_items, max_length=200, batch_size=1024, ks=(10, 100), device=device)
val = evaluate(model, f"{cfg.data_dir}/val.parquet", **kw)
test = evaluate(model, f"{cfg.data_dir}/test.parquet", **kw)
print(json.dumps({"val": val, "test": test}, indent=2))
(ART / "eval_quality.json").write_text(json.dumps(
    {"split": "test", "ks": [10, 100], "mask_history": False, "checkpoint": "hstu-ep13",
     "stopped_at_epoch": 15, "metrics": test, "val_rescored": val}, indent=2) + "\n")
