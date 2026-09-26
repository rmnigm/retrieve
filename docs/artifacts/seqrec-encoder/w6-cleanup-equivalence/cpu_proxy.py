import dataclasses, json, sys, tempfile, random
from pathlib import Path
import polars as pl
import training.train as tt
from training.config import TrainConfig

root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
rng = random.Random(0)
pl.DataFrame(
    {
        "item_ids": [
            [rng.randint(1, 500) for _ in range(rng.randint(2, 40))] for _ in range(300)
        ]
    }
).write_parquet(root / "train.parquet")
(root / "item_id_map.json").write_text(json.dumps({str(i): i for i in range(1, 501)}))
recipes = {
    "ssm": dict(
        loss="sampled_softmax",
        normalize=True,
        logq=True,
        num_negatives=64,
        inbatch_negatives=32,
        warmup_steps=10,
        dropout=0.5,
    ),
    "gbce": dict(loss="gbce", num_negatives=16, warmup_steps=0, dropout=0.5),
}
out = {}
for name, r in recipes.items():
    losses = []
    orig = tt.step_loss

    def rec(*a, **k):
        l = orig(*a, **k)
        losses.append(float(l.detach()))
        return l

    tt.step_loss = rec
    tt.evaluate = lambda *a, **k: {"ndcg@10": 0.0}
    with tempfile.TemporaryDirectory() as ck:
        tt.train(
            TrainConfig(
                data_dir=str(root),
                checkpoint_dir=ck,
                device="cpu",
                compile=False,
                num_epochs=2,
                batch_size=32,
                max_seq_length=20,
                embedding_dim=16,
                ffn_hidden_dim=32,
                wandb_enabled=False,
                **r,
            )
        )
    tt.step_loss = orig
    out[name] = losses
print(tt.__file__)
print(json.dumps(out))
