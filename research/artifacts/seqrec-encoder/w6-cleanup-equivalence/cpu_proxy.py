"""W6 CPU proxy of the equivalence gate: 2 epochs of train() on a synthetic 300-user,
500-item split, per-step losses for sampled softmax + logQ and for gBCE. Run once with the
pre-cleanup trainer first on PYTHONPATH (``git archive dev/hstu evaluation/training``) and
once without; the two JSON lines must be equal. Single-threaded: on 64 threads the gBCE
losses of one and the same code differ run to run in the last bit at a few steps.

    cd evaluation && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run python \
        ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/cpu_proxy.py SYNTH_DIR
"""

import json
import random
import sys
import tempfile
from pathlib import Path

import polars as pl

import training.train as tt
from training.config import TrainConfig

root = Path(sys.argv[1])
root.mkdir(exist_ok=True)
rng = random.Random(0)
seqs = [[rng.randint(1, 500) for _ in range(rng.randint(2, 40))] for _ in range(300)]
pl.DataFrame({"item_ids": seqs}).write_parquet(root / "train.parquet")
(root / "item_id_map.json").write_text(json.dumps({str(i): i for i in range(1, 501)}))
RECIPES = {
    "ssm": {"loss": "sampled_softmax", "normalize": True, "logq": True, "num_negatives": 64,
            "inbatch_negatives": 32, "warmup_steps": 10},
    "gbce": {"loss": "gbce", "num_negatives": 16, "warmup_steps": 0},
}  # fmt: skip
SMALL = {"device": "cpu", "compile": False, "num_epochs": 2, "batch_size": 32, "dropout": 0.5,
         "max_seq_length": 20, "embedding_dim": 16, "ffn_hidden_dim": 32, "wandb_enabled": False}  # fmt: skip

step_loss = tt.step_loss
tt.evaluate = lambda *a, **k: {"ndcg@10": 0.0}
out = {}
for name, recipe in RECIPES.items():
    losses = out[name] = []

    def recording_step_loss(*args, _losses=losses, **kwargs):
        loss = step_loss(*args, **kwargs)
        _losses.append(float(loss.detach()))
        return loss

    tt.step_loss = recording_step_loss
    with tempfile.TemporaryDirectory() as ckpt:
        tt.train(
            TrainConfig(data_dir=str(root), checkpoint_dir=ckpt, **SMALL, **recipe)
        )
print(tt.__file__)
print(json.dumps(out))
