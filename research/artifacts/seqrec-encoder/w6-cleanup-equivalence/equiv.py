"""W6 equivalence: 50 training steps (1 epoch, compile off, seed 42) on yambda-500m d64, the
per-step losses dumped as Python floats. RECIPE is `e1c` (the E1c config.json, or with
`defaults` the bare TrainConfig defaults) or `gbce` (the Gate B config.json). Val/test eval is
stubbed: only the training losses are compared.

    cd evaluation && uv run python ../docs/artifacts/seqrec-encoder/w6-cleanup-equivalence/equiv.py \
        {e1c|e1c-defaults|gbce} OUT.json
"""

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import training.train as tt
from training.config import TrainConfig

ART = Path(__file__).resolve().parent.parent
RUN = {
    "compile": False,
    "max_batches_per_epoch": 50,
    "num_epochs": 1,
    "wandb_enabled": False,
}
recipe, out = sys.argv[1], Path(sys.argv[2])
if recipe == "e1c-defaults":
    base = {"data_dir": "/data/yambda-500m/trainer"}
else:
    src = {"e1c": "e1c-yambda-d64-sasrec-ssm-logq", "gbce": "gate-b-yambda-d64"}[recipe]
    base = dataclasses.asdict(TrainConfig.load(ART / src / "config.json"))

losses = []
step_loss = tt.step_loss


def recording_step_loss(*args, **kwargs):
    loss = step_loss(*args, **kwargs)
    losses.append(loss.detach())
    return loss


tt.step_loss = recording_step_loss
tt.evaluate = lambda *a, **k: {"ndcg@10": 0.0}
with tempfile.TemporaryDirectory() as ckpt:
    config = TrainConfig(**{**base, **RUN, "checkpoint_dir": ckpt})
    tt.train(config)
values = [float(v) for v in losses]
out.write_text(json.dumps({"recipe": recipe, "module": tt.__file__,
                           "config": dataclasses.asdict(config) | {"checkpoint_dir": None},
                           "losses": values}, indent=1) + "\n")  # fmt: skip
print(recipe, len(values), values[0], values[-1])
