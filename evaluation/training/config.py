from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class TrainConfig:
    data_dir: str = "data/yambda/50m"
    max_seq_length: int = 200

    embedding_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    ffn_hidden_dim: int = 256
    dropout: float = 0.5
    reuse_item_embeddings: bool = False

    loss: str = "sampled_softmax"
    num_negatives: int = 8192
    inbatch_negatives: int = 4096
    gbce_t: float = 0.75
    temperature: float = 0.05
    normalize: bool | None = None
    logq: bool | None = None

    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    compile: bool = True
    num_epochs: int = 100
    max_batches_per_epoch: int | None = None
    patience: int = 10

    eval_batch_size: int = 1024
    eval_ks: tuple[int, ...] = (10, 100)
    eval_every: int = 2
    eval_max_users: int | None = None
    eval_score_chunk: int = 262_144
    mask_history: bool = False
    early_stop_metric: str = "ndcg@10"

    wandb_enabled: bool = True
    wandb_project: str = "seqrec-encoder"
    wandb_run_name: str | None = None
    log_every: int = 50

    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.loss not in ("gbce", "sampled_softmax"):
            raise ValueError(f"loss={self.loss!r}: expected 'gbce' or 'sampled_softmax'")
        if self.normalize is None:
            self.normalize = self.loss == "sampled_softmax"
        if self.logq is None:
            self.logq = self.loss == "sampled_softmax"
        if self.normalize and self.loss == "gbce":
            raise ValueError("normalize=True is only implemented for loss='sampled_softmax'")
        if self.logq and self.loss == "gbce":
            raise ValueError("logq=True is only implemented for loss='sampled_softmax'")
        self.eval_ks = tuple(self.eval_ks)

    @property
    def num_items(self) -> int:
        with open(Path(self.data_dir) / "item_id_map.json") as f:
            return len(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> TrainConfig:
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in dataclasses.fields(cls)}
        # A config.json without "loss" predates it: a published gBCE checkpoint.
        data = {"loss": "gbce", **{k: v for k, v in data.items() if k in valid}}
        # Saved normalize/logq are resolved for the saved loss; a new loss re-resolves them.
        if overrides.get("loss", data["loss"]) != data["loss"]:
            data = {k: v for k, v in data.items() if k not in ("normalize", "logq")}
        return cls(**{**data, **overrides})
