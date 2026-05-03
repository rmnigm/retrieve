from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class GSASRecConfig:
    data_dir: str = "data/yambda/50m"
    max_seq_length: int = 200

    embedding_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 2
    ffn_hidden_dim: int = 256
    dropout: float = 0.0
    reuse_item_embeddings: bool = False

    negs_per_pos: int = 256
    gbce_t: float = 0.75
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 200
    max_batches_per_epoch: int | None = None
    patience: int = 20

    eval_batch_size: int = 512
    eval_ks: tuple[int, ...] = (10, 100)
    eval_every: int = 1
    eval_max_users: int | None = None
    eval_score_chunk: int = 262_144
    mask_history: bool = False
    early_stop_metric: str = "ndcg@10"

    wandb_enabled: bool = True
    wandb_project: str = "yambda-gsasrec"
    wandb_run_name: str | None = None
    log_every: int = 50

    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42

    text_embedding_path: str | None = None
    tie_content_output: bool = False
    content_proj_type: str = "linear"

    @property
    def num_items(self) -> int:
        with open(Path(self.data_dir) / "item_id_map.json") as f:
            return len(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GSASRecConfig:
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
