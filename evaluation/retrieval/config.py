"""YAML config for the yambda retrieval benchmark.

Boring dataclass + ``yaml.safe_load``. No env-var interpolation, no schema
versioning, no validation framework — bad keys raise ``TypeError`` from the
dataclass constructor with a useful message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncodeConfig:
    batch_size: int = 512
    num_workers: int = 8
    max_seq_length: int = 200


@dataclass
class EvalConfig:
    checkpoint: str
    data_dir: str
    output: str | None = None
    split: str = "test"
    device: str = "cuda"
    ks: list[int] = field(default_factory=lambda: [100, 500])
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    algorithms: list[str] = field(default_factory=list)
    algo_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_eval_config(path: Path) -> EvalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    encode = EncodeConfig(**(raw.pop("encode", None) or {}))
    return EvalConfig(encode=encode, **raw)
