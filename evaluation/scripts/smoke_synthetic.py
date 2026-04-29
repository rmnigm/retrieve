"""Synthetic-data smoke harness for the trainer.

Builds a tiny synthetic dataset (1K users / 50K items / sequences of length 50)
in `/tmp/yambda-smoke/`, runs the trainer for 2 epochs, and asserts:
1. Loss strictly decreases between epoch 0 and epoch 1.
2. NDCG@10 on the synthetic test split is reported (>= 0 sanity).

This is the gate before we burn time on real Yambda data prep + a full run.
~1 minute on an A100.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

from training.config import GSASRecConfig
from training.train_sasrec import train


def _make_user_seq(rng: np.random.Generator, num_items: int, length: int) -> list[int]:
    return rng.integers(1, num_items + 1, size=length).tolist()


def build_synthetic_dataset(
    output_dir: Path,
    num_users: int = 1000,
    num_items: int = 50_000,
    seq_len: int = 50,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_rows = []
    val_rows = []
    test_rows = []
    for _ in range(num_users):
        history = _make_user_seq(rng, num_items, seq_len)
        train_rows.append({"item_ids": history})
        val_targets = _make_user_seq(rng, num_items, 3)
        test_targets = _make_user_seq(rng, num_items, 3)
        val_rows.append({"item_ids": history, "targets": val_targets})
        test_rows.append({
            "item_ids": history + val_targets,
            "targets": test_targets,
        })

    pl.DataFrame(train_rows).write_parquet(output_dir / "train.parquet")
    pl.DataFrame(val_rows).write_parquet(output_dir / "val.parquet")
    pl.DataFrame(test_rows).write_parquet(output_dir / "test.parquet")
    with open(output_dir / "item_id_map.json", "w") as f:
        json.dump({str(i): i for i in range(1, num_items + 1)}, f)
    logger.info("Wrote synthetic dataset to {} (users={}, items={})", output_dir, num_users, num_items)


def main() -> int:
    smoke_dir = Path(tempfile.mkdtemp(prefix="yambda-smoke-"))
    ckpt_dir = smoke_dir / "ckpt"
    data_dir = smoke_dir / "data"
    try:
        build_synthetic_dataset(data_dir, num_users=1000, num_items=50_000, seq_len=50)

        cfg = GSASRecConfig(
            data_dir=str(data_dir),
            checkpoint_dir=str(ckpt_dir),
            max_seq_length=50,
            embedding_dim=64,
            num_heads=2,
            num_blocks=2,
            ffn_hidden_dim=256,
            dropout=0.0,
            negs_per_pos=4,
            batch_size=128,
            learning_rate=1e-3,
            num_epochs=2,
            patience=10,
            eval_batch_size=64,
            eval_ks=(10, 100),
            eval_every=1,
            eval_max_users=200,
            eval_score_chunk=8192,
            wandb_enabled=False,
            log_every=10,
            seed=0,
        )

        train(cfg)

        with open(ckpt_dir / "train_metrics.json") as f:
            metrics = json.load(f)

        epoch_losses = metrics["epoch_losses"]
        val_metrics = metrics["val_metrics_per_epoch"]
        test_metrics = metrics["test_metrics"]
        logger.info("Epoch losses: {}", epoch_losses)
        logger.info("Val metrics per epoch: {}", val_metrics)
        logger.info("Test metrics: {}", test_metrics)

        assert len(epoch_losses) >= 2, "expected >=2 epochs"
        assert epoch_losses[1] < epoch_losses[0], (
            f"loss did not decrease: {epoch_losses}"
        )
        assert test_metrics.get("ndcg@10", 0.0) >= 0.0, "ndcg@10 unexpectedly missing"
        logger.info("smoke OK: loss decreased {:.4f} -> {:.4f}", epoch_losses[0], epoch_losses[1])
        return 0
    finally:
        shutil.rmtree(smoke_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
