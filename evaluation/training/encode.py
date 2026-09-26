"""History → query vector with a trained ``Encoder``: model load, ``predict_last`` over the eval
split, and the on-disk cache next to the checkpoint (``encoded_queries_v2.pt``, keyed on the
checkpoint's mtime + ``max_seq_length``, the *full* split — ``users_limit`` is applied by the
harness after loading). Yambda / goodreads configs point at a checkpoint; pre-encoded text
datasets never call into this module.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_datasets.layout import atomic_write
from training.config import TrainConfig
from training.evaluate import EvalDataset, collate_eval
from training.model import Encoder, build_encoder

ENCODE_CACHE = "encoded_queries_v2.pt"
_CACHE_FREE_FRACTION, _CACHE_RESERVE_BYTES = 0.7, 4 * 2**30
# The legacy 500M checkpoints ship no config.json; they all share these.
D128_DROP05_DEFAULTS = {
    "max_seq_length": 200,
    "embedding_dim": 128,
    "num_heads": 2,
    "num_blocks": 2,
    "ffn_hidden_dim": 512,
    "dropout": 0.5,
    "reuse_item_embeddings": False,
    "loss": "gbce",
}


def load_model_for_eval(checkpoint_path: Path, num_items: int, device: torch.device) -> Encoder:
    """An ``Encoder`` from ``checkpoint_path``, hyperparameters from the sibling ``config.json``
    when present, else ``D128_DROP05_DEFAULTS``. State dicts saved by the retired ``GSASRec``
    (``encoder.layers.N.``) load into ``blocks.N.``: the math is the same."""
    cfg_path = checkpoint_path.parent / "config.json"
    cfg = TrainConfig.load(cfg_path) if cfg_path.exists() else TrainConfig(**D128_DROP05_DEFAULTS)
    model = build_encoder(cfg, num_items).to(device).eval()
    state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
    model.load_state_dict({k.replace("encoder.layers.", "blocks."): v for k, v in state.items()})
    return model


@torch.inference_mode()
def encode_queries(
    model: Encoder,
    data_path: Path,
    *,
    max_length: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``model.predict_last`` over the whole eval split: ``(queries [N, D], targets [N, T_max]
    -1-padded, num_targets [N])`` on CPU, target ids as stored (1-indexed)."""
    dataset = EvalDataset(str(data_path), max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=encode_batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    q_chunks: list[torch.Tensor] = []
    t_lists: list[list[int]] = []
    amp_enabled = device.type == "cuda"
    for item_seqs, targets, num_targets in tqdm(loader, desc="encode queries"):
        item_seqs = item_seqs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            q = model.predict_last(item_seqs)
        q_chunks.append(q.float().detach().cpu())
        for row, n in zip(targets, num_targets, strict=True):
            t_lists.append(row[:n].tolist())

    queries = torch.cat(q_chunks, dim=0)
    n_targets = torch.tensor([len(t) for t in t_lists], dtype=torch.long)
    t_max = max(int(n_targets.max().item()), 1)
    targets = torch.full((queries.shape[0], t_max), -1, dtype=torch.long)
    for i, t in enumerate(t_lists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return queries, targets, n_targets


def encode_split(
    checkpoint: Path,
    data_dir: Path,
    *,
    max_seq_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(item_embs [N, D] on device, queries [U, D], targets [U, T] 0-indexed, n_targets [U])``
    for ``data_dir/test.parquet`` — from the cache when its key matches, else encoded and
    cached (atomically, within a free-disk budget). The training-side padding row (item id
    0) is dropped from ``item_embs`` and the target ids shifted ``-1`` to match."""
    cache = checkpoint.parent / ENCODE_CACHE
    key = {"ckpt_mtime": checkpoint.stat().st_mtime, "max_seq_length": max_seq_length}
    if cache.exists():
        blob = torch.load(str(cache), map_location="cpu", weights_only=True)
        if all(blob.get(k) == v for k, v in key.items()):
            logger.info("loaded encoded queries from {}", cache)
            item_embs = blob["item_embs"].to(device).contiguous()
            return item_embs, blob["queries"], blob["targets"], blob["n_targets"]
        logger.info("{}: stale (ckpt mtime / max_seq_length changed); re-encoding", cache)
    with open(data_dir / "item_id_map.json") as f:
        num_items = len(json.load(f))
    model = load_model_for_eval(checkpoint, num_items=num_items, device=device)
    item_embs = model.scoring_table().detach()[1:].to(device).contiguous()
    queries, targets, n_targets = encode_queries(
        model,
        data_dir / "test.parquet",
        max_length=max_seq_length,
        encode_batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    targets = torch.where(targets >= 0, targets - 1, targets)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    blob = {
        "item_embs": item_embs.cpu(),
        "queries": queries,
        "targets": targets,
        "n_targets": n_targets,
        **key,
    }
    est = sum(t.numel() * t.element_size() for t in blob.values() if torch.is_tensor(t))
    free = shutil.disk_usage(str(checkpoint.parent)).free
    budget = int(free * _CACHE_FREE_FRACTION) - _CACHE_RESERVE_BYTES
    if est > budget:
        logger.warning(
            "not caching encoded queries: {:.1f} GB > budget {:.1f} GB", est / 2**30, budget / 2**30
        )
    else:
        atomic_write(cache, lambda fh: torch.save(blob, fh))
        logger.info("wrote {} ({:.2f} GB)", cache, est / 2**30)
    return item_embs, queries, targets, n_targets


__all__ = [
    "D128_DROP05_DEFAULTS",
    "ENCODE_CACHE",
    "encode_queries",
    "encode_split",
    "load_model_for_eval",
]
