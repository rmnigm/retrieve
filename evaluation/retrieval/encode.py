"""Model load + query encoding for checkpoint-driven eval runs.

The one place the eval harness is allowed to import ``training.*`` — the
retrieval → training coupling lives entirely in this file. Yambda /
goodreads configs point at a SASRec checkpoint; ``load_model_for_eval``
rebuilds the model and ``encode_queries`` runs ``predict_last`` over the
eval split once (``queries_cache.py`` persists the result so per-algo
subprocesses don't re-encode). Pre-encoded datasets (arxiv) never call
into this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.evaluate import EvalDataset, collate_eval
from training.model import GSASRec

# Fallback hyperparams for the legacy 500M ckpts that don't ship a config.json
# alongside the .pt — they all happen to share these.
D128_DROP05_DEFAULTS = {
    "max_seq_length": 200,
    "embedding_dim": 128,
    "num_heads": 2,
    "num_blocks": 2,
    "ffn_hidden_dim": 512,
    "dropout": 0.5,
    "reuse_item_embeddings": False,
}


def load_model_for_eval(
    checkpoint_path: Path, num_items: int, device: torch.device
) -> GSASRec:
    """Load a `GSASRec` from disk for retrieval eval.

    Reads the sibling ``config.json`` when present (5B / freshly-trained
    ckpts have it via ``GSASRecConfig.save``); falls back to the
    d128-drop0.5 hyperparams for the legacy 500M ckpts that don't ship one.
    """
    cfg_path = checkpoint_path.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        params = {k: cfg[k] for k in D128_DROP05_DEFAULTS if k in cfg}
    else:
        params = dict(D128_DROP05_DEFAULTS)
    logger.info("model: GSASRec params={}", params)
    model = GSASRec(num_items=num_items, **params).to(device).eval()
    state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
    model.load_state_dict(state)
    return model


@torch.inference_mode()
def encode_queries(
    model: nn.Module,
    data_path: Path,
    *,
    max_length: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``model.predict_last`` over the entire eval split once.

    Returns ``(queries [N, D], targets [N, T_max], num_targets [N])`` on CPU.
    Encoding batch size is independent of the per-algo perf batch.
    """
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
        # Match training/evaluate.py: fp32 attention NaNs out on left-padded
        # rows where the first positions are fully masked; autocast dispatches
        # to a SDPA kernel that handles the all-masked-keys case.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            q = model.predict_last(item_seqs)
        q = q.float().detach().cpu()
        q_chunks.append(q)
        for row, n in zip(targets, num_targets, strict=True):
            t_lists.append(row[:n].tolist())

    queries = torch.cat(q_chunks, dim=0)
    n_users = queries.shape[0]
    n_targets = torch.tensor([len(t) for t in t_lists], dtype=torch.long)
    t_max = max(int(n_targets.max().item()), 1)
    targets = torch.full((n_users, t_max), -1, dtype=torch.long)
    for i, t in enumerate(t_lists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return queries, targets, n_targets


__all__ = [
    "D128_DROP05_DEFAULTS",
    "encode_queries",
    "load_model_for_eval",
]
