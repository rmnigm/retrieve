"""Training sequences as pre-padded tensors: the whole split sits on the GPU and a batch is a
``randperm`` slice of it, so there is no DataLoader and no host-side negative sampling."""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import torch


def _left_pad(column: pl.Series, width: int) -> torch.Tensor:
    tails = column.list.tail(width)
    lengths = torch.from_numpy(tails.list.len().to_numpy().astype("int64"))
    flat = torch.from_numpy(tails.explode().to_numpy().astype("int64"))
    rows = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    starts = torch.cumsum(lengths, 0) - lengths
    cols = torch.arange(len(flat)) - starts[rows] + (width - lengths)[rows]
    out = torch.zeros(len(lengths), width, dtype=torch.long)
    out[rows, cols] = flat
    return out


def load_sequences(parquet_path: str, max_length: int, device: torch.device) -> torch.Tensor:
    """``items [U, L+1]``, left-padded with 0, each row the last ``L+1`` events of a user."""
    df = pl.read_parquet(parquet_path, columns=["item_ids"])
    return _left_pad(df["item_ids"], max_length + 1).to(device)


def load_val_transitions(
    parquet_path: str, max_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Val rows as train rows: ``items [U, L+1]``, the last ``L+1`` of ``item_ids ++ targets``,
    and ``first [U]``, the first target position whose target is a val-day item."""
    df = pl.read_parquet(parquet_path, columns=["item_ids", "targets"])
    items = _left_pad(df["item_ids"].list.concat(df["targets"]), max_length + 1)
    n_targets = torch.from_numpy(df["targets"].list.len().to_numpy().astype("int64"))
    return items.to(device), (max_length - n_targets).clamp(min=0).to(device)


def target_mask(items: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
    """``[U, L]`` bool: target position ``j`` is trained, a real item at ``j >= first``."""
    positions = torch.arange(items.shape[1] - 1, device=items.device)
    return (items[:, 1:] != 0) & (positions >= first.unsqueeze(1))


def train_batches(
    items: torch.Tensor, first: torch.Tensor, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Shuffled full batches (the last partial one is dropped), drawn with the device RNG."""
    perm = torch.randperm(items.shape[0], device=items.device)
    for start in range(0, items.shape[0] - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        yield items[idx], first[idx]
