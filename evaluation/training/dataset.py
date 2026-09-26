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


def train_batches(items: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    """Shuffled full batches (the last partial one is dropped), drawn with the device RNG."""
    perm = torch.randperm(items.shape[0], device=items.device)
    for start in range(0, items.shape[0] - batch_size + 1, batch_size):
        yield items[perm[start : start + batch_size]]
