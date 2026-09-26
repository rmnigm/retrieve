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


def load_sequences(
    parquet_path: str, max_length: int, use_time: bool, device: torch.device
) -> dict[str, torch.Tensor]:
    """``items [U, L+1]`` (and ``timestamps [U, L+1]`` under ``use_time``), left-padded with 0,
    each row the last ``L+1`` events of a user."""
    df = pl.read_parquet(parquet_path)
    if use_time and "timestamps" not in df.columns:
        raise ValueError(f"use_time=True but {parquet_path} has no 'timestamps' column")
    out = {"items": _left_pad(df["item_ids"], max_length + 1)}
    if use_time:
        out["timestamps"] = _left_pad(df["timestamps"], max_length + 1)
    return {k: v.to(device) for k, v in out.items()}


def train_batches(
    tensors: dict[str, torch.Tensor], batch_size: int
) -> Iterator[dict[str, torch.Tensor]]:
    """Shuffled full batches (the last partial one is dropped), drawn with the device RNG."""
    n = tensors["items"].shape[0]
    perm = torch.randperm(n, device=tensors["items"].device)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        yield {k: v[idx] for k, v in tensors.items()}
