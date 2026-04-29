from __future__ import annotations

from functools import partial

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        max_length: int = 50,
        padding_value: int = 0,
    ):
        df = pl.read_parquet(parquet_path)
        self.sequences = df["item_ids"].to_list()
        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        if len(seq) > self.max_length + 1:
            seq = seq[-(self.max_length + 1):]
        if len(seq) < self.max_length + 1:
            pad = [self.padding_value] * (self.max_length + 1 - len(seq))
            seq = pad + seq
        return torch.tensor(seq, dtype=torch.long)


def collate_train_with_negatives(
    batch: list[torch.Tensor],
    num_items: int,
    num_negatives: int,
    shared_batch_negatives: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stacked = torch.stack(batch, dim=0)
    input_seq = stacked[:, :-1]
    target_seq = stacked[:, 1:]
    if shared_batch_negatives:
        # One [K] sample per batch, broadcast across all (B, L) positions in gbce_loss.
        # Touches B*L*K -> K unique negatives per step → ~150× fewer rows in the
        # embedding-table grad on Yambda-5B (9.39M items).
        negatives = torch.randint(low=1, high=num_items + 1, size=(num_negatives,))
    else:
        negatives = torch.randint(
            low=1,
            high=num_items + 1,
            size=(input_seq.size(0), input_seq.size(1), num_negatives),
        )
    return input_seq, target_seq, negatives


def get_train_dataloader(
    parquet_path: str,
    batch_size: int,
    max_length: int,
    num_items: int,
    negs_per_pos: int,
    shared_batch_negatives: bool = False,
) -> DataLoader:
    dataset = SequenceDataset(parquet_path, max_length=max_length)
    collate_fn = partial(
        collate_train_with_negatives,
        num_items=num_items,
        num_negatives=negs_per_pos,
        shared_batch_negatives=shared_batch_negatives,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
