"""Per-epoch retrieval-quality evaluation.

Plain torch path: scores the model's last-position query against the full item
catalog, masks history (and padding), takes top-K, and computes
NDCG/Recall/Coverage at the requested cutoffs. No `retrieve` framework.
"""

from __future__ import annotations

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from retrieval.metrics import accumulate_metrics, finalize_metrics


class EvalDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        max_length: int,
        padding_value: int = 0,
    ) -> None:
        df = pl.read_parquet(parquet_path)
        self.sequences = df["item_ids"].to_list()
        self.targets = df["targets"].to_list()
        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        seq = self.sequences[idx]
        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]
        if len(seq) < self.max_length:
            seq = [self.padding_value] * (self.max_length - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long), self.targets[idx]


def collate_eval(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, tlists = zip(*batch, strict=True)
    item_seqs = torch.stack(seqs, dim=0)
    num_targets = torch.tensor([len(t) for t in tlists], dtype=torch.long)
    max_t = int(num_targets.max().item()) if len(tlists) > 0 else 0
    targets = torch.full((len(tlists), max(max_t, 1)), -1, dtype=torch.long)
    for i, t in enumerate(tlists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return item_seqs, targets, num_targets


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    parquet_path: str,
    num_items: int,
    max_length: int,
    batch_size: int = 512,
    ks: tuple[int, ...] = (10, 100),
    device: str | torch.device = "cuda",
    mask_history: bool = False,
    num_workers: int = 8,
    use_amp: bool = True,
) -> dict[str, float]:
    dev = torch.device(device)
    was_training = model.training
    model.eval()

    dataset = EvalDataset(parquet_path, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    item_embs = model.get_output_embeddings().weight.detach()  # [N+1, D]
    coverage_seen = {
        k: torch.zeros(num_items + 1, dtype=torch.bool, device=dev) for k in ks
    }
    accum: dict[str, list[float]] | None = None
    k_max = min(max(ks), num_items)
    amp_enabled = use_amp and dev.type == "cuda"

    for item_seqs, targets, num_targets in loader:
        item_seqs = item_seqs.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        num_targets = num_targets.to(dev, non_blocking=True)

        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
            query = model.predict_last(item_seqs)  # [B, D]
            scores = query @ item_embs.T  # [B, N+1]
        scores = scores.float()
        scores[:, 0] = float("-inf")
        if mask_history:
            scores.scatter_(1, item_seqs, float("-inf"))

        _, topk = scores.topk(k_max, dim=1)  # [B, k_max]
        accum = accumulate_metrics(topk, targets, num_targets, list(ks), accum)
        for k in ks:
            coverage_seen[k].scatter_(0, topk[:, :k].reshape(-1), True)

    out_full = finalize_metrics(accum) if accum is not None else {}
    out: dict[str, float] = {
        key: val for key, val in out_full.items()
        if key.startswith(("ndcg@", "recall@"))
    }
    for k in ks:
        out[f"coverage@{k}"] = float(coverage_seen[k][1:].sum().item()) / max(num_items, 1)

    if was_training:
        model.train()
    return out
