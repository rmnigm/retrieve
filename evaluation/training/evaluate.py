"""Per-epoch retrieval-quality evaluation.

Plain torch path: scores the model's last-position query against the full item
catalog, masks history (and padding), takes top-K, and computes
NDCG/Recall/Coverage at the requested cutoffs. No ``retrieve`` layers, and its
own recall / ndcg (plan V D5): a checkpoint's reported quality must not move
when the harness's metric code does. ``tests/training/test_encode.py`` pins the
two to agree to 1e-9.
"""

from __future__ import annotations

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def hits_at(ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` bool: candidate at rank r is one of the row's non-``-1`` targets."""
    eq = ids.unsqueeze(2) == targets.unsqueeze(1)
    valid = (ids.unsqueeze(2) != -1) & (targets.unsqueeze(1) != -1)
    return (eq & valid).any(dim=2)


def recall_at_k(hits: torch.Tensor, num_targets: torch.Tensor, k: int) -> torch.Tensor:
    return hits[:, :k].sum(dim=1).float() / num_targets.float().clamp(min=1)


def ndcg_at_k(hits: torch.Tensor, num_targets: torch.Tensor, k: int) -> torch.Tensor:
    h = hits[:, :k].float()
    positions = torch.arange(1, k + 1, device=h.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg = (h * discounts.unsqueeze(0)).sum(dim=1)
    ideal = positions.unsqueeze(0) <= num_targets.unsqueeze(1).float().clamp(max=k)
    idcg = (ideal.float() * discounts.unsqueeze(0)).sum(dim=1)
    return dcg / idcg.clamp(min=1e-8)


class EvalDataset(Dataset):
    """History (and, under ``use_time``, its timestamps) left-padded to ``max_length``, plus
    the row's target ids."""

    def __init__(self, parquet_path: str, max_length: int, use_time: bool = False) -> None:
        df = pl.read_parquet(parquet_path)
        if use_time and "timestamps" not in df.columns:
            raise ValueError(f"use_time=True but {parquet_path} has no 'timestamps' column")
        self.sequences = df["item_ids"].to_list()
        self.timestamps = df["timestamps"].to_list() if use_time else None
        self.targets = df["targets"].to_list()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def _pad(self, seq: list[int]) -> torch.Tensor:
        seq = seq[-self.max_length :]
        return torch.tensor([0] * (self.max_length - len(seq)) + seq, dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, list[int]]:
        ts = None if self.timestamps is None else self._pad(self.timestamps[idx])
        return self._pad(self.sequences[idx]), ts, self.targets[idx]


def collate_eval(
    batch: list[tuple[torch.Tensor, torch.Tensor | None, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    seqs, times, tlists = zip(*batch, strict=True)
    item_seqs = torch.stack(seqs, dim=0)
    timestamps = None if times[0] is None else torch.stack(times, dim=0)
    num_targets = torch.tensor([len(t) for t in tlists], dtype=torch.long)
    max_t = int(num_targets.max().item()) if len(tlists) > 0 else 0
    targets = torch.full((len(tlists), max(max_t, 1)), -1, dtype=torch.long)
    for i, t in enumerate(tlists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return item_seqs, timestamps, targets, num_targets


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
    num_workers: int = 4,
    use_amp: bool = True,
    max_users: int | None = None,
    score_chunk: int = 262_144,
) -> dict[str, float]:
    dev = torch.device(device)
    was_training = model.training
    model.eval()

    dataset = EvalDataset(parquet_path, max_length=max_length, use_time=model.use_time)
    if max_users is not None and max_users < len(dataset):
        # Deterministic prefix; the parquet rows are already in user-id order so
        # this is reproducible across calls without needing a generator.
        dataset = torch.utils.data.Subset(dataset, range(max_users))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    item_embs = model.scoring_table().detach()  # [N+1, D]
    n_total = item_embs.shape[0]
    chunk = min(max(score_chunk, 1), n_total)
    coverage_seen = {k: torch.zeros(num_items + 1, dtype=torch.bool, device=dev) for k in ks}
    sums = {
        f"{m}@{k}": torch.zeros((), dtype=torch.float64, device=dev)
        for k in ks
        for m in ("ndcg", "recall")
    }
    n_rows = 0
    k_max = min(max(ks), num_items)
    amp_enabled = use_amp and dev.type == "cuda"

    for item_seqs, timestamps, targets, num_targets in loader:
        item_seqs = item_seqs.to(dev, non_blocking=True)
        if timestamps is not None:
            timestamps = timestamps.to(dev, non_blocking=True)
        targets = targets.to(dev, non_blocking=True)
        num_targets = num_targets.to(dev, non_blocking=True)

        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_enabled):
            query = model.predict_last(item_seqs, timestamps)  # [B, D]
        query = query.float()

        # Chunked scoring: keeps the [B, N+1] score matrix from materializing
        # all at once for large catalogs (Yambda-5B has 9.39M items).
        b = query.shape[0]
        topk_vals = torch.full((b, k_max), float("-inf"), device=dev)
        topk_idx = torch.zeros((b, k_max), dtype=torch.long, device=dev)
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            scores_chunk = query @ item_embs[start:end].T.float()  # [B, chunk]
            if start == 0:
                scores_chunk[:, 0] = float("-inf")  # padding row
            if mask_history:
                rel = item_seqs - start
                in_chunk = (rel >= 0) & (rel < (end - start))
                if in_chunk.any():
                    rows = torch.arange(b, device=dev).unsqueeze(1).expand_as(rel)[in_chunk]
                    cols = rel[in_chunk]
                    scores_chunk[rows, cols] = float("-inf")
            kk = min(k_max, end - start)
            chunk_vals, chunk_idx = scores_chunk.topk(kk, dim=1)
            chunk_idx = chunk_idx + start
            cat_vals = torch.cat([topk_vals, chunk_vals], dim=1)
            cat_idx = torch.cat([topk_idx, chunk_idx], dim=1)
            sel = cat_vals.topk(k_max, dim=1)
            topk_vals = sel.values
            topk_idx = cat_idx.gather(1, sel.indices)

        topk = topk_idx
        hits = hits_at(topk, targets)
        for k in ks:
            sums[f"ndcg@{k}"] += ndcg_at_k(hits, num_targets, k).double().sum()
            sums[f"recall@{k}"] += recall_at_k(hits, num_targets, k).double().sum()
            coverage_seen[k].scatter_(0, topk[:, :k].reshape(-1), True)
        n_rows += b

    out: dict[str, float] = {key: (v / max(n_rows, 1)).item() for key, v in sums.items()}
    for k in ks:
        out[f"coverage@{k}"] = float(coverage_seen[k][1:].sum().item()) / max(num_items, 1)

    if was_training:
        model.train()
    return out
