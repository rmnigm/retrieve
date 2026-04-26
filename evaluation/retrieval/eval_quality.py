from __future__ import annotations

import json
from pathlib import Path

import click
import polars as pl
import torch
from loguru import logger
from torch.export import load
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from retrieval.metrics import accumulate_metrics, finalize_metrics


class EvalDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        max_length: int,
        padding_value: int = 0,
        item_attrs: torch.Tensor | None = None,
    ) -> None:
        df = pl.read_parquet(parquet_path)
        self.sequences = df["item_ids"].to_list()
        self.targets = df["targets"].to_list()
        self.max_length = max_length
        self.padding_value = padding_value
        self.item_attrs = item_attrs

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, list[int], torch.Tensor | None]:
        seq = self.sequences[idx]
        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]
        if len(seq) < self.max_length:
            seq = [self.padding_value] * (self.max_length - len(seq)) + seq
        targets = self.targets[idx]
        query_attrs = None
        if self.item_attrs is not None and len(targets) > 0:
            # Filter by first target's attrs (category + brand), taking attribute 0.
            target_id = targets[0]
            # item_attrs: [N, C, A_max] with -1 pad. Pull first non-pad value per clause.
            target_attrs = self.item_attrs[target_id]  # [C, A_max]
            c = target_attrs.shape[0]
            query_attrs = torch.full((c,), -1, dtype=torch.long)
            for ci in range(c):
                clause_vals = target_attrs[ci]
                positive = clause_vals[clause_vals >= 0]
                if positive.numel() > 0:
                    query_attrs[ci] = positive[0]
        return torch.tensor(seq, dtype=torch.long), targets, query_attrs


def collate_eval(
    batch: list[tuple[torch.Tensor, list[int], torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    seqs, tlists, qattrs = zip(*batch, strict=True)
    item_seqs = torch.stack(seqs, dim=0)
    num_targets = torch.tensor([len(t) for t in tlists], dtype=torch.long)
    max_t = int(num_targets.max().item())
    targets = torch.full((len(tlists), max_t), fill_value=-1, dtype=torch.long)
    for i, t in enumerate(tlists):
        targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    query_attrs = None
    if qattrs[0] is not None:
        query_attrs = torch.stack(list(qattrs), dim=0)
    return item_seqs, targets, num_targets, query_attrs


@click.command()
@click.option("--checkpoint-dir", type=str, required=True)
@click.option("--index", type=str, default="fullscan")
@click.option("--data-dir", type=str, default=None)
@click.option("--split", type=click.Choice(["test", "val"]), default="test")
@click.option("--batch-size", type=int, default=256)
@click.option("--max-seq-length", type=int, default=None)
@click.option("--k", type=int, multiple=True, default=[10, 50, 100])
@click.option("--device", type=str, default="cuda")
@click.option(
    "--attrs-path",
    type=str,
    default=None,
    help="Path to item_attrs.pt. When set, per-query filters are synthesized from target attrs.",
)
@click.option(
    "--use-attrs/--no-use-attrs",
    default=False,
    help="Pass query_clause_attrs into the exported index (required for silvertorch).",
)
def main(
    checkpoint_dir: str,
    index: str,
    data_dir: str | None,
    split: str,
    batch_size: int,
    max_seq_length: int | None,
    k: tuple[int, ...],
    device: str,
    attrs_path: str | None,
    use_attrs: bool,
) -> None:
    ckpt = Path(checkpoint_dir)
    ks = sorted(k)

    with open(ckpt / "config.json") as f:
        cfg = json.load(f)
    max_len = max_seq_length or cfg["max_seq_length"]
    data = Path(data_dir or cfg["data_dir"]) / f"{split}.parquet"

    encoder = load(ckpt / "encoder.pt2").module().to(device).eval()
    idx = load(ckpt / f"index_{index}.pt2").module().to(device).eval()

    item_attrs: torch.Tensor | None = None
    if attrs_path is not None:
        item_attrs = torch.load(attrs_path, map_location="cpu", weights_only=True)

    dataset = EvalDataset(str(data), max_length=max_len, item_attrs=item_attrs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
    )

    accum = None
    dev = torch.device(device)
    with torch.inference_mode():
        for item_seqs, targets, num_targets, query_attrs in tqdm(loader, desc="Eval"):
            item_seqs = item_seqs.to(dev)
            targets = targets.to(dev)
            num_targets = num_targets.to(dev)
            query = encoder(item_seqs)
            if use_attrs:
                if query_attrs is None:
                    raise RuntimeError(
                        "use_attrs=True requires --attrs-path to be set."
                    )
                query_attrs = query_attrs.to(dev)
                candidate_ids, _ = idx(query, query_attrs)
            else:
                candidate_ids, _ = idx(query)
            accum = accumulate_metrics(candidate_ids, targets, num_targets, ks, accum)

    results = finalize_metrics(accum)
    logger.info("Results: {}", json.dumps(results, indent=2))

    out = {
        "split": split,
        "ks": ks,
        "index": index,
        "use_attrs": use_attrs,
        "metrics": results,
    }
    suffix = "_attrs" if use_attrs else ""
    with open(ckpt / f"eval_quality{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
