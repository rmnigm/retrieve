from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import torch.nn as nn
from loguru import logger
from torch.export import Dim, export, save

from retrieve import (
    FullScanKNN,
    build_ivf_int8,
    build_linr_index,
    build_silvertorch,
)
from training.config import GSASRecConfig
from training.model import GSASRec


class EncoderWrapper(nn.Module):
    def __init__(self, encoder: GSASRec) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        return self.encoder.predict_last(item_seq)


class IndexWrapper(nn.Module):
    def __init__(self, index: nn.Module) -> None:
        super().__init__()
        self.index = index

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.index(query)


class AttrIndexWrapper(nn.Module):
    def __init__(self, index: nn.Module) -> None:
        super().__init__()
        self.index = index

    def forward(
        self,
        query: torch.Tensor,
        query_clause_attrs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.index(query, query_clause_attrs)


ATTR_AWARE = {"silvertorch"}
SUPPORTS_CLAUSES = {"linr_v1", "linr_v2", "linr_v3", "ivf_int8", "silvertorch"}


def _build_index(
    name: str,
    item_embs: torch.Tensor,
    k: int,
    *,
    item_clause_attrs: torch.Tensor | None = None,
    n_lists: int = 256,
    n_probe: int = 16,
    m_bits: int = 1024,
    k_hash: int = 5,
) -> nn.Module:
    if name == "fullscan":
        idx = FullScanKNN(k=k)
        idx.register_index(item_embs)
        return idx
    if name == "ivf_int8":
        return build_ivf_int8(
            item_embs,
            k=k,
            n_lists=n_lists,
            n_probe=n_probe,
            item_clause_attrs=item_clause_attrs,
        )
    if name == "silvertorch":
        if item_clause_attrs is None:
            raise ValueError("silvertorch requires item_clause_attrs.")
        return build_silvertorch(
            item_embs,
            k=k,
            n_lists=n_lists,
            n_probe=n_probe,
            m_bits=m_bits,
            k_hash=k_hash,
            item_clause_attrs=item_clause_attrs,
        )
    version = {"linr_v1": 1, "linr_v2": 2, "linr_v3": 3}[name]
    return build_linr_index(
        version=version,
        item_embs=item_embs,
        k=k,
        item_clause_attrs=item_clause_attrs,
    )


@click.command()
@click.option("--checkpoint-dir", type=str, required=True)
@click.option(
    "--index",
    type=click.Choice(
        ["fullscan", "linr_v1", "linr_v2", "linr_v3", "ivf_int8", "silvertorch"]
    ),
    default="fullscan",
)
@click.option("--k", type=int, default=100)
@click.option("--device", type=str, default="cpu")
@click.option(
    "--attrs-path",
    type=str,
    default=None,
    help="Path to item_attrs.pt; required for silvertorch and enables clauses elsewhere.",
)
@click.option("--n-lists", type=int, default=256)
@click.option("--n-probe", type=int, default=16)
@click.option("--m-bits", type=int, default=1024)
@click.option("--k-hash", type=int, default=5)
def main(
    checkpoint_dir: str,
    index: str,
    k: int,
    device: str,
    attrs_path: str | None,
    n_lists: int,
    n_probe: int,
    m_bits: int,
    k_hash: int,
) -> None:
    ckpt = Path(checkpoint_dir)
    config = GSASRecConfig.load(ckpt / "config.json")
    with open(ckpt / "stats.json") as f:
        num_items = json.load(f)["num_items"]

    encoder = GSASRec(
        num_items=num_items,
        max_seq_length=config.max_seq_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        ffn_hidden_dim=config.ffn_hidden_dim,
        dropout=0.0,
        reuse_item_embeddings=config.reuse_item_embeddings,
    )
    encoder.load_state_dict(
        torch.load(ckpt / "best_model.pt", map_location="cpu", weights_only=True)
    )
    encoder.eval().to(device)

    item_embs = torch.load(ckpt / "item_embs.pt", map_location=device, weights_only=True)

    item_attrs: torch.Tensor | None = None
    if attrs_path is not None and index in SUPPORTS_CLAUSES:
        item_attrs = torch.load(attrs_path, map_location=device, weights_only=True)
        if item_attrs.shape[0] != item_embs.shape[0]:
            raise ValueError(
                "item_attrs first dim ({}) must match item_embs ({})".format(
                    item_attrs.shape[0], item_embs.shape[0]
                )
            )
        logger.info(
            "Loaded item_attrs from {}: shape={} ", attrs_path, tuple(item_attrs.shape)
        )

    enc_wrapper = EncoderWrapper(encoder).to(device).eval()
    example_seq = torch.zeros(2, config.max_seq_length, dtype=torch.long, device=device)
    example_seq[:, -1] = 1
    batch = Dim("batch", min=1, max=32)
    enc_exp = export(
        enc_wrapper, (example_seq,), dynamic_shapes={"item_seq": {0: batch}}
    )
    save(enc_exp, ckpt / "encoder.pt2")
    logger.info("Saved encoder.pt2")

    idx = _build_index(
        index,
        item_embs,
        k,
        item_clause_attrs=item_attrs,
        n_lists=n_lists,
        n_probe=n_probe,
        m_bits=m_bits,
        k_hash=k_hash,
    ).to(device).eval()

    example_query = torch.zeros(2, config.embedding_dim, device=device)
    if index in ATTR_AWARE:
        c_dim = item_attrs.shape[1] if item_attrs is not None else 1
        example_attrs = torch.full(
            (2, c_dim), -1, dtype=torch.long, device=device
        )
        wrapped = AttrIndexWrapper(idx).to(device).eval()
        idx_exp = export(
            wrapped,
            (example_query, example_attrs),
            dynamic_shapes={"query": {0: batch}, "query_clause_attrs": {0: batch}},
        )
    else:
        wrapped = IndexWrapper(idx).to(device).eval()
        idx_exp = export(wrapped, (example_query,), dynamic_shapes={"query": {0: batch}})
    save(idx_exp, ckpt / f"index_{index}.pt2")
    logger.info("Saved index_{}.pt2", index)

    with open(ckpt / "index_meta.json", "w") as f:
        json.dump(
            {
                "index": index,
                "k": k,
                "num_items": int(item_embs.shape[0]),
                "embedding_dim": int(item_embs.shape[1]),
                "dtype": str(item_embs.dtype),
                "attr_aware": index in ATTR_AWARE,
                "n_lists": n_lists if index in {"ivf_int8", "silvertorch"} else None,
                "n_probe": n_probe if index in {"ivf_int8", "silvertorch"} else None,
                "m_bits": m_bits if index == "silvertorch" else None,
                "k_hash": k_hash if index == "silvertorch" else None,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
