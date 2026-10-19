"""The sequence ``Encoder`` (item table → SASRec blocks → final norm), the gSASRec body;
docs/system/datasets.md § Training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import TrainConfig


class SASRecBlock(nn.TransformerEncoderLayer):
    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float):
        super().__init__(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, attn_mask):
        b, _, length, _ = attn_mask.shape
        blocked = (~attn_mask).expand(b, self.self_attn.num_heads, length, length)
        return super().forward(x, src_mask=blocked.reshape(-1, length, length))


class Encoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_length: int = 200,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.0,
        reuse_item_embeddings: bool = False,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_embedding = (
            None if reuse_item_embeddings else nn.Embedding(num_items + 1, embedding_dim)
        )
        self._init_weights(initializer_range=0.02)

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        for key, value in self.named_parameters():
            if "weight" in key:
                if "norm" in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range,
                    )
            elif "bias" in key:
                nn.init.zeros_(value.data)

    def get_output_embeddings(self) -> nn.Embedding:
        if self.output_embedding is not None:
            return self.output_embedding
        return self.item_embedding

    def scoring_table(self) -> torch.Tensor:
        """``[N+1, D]`` item vectors that queries are scored against (unit rows under
        ``normalize``)."""
        table = self.get_output_embeddings().weight
        return F.normalize(table, dim=-1) if self.normalize else table

    def body(self, x, attn_mask):
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.final_norm(x)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        length = items.shape[1]
        key_valid = items != 0
        eye = torch.eye(length, dtype=torch.bool, device=items.device)
        causal = torch.ones(length, length, dtype=torch.bool, device=items.device).tril()
        # The diagonal keeps every left-padding row non-empty, so no row softmaxes to NaN.
        attn_mask = causal & (key_valid[:, None, None, :] | eye)
        x = self.item_embedding(items) + self.position_embedding.weight[:length]
        return self.body(self.embedding_dropout(x), attn_mask)

    def predict_last(self, items: torch.Tensor) -> torch.Tensor:
        # Left-padded: the last real token is always at index -1.
        q = self.forward(items)[:, -1]
        return F.normalize(q.float(), dim=-1) if self.normalize else q


def build_encoder(cfg: TrainConfig, num_items: int) -> Encoder:
    return Encoder(
        num_items=num_items,
        max_seq_length=cfg.max_seq_length,
        embedding_dim=cfg.embedding_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        dropout=cfg.dropout,
        reuse_item_embeddings=cfg.reuse_item_embeddings,
        normalize=cfg.normalize,
    )
