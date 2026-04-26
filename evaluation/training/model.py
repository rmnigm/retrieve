from __future__ import annotations

import torch
import torch.nn as nn


class GSASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_length: int = 200,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_blocks: int = 3,
        ffn_hidden_dim: int = 512,
        dropout: float = 0.5,
        reuse_item_embeddings: bool = False,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.padding_idx = 0

        self.item_embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=self.padding_idx
        )
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.final_norm = nn.LayerNorm(embedding_dim)

        self.reuse_item_embeddings = reuse_item_embeddings
        if not reuse_item_embeddings:
            self.output_embedding = nn.Embedding(num_items + 1, embedding_dim)
        else:
            self.output_embedding = None

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

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        _, seq_len = item_seq.shape
        seq_emb = self.item_embedding(item_seq)
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        seq_emb = seq_emb + self.position_embedding(positions)
        seq_emb = self.embedding_dropout(seq_emb)

        padding_mask = item_seq == self.padding_idx
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=item_seq.device
        )
        hidden = self.encoder(
            seq_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )
        return self.final_norm(hidden)

    def predict_last(self, item_seq: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(item_seq)
        seq_lens = ((item_seq != self.padding_idx).sum(dim=1) - 1).clamp(min=0)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]
