"""The one sequence ``Encoder`` (item table → blocks → final norm → out_proj) with two block
types, ``sasrec`` and ``hstu``; docs/system/datasets.md § Training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import TrainConfig

TIME_LOG_BASE = 1.5


def time_bucket(dt: torch.Tensor, buckets: int) -> torch.Tensor:
    """Log-scaled bucket of a non-negative time delta in seconds."""
    b = torch.log1p(dt.clamp(min=0).float()) / math.log(TIME_LOG_BASE)
    return b.long().clamp(max=buckets - 1)


class SASRecBlock(nn.TransformerEncoderLayer):
    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float, **_):
        super().__init__(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x, attn_mask, rel_time):
        b, _, length, _ = attn_mask.shape
        blocked = (~attn_mask).expand(b, self.self_attn.num_heads, length, length)
        return super().forward(x, src_mask=blocked.reshape(-1, length, length))


class RelativeBias(nn.Module):
    """Per-head bias from the relative position ``i - j`` plus, with ``time_buckets``, the
    log-bucketed time delta ``t_i - t_j``."""

    def __init__(self, heads: int, max_len: int, time_buckets: int | None):
        super().__init__()
        self.position = nn.Embedding(max_len, heads)
        self.time = nn.Embedding(time_buckets, heads) if time_buckets else None

    def forward(self, length: int, rel_time: torch.Tensor | None) -> torch.Tensor:
        pos = torch.arange(length, device=self.position.weight.device)
        bias = self.position((pos[:, None] - pos[None, :]).clamp(min=0)).permute(2, 0, 1)[None]
        if self.time is not None:
            bias = bias + self.time(rel_time).permute(0, 3, 1, 2)
        return bias


class HSTUBlock(nn.Module):
    """HSTU (Zhai et al. 2024, eq. 1-3) with softmax attention in place of the paper's
    pointwise SiLU/n: ``U,V,Q,K = SiLU(f1(n))``, ``A = softmax(QK^T/sqrt(d) + rab)``,
    ``y = f2(norm(AV) * U)``."""

    def __init__(
        self, hidden: int, heads: int, dropout: float, max_len: int, time_buckets: int | None, **_
    ):
        super().__init__()
        self.heads = heads
        self.norm = nn.RMSNorm(hidden)
        self.f1 = nn.Linear(hidden, 4 * hidden)
        self.rab = RelativeBias(heads, max_len, time_buckets)
        self.attn_norm = nn.RMSNorm(hidden)
        self.f2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, rel_time):
        b, length, h = x.shape
        u, v, q, k = F.silu(self.f1(self.norm(x))).chunk(4, dim=-1)
        q, k, v = (t.view(b, length, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        bias = self.rab(length, rel_time).masked_fill(~attn_mask, float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        a = a.transpose(1, 2).reshape(b, length, h)
        return x + self.dropout(self.f2(self.attn_norm(a) * u))


BLOCKS = {"sasrec": SASRecBlock, "hstu": HSTUBlock}


class Encoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        encoder: str = "sasrec",
        max_seq_length: int = 200,
        embedding_dim: int = 64,
        hidden_dim: int | None = None,
        num_blocks: int = 2,
        num_heads: int = 2,
        ffn_hidden_dim: int = 256,
        dropout: float = 0.0,
        use_time: bool = False,
        time_buckets: int = 64,
        reuse_item_embeddings: bool = False,
        normalize: bool = False,
    ):
        super().__init__()
        hidden = hidden_dim or embedding_dim
        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.use_time = use_time
        self.time_buckets = time_buckets
        self.normalize = normalize

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.in_proj = (
            nn.Linear(embedding_dim, hidden) if hidden != embedding_dim else nn.Identity()
        )
        self.position_embedding = (
            nn.Embedding(max_seq_length, hidden) if encoder == "sasrec" else None
        )
        self.time_gap_embedding = nn.Embedding(time_buckets, hidden) if use_time else None
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            BLOCKS[encoder](
                hidden=hidden,
                heads=num_heads,
                ffn=ffn_hidden_dim,
                dropout=dropout,
                max_len=max_seq_length,
                time_buckets=time_buckets if use_time else None,
            )
            for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.out_proj = (
            nn.Linear(hidden, embedding_dim) if hidden != embedding_dim else nn.Identity()
        )
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

    def body(self, x, attn_mask, rel_time):
        for block in self.blocks:
            x = block(x, attn_mask, rel_time)
        return self.out_proj(self.final_norm(x))

    def forward(self, items: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        length = items.shape[1]
        key_valid = items != 0
        eye = torch.eye(length, dtype=torch.bool, device=items.device)
        causal = torch.ones(length, length, dtype=torch.bool, device=items.device).tril()
        # The diagonal keeps every left-padding row non-empty, so no row softmaxes to NaN.
        attn_mask = causal & (key_valid[:, None, None, :] | eye)

        x = self.in_proj(self.item_embedding(items))
        if self.position_embedding is not None:
            x = x + self.position_embedding.weight[:length]
        rel_time = None
        if self.use_time:
            gap = torch.diff(timestamps, dim=1, prepend=timestamps[:, :1])
            gap = torch.where(F.pad(key_valid, (1, -1)), gap, 0)
            x = x + self.time_gap_embedding(time_bucket(gap, self.time_buckets))
            rel_time = timestamps[:, :, None] - timestamps[:, None, :]
            rel_time = time_bucket(rel_time, self.time_buckets)
        return self.body(self.embedding_dropout(x), attn_mask, rel_time)

    def predict_last(
        self, items: torch.Tensor, timestamps: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Left-padded: the last real token is always at index -1.
        q = self.forward(items, timestamps)[:, -1]
        return F.normalize(q.float(), dim=-1) if self.normalize else q


def build_encoder(cfg: TrainConfig, num_items: int) -> Encoder:
    return Encoder(
        num_items=num_items,
        encoder=cfg.encoder,
        max_seq_length=cfg.max_seq_length,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        dropout=cfg.dropout,
        use_time=cfg.use_time,
        time_buckets=cfg.time_buckets,
        reuse_item_embeddings=cfg.reuse_item_embeddings,
        normalize=cfg.normalize,
    )
