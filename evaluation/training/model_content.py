from __future__ import annotations

import torch
import torch.nn as nn

from .model import GSASRec


class ContentItemEmbedding(nn.Module):
    """Quacks like nn.Embedding for the rest of the codebase.

    - `forward(ids)` returns `proj(text_emb[ids])`
    - `.weight` materializes the full `(num_items+1, embedding_dim)` table
      (used at save/eval time)
    - `.num_embeddings` exposes the row count (used by gbce_loss)

    The frozen text matrix is stored as a non-persistent buffer so the
    state_dict stays small; callers must re-supply it at load time.
    """

    def __init__(
        self,
        text_emb: torch.Tensor,
        embedding_dim: int,
        proj_type: str = "linear",
        mlp_hidden_mult: int = 2,
    ):
        super().__init__()
        self.register_buffer("text_emb", text_emb.float(), persistent=False)
        d_in = text_emb.shape[1]
        if proj_type == "linear":
            self.proj = nn.Linear(d_in, embedding_dim, bias=False)
        elif proj_type == "mlp":
            d_hidden = mlp_hidden_mult * embedding_dim
            self.proj = nn.Sequential(
                nn.Linear(d_in, d_hidden, bias=True),
                nn.GELU(),
                nn.Linear(d_hidden, embedding_dim, bias=False),
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type!r}")
        self.proj_type = proj_type

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.text_emb[ids])

    @property
    def weight(self) -> torch.Tensor:
        return self.proj(self.text_emb)

    @property
    def num_embeddings(self) -> int:
        return self.text_emb.shape[0]


class GSASRecContent(GSASRec):
    """gSASRec variant with `proj(frozen_text_emb)` as the input embedding.

    Input is always content-driven. Output is either:
      - a learned `nn.Embedding(num_items+1, D)` (Variant B, default), or
      - the same `ContentItemEmbedding` as the input (Variant C, tied) — true
        weight tying: a single `proj` and a single frozen `text_emb` feed
        both lookup paths.
    """

    def __init__(
        self,
        num_items: int,
        text_emb: torch.Tensor,
        tie_content_output: bool = False,
        content_proj_type: str = "linear",
        **kwargs,
    ):
        # We manage tying ourselves, so the parent's reuse path stays off.
        kwargs["reuse_item_embeddings"] = False
        super().__init__(num_items=num_items, **kwargs)

        if text_emb.shape[0] != num_items + 1:
            raise ValueError(
                f"text_emb rows {text_emb.shape[0]} != num_items+1 {num_items + 1}"
            )

        embedding_dim = kwargs.get("embedding_dim", self.embedding_dim)
        self.item_embedding = ContentItemEmbedding(
            text_emb, embedding_dim, proj_type=content_proj_type
        )

        if tie_content_output:
            self.output_embedding = self.item_embedding

        # Match the trunc-normal scheme the parent uses for its other weights;
        # nn.Linear's default kaiming-uniform init lands at a different scale.
        for m in self.item_embedding.proj.modules() if content_proj_type == "mlp" \
                else [self.item_embedding.proj]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
