from __future__ import annotations

from typing import Literal

from torch import Tensor

from retrieve.layers.linr.v1 import LiNR_V1, build_linr_v1
from retrieve.layers.linr.v2 import LiNR_V2, build_linr_v2
from retrieve.layers.linr.v3 import LiNR_V3, build_linr_v3

Backend = Literal["torch", "triton"]


def build_linr_index(
    version: int,
    item_embs: Tensor,
    k: int,
    *,
    backend: Backend = "triton",
    **kwargs,
) -> LiNR_V1 | LiNR_V2 | LiNR_V3:
    """Build a LiNR index of the requested version and backend.

    ``backend="triton"`` returns the fused-kernel subclass; ``"torch"`` the
    pure-PyTorch baseline. The Triton modules are imported lazily so a
    torch-only caller never pays the triton-import cost.
    """
    if backend == "torch":
        torch_builders = {1: build_linr_v1, 2: build_linr_v2, 3: build_linr_v3}
        if version not in torch_builders:
            raise ValueError(f"Unknown LiNR version: {version}")
        return torch_builders[version](item_embs, k, **kwargs)

    if backend == "triton":
        from retrieve.layers.linr.v1_triton import build_linr_v1_triton
        from retrieve.layers.linr.v2_triton import build_linr_v2_triton
        from retrieve.layers.linr.v3_triton import build_linr_v3_triton

        triton_builders = {
            1: build_linr_v1_triton,
            2: build_linr_v2_triton,
            3: build_linr_v3_triton,
        }
        if version not in triton_builders:
            raise ValueError(f"Unknown LiNR version: {version}")
        return triton_builders[version](item_embs, k, **kwargs)

    raise ValueError(f"Unknown backend: {backend!r}; expected 'torch' or 'triton'")
