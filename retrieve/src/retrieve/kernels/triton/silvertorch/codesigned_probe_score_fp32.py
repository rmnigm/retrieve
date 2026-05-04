"""Re-export shim. The fp32 wrapper now lives alongside the int8 wrapper in
``codesigned_probe_score.py`` — both call the same merged kernel parameterized
by an ``INT8: tl.constexpr`` flag. This module exists only to preserve the
historic import path used by ``layers/silvertorch/fp32.py``.
"""

from __future__ import annotations

from retrieve.kernels.triton.silvertorch.codesigned_probe_score import (
    codesigned_probe_score_fp32,
)

__all__ = ["codesigned_probe_score_fp32"]
