"""Deprecated 0.1 kernel paths: ``retrieve.kernels.silvertorch.official`` is
``retrieve.ops.official``. Temporary tooling for the old-harness golden worktree (plan L D10),
deleted at roadmap C5; the Triton kernels are ``retrieve.ops.triton``."""

import sys
import types
import warnings

from retrieve.ops import official

warnings.warn(
    "retrieve.kernels is deprecated (torchretrieve 0.2): use retrieve.ops.triton / "
    "retrieve.ops.reference / retrieve.ops.official; removed at roadmap C5",
    DeprecationWarning,
    stacklevel=2,
)

silvertorch = types.ModuleType("retrieve.kernels.silvertorch")
silvertorch.official = official
sys.modules["retrieve.kernels.silvertorch"] = silvertorch
sys.modules["retrieve.kernels.silvertorch.official"] = official
