"""Registered kernels, one namespace per backend: ``retrieve.ops.triton`` (the Triton kernels,
``torch.ops.retrieve.*``), ``retrieve.ops.reference`` (the same signatures in pure torch — the
``"torch"`` backend and the parity oracle) and ``retrieve.ops.official`` (Meta's
``torch.ops.st.*`` behind the B1 adapter). Each is imported on first attribute access, so
``import retrieve.ops`` registers nothing; ``retrieve.ops.tune`` is the offline autotuner."""

from __future__ import annotations

import importlib

_NAMESPACES = ("triton", "reference", "official")


def __getattr__(name: str):
    if name in _NAMESPACES:
        return importlib.import_module(f"retrieve.ops.{name}")
    raise AttributeError(f"module 'retrieve.ops' has no attribute {name!r}")


def available_backends() -> tuple[str, ...]:
    """``("triton", "torch")``, plus ``"official"`` when Meta's package loads here."""
    from retrieve.ops.official import is_available

    return ("triton", "torch", "official") if is_available() else ("triton", "torch")


__all__ = ["available_backends"]
