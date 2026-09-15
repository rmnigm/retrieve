"""Meta's own modules (``silvertorch.modules``: ``BloomIndexSearchModule``,
``FilterQueryParserModule`` and their builders), re-exported on first access once the official
extension loads; without the ``official`` extra the access raises ``OfficialMissing`` with the
install hint. ``SilverTorch(backend="official")`` does not go through these — it calls the ops
(``retrieve.ops.official``) directly."""

from __future__ import annotations

import importlib

_NAMES = (
    "BloomIndexSearchModule",
    "BloomIndexSearchModuleBuilder",
    "FilterQueryParserModule",
    "FilterQueryParserModuleBuilder",
)
__all__ = list(_NAMES)


def __getattr__(name: str):
    if name not in _NAMES:
        raise AttributeError(f"module 'retrieve.modules.official' has no attribute {name!r}")
    from retrieve.ops.official import ensure_loaded

    ensure_loaded()
    return getattr(importlib.import_module("silvertorch.modules"), name)
