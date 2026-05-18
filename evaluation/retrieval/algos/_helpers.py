"""Shared algo-construction helpers.

Lives in its own module (rather than ``__init__.py``) so individual algo
files can import it without triggering a circular import through
``algos/__init__.py``'s class re-exports.
"""

from __future__ import annotations

import torch.nn as nn

from retrieve.interfaces import FilterModule


def collect_modules(
    *base: nn.Module, filter_mod: FilterModule | None
) -> list[nn.Module]:
    """Build the ``algo.algo_modules`` list the driver iterates for memory cleanup."""
    mods = list(base)
    if filter_mod is not None:
        mods.append(filter_mod)
    return mods
