"""Deprecated 0.1 import paths, re-exported from the 0.2 layout under a ``DeprecationWarning``.

Temporary tooling, not an API promise: it exists so the old-harness golden worktree runs
against the new library (plan L D10) and is deleted at roadmap C5. ``retrieve.layers.filters``,
``retrieve.layers.silvertorch`` and ``retrieve.layers.linr.postfilter_knn_int8`` — the paths the
old harness imports — are aliased as modules too."""

import sys
import types
import warnings

from retrieve import indexing, modules
from retrieve.functional import combine_indices, combine_masks, compact_mask, post_filter_topk
from retrieve.indexing import KMeans, quantize_int8, quantize_oporp_1bit, quantize_simhash_1bit
from retrieve.modules import (
    BloomFilter,
    ExactAttributeFilter,
    FullScanKNN,
    OfficialConfig,
    OneBitKNN,
    PostfilterKNN,
    PostfilterKNNInt8,
    PrefilterKNN,
    SilverTorch,
    SilverTorchBuilder,
    SimHashKNN,
)

warnings.warn(
    "retrieve.layers is deprecated (torchretrieve 0.2): import from retrieve, "
    "retrieve.modules, retrieve.functional or retrieve.indexing; removed at roadmap C5",
    DeprecationWarning,
    stacklevel=2,
)

KMeansTorch = KMeans


def build_silvertorch(item_embs, k, *, item_clause_attrs=None, clause_is_reverse=None, **kwargs):
    """The 0.1 one-call constructor, over ``SilverTorchBuilder`` (``kwargs`` are the module's)."""
    b = SilverTorchBuilder(k=k, **kwargs).set_item_embeddings(item_embs)
    if item_clause_attrs is not None:
        b.set_item_attributes(item_clause_attrs, clause_is_reverse)
    return b.build()


def _alias(name: str, **attrs: object) -> None:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, mod)


_alias(
    "retrieve.layers.filters",
    BloomFilter=BloomFilter,
    ExactAttributeFilter=ExactAttributeFilter,
    combine_indices=combine_indices,
    combine_masks=combine_masks,
    bloom_hash=indexing.bloom_hash,
)
_alias(
    "retrieve.layers.silvertorch",
    SilverTorch=SilverTorch,
    OfficialConfig=OfficialConfig,
    build_silvertorch=build_silvertorch,
)
_alias("retrieve.layers.linr", **{n: getattr(modules, n) for n in modules.__all__ if "KNN" in n})
_alias("retrieve.layers.linr.postfilter_knn_int8", PostfilterKNNInt8=PostfilterKNNInt8)

__all__ = [
    "BloomFilter",
    "ExactAttributeFilter",
    "FullScanKNN",
    "KMeansTorch",
    "OfficialConfig",
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
    "SilverTorch",
    "SimHashKNN",
    "build_silvertorch",
    "combine_indices",
    "combine_masks",
    "compact_mask",
    "post_filter_topk",
    "quantize_int8",
    "quantize_oporp_1bit",
    "quantize_simhash_1bit",
]
