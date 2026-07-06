"""Public surface for the dataset package — download/parse/attrs/encode CLIs.

The CLIs themselves are exposed as console scripts in `pyproject.toml`
(`yambda`, `arxiv`, `goodreads`); this module re-exports the library
helpers that retrieval and training code import directly.
"""

from .common import dense_remap_ids, sample_rare_biased_wide, synthesize_qa_narrow
from .constants import Constants
from .yambda import Data, preprocess

__all__ = [
    "Constants",
    "Data",
    "dense_remap_ids",
    "preprocess",
    "sample_rare_biased_wide",
    "synthesize_qa_narrow",
]
