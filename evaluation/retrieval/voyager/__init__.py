"""Voyager (Spotify HNSW) CPU quality baseline.

Lives in its own subpackage because it's the only CPU baseline in an
otherwise GPU/torch/triton bench: it shares no kernels with the main
algos, runs entirely on CPU, doesn't participate in filter sweeps, and
exists purely to give a "realistic CPU ANN deployment" quality
reference for the big-catalog quality runs (yambda-500m, goodreads,
arxiv). The unified ``evaluate`` CLI sweeps GPU algos × backends ×
filter kinds; this one runs voyager × params × k and emits recall/ndcg
only.
"""

from retrieval.voyager.baseline import VoyagerHNSW

__all__ = ["VoyagerHNSW"]
