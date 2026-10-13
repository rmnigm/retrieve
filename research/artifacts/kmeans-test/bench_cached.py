"""``bench.cli`` with SilverTorch's k-means replaced by the centroids fit_centroids.py cached in $KMEANS_CACHE.

The assignment is recomputed from the cached centroids (the library's ``KMeans.assign``), so the index equals a
build with that variant; a missing cache file raises instead of fitting. Usage: as ``python -m bench.cli``.
"""

import os
import sys
from pathlib import Path

import retrieve.modules.silvertorch as st
import torch
from retrieve.indexing.kmeans import KMeans

CACHE = Path(os.environ["KMEANS_CACHE"])


class CachedKMeans(KMeans):
    def fit(self, embs):
        c = torch.load(
            CACHE / f"{self.init}-{self.n_iter}-{self.n_lists}-s{self.seed}.pt",
            map_location=embs.device,
        )
        return c, self.assign(embs, c)


st.KMeans = CachedKMeans
from bench.cli import main  # noqa: E402

sys.exit(main())
