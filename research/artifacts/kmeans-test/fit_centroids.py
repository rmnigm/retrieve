"""Fit one k-means variant on a dataset's items with the library's KMeans and cache the centroids for bench_cached.py.

Writes ``<out>/<init>-<n_iter>-<n_lists>-s<seed>.pt`` (centroids) and ``.json`` (fit seconds, cluster-size balance:
max / median, CV, empty lists). k-means++ is the library's full D² seeding over all N items (no subsample).

usage: fit_centroids.py DATASET_CONTENT_DIR OUT N_LISTS INIT N_ITER
"""

import json
import sys
import time
from pathlib import Path

import torch
from eval_datasets import layout
from retrieve.indexing.kmeans import KMeans

content, out = Path(sys.argv[1]), Path(sys.argv[2])
n_lists, init, n_iter = int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
embs = layout.load_text_items(content, torch.device("cuda"))
torch.cuda.synchronize()
t0 = time.perf_counter()
centroids, assign = KMeans(n_lists=n_lists, n_iter=n_iter, seed=0, init=init).fit(embs)
torch.cuda.synchronize()
fit_s = time.perf_counter() - t0
sizes = torch.bincount(assign, minlength=n_lists).double()
stem = f"{init}-{n_iter}-{n_lists}-s0"
torch.save(centroids.cpu(), out / f"{stem}.pt")
stats = dict(
    n_items=embs.shape[0], n_lists=n_lists, init=init, n_iter=n_iter, fit_s=fit_s,
    max_over_median=(sizes.max() / sizes.median()).item(), cv=(sizes.std() / sizes.mean()).item(),
    empty=int((sizes == 0).sum()), max=int(sizes.max()), median=sizes.median().item(),
)  # fmt: skip
(out / f"{stem}.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats))
