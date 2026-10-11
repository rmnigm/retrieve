"""Roadmap B2 / library review A3(a): does ``torch.argsort(assignments, stable=True)`` change
anything ``SilverTorch._build_ivf`` produces?

For each regime the IVF is built twice from the *same* k-means assignment (the k-means itself is
not bit-deterministic across runs, so it runs once per regime): once with the current
``torch.argsort(assignments)`` and once with ``stable=True``. Compared: the permutation itself,
the padded layout, and — through a real ``SilverTorch`` layer whose buffers are swapped for the
stable variant — the forward's ids and scores on every backend that is installed. Prints one
row per regime; ``python argsort_stable_probe.py --json out.json`` also dumps the numbers.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "retrieve"
    ),
)
from tests.conftest import make_index, make_query  # noqa: E402

from retrieve.layers.silvertorch import SilverTorch  # noqa: E402
from retrieve.layers.utils.kmeans import KMeansTorch  # noqa: E402

# (N, n_lists, n_probe, k): the layer-level regimes the suite builds through _build_ivf
# (test_silvertorch / T6: N=4096; TestEdgeCases: N=256=n_lists; test_bloom_hash: 256/8) plus
# sizes on both sides of torch's CUDA small-sort threshold (segments <= 4096 sort in
# registers, larger ones through cub radix sort).
REGIMES = [
    (256, 8, 4, 8),
    (256, 256, 256, 64),
    (1024, 16, 4, 8),
    (2880, 32, 8, 32),
    (4096, 64, 8, 64),
    (4097, 64, 8, 64),
    (6144, 64, 8, 32),
    (16384, 64, 8, 64),
    (131072, 1024, 32, 100),
]
D = 128


def layout(assignments: torch.Tensor, n_lists: int, stable: bool):
    """``_build_ivf``'s tail, verbatim, parameterised over the sort."""
    n = assignments.numel()
    cluster_sizes = torch.bincount(assignments, minlength=n_lists)
    max_size = int(cluster_sizes.max().item())
    sort_idx = (
        torch.argsort(assignments, stable=True)
        if stable
        else torch.argsort(assignments)
    )
    sorted_clusters = assignments[sort_idx]
    offsets = torch.zeros(n_lists + 1, dtype=torch.long, device=assignments.device)
    offsets[1:] = cluster_sizes.cumsum(0)
    padded = torch.full(
        (n_lists, max_size), -1, dtype=torch.long, device=assignments.device
    )
    within_slot = torch.arange(n, device=assignments.device) - offsets[sorted_clusters]
    padded[sorted_clusters, within_slot] = sort_idx
    return sort_idx, padded, offsets, cluster_sizes


def within_cluster_sorted(
    sort_idx: torch.Tensor, sorted_clusters: torch.Tensor
) -> bool:
    """True iff ids are ascending inside every cluster run (what a stable sort guarantees)."""
    same = sorted_clusters[1:] == sorted_clusters[:-1]
    return bool((sort_idx[1:][same] > sort_idx[:-1][same]).all().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    backends = ["torch", "triton"]
    try:
        from retrieve.kernels.silvertorch.official import ensure_loaded

        ensure_loaded()
        backends.append("official")
    except ImportError:
        pass
    rows = []
    for n, n_lists, n_probe, k in REGIMES:
        embs = make_index(n, D)
        query = make_query(16, D)
        _, assignments = KMeansTorch(n_lists=n_lists, n_iter=3, seed=0).fit(embs)
        idx_u, pad_u, off_u, _ = layout(assignments, n_lists, stable=False)
        idx_s, pad_s, off_s, _ = layout(assignments, n_lists, stable=True)
        row = {
            "n": n,
            "n_lists": n_lists,
            "perm_equal": bool(torch.equal(idx_u, idx_s)),
            "perm_diff_positions": int((idx_u != idx_s).sum().item()),
            "padded_equal": bool(torch.equal(pad_u, pad_s)),
            "unstable_is_id_ordered_within_clusters": within_cluster_sorted(
                idx_u, assignments[idx_u]
            ),
            "stable_is_id_ordered_within_clusters": within_cluster_sorted(
                idx_s, assignments[idx_s]
            ),
            "forward": {},
        }
        for backend in backends:
            m = SilverTorch(
                k=k, n_lists=n_lists, n_probe=n_probe, n_iter=3, backend=backend
            )
            m.register_index(embs)  # its own k-means; the buffers are replaced below
            base = copy.deepcopy(m)
            stab = copy.deepcopy(m)
            for mod, (idx, pad, off) in (
                (base, (idx_u, pad_u, off_u)),
                (stab, (idx_s, pad_s, off_s)),
            ):
                codes = (
                    m.item_codes if backend != "official" else m.item_codes[m.inv_perm]
                )
                if backend == "official":
                    inv = torch.empty_like(idx)
                    inv[idx] = torch.arange(n, device="cuda")
                    mod.item_codes = codes[idx].contiguous()
                    mod.sort_perm, mod.inv_perm, mod.cluster_offsets = idx, inv, off
                    mod.cluster_sizes = torch.bincount(assignments, minlength=n_lists)
                else:
                    mod.padded_cluster_items = pad
                    mod.cluster_sizes = torch.bincount(assignments, minlength=n_lists)
                mod._max_cluster_size = int(pad.shape[1])
            ids_u, sc_u = base(query)
            ids_s, sc_s = stab(query)
            tied = 0
            if not torch.equal(ids_u, ids_s):
                # every differing position must sit inside a run of equal scores
                for b in range(ids_u.shape[0]):
                    for j in range(ids_u.shape[1]):
                        if ids_u[b, j] != ids_s[b, j]:
                            tied += 1
                            assert (sc_u[b] == sc_u[b, j]).sum() > 1, (backend, b, j)
            row["forward"][backend] = {
                "scores_equal": bool(torch.equal(sc_u, sc_s)),
                "ids_equal": bool(torch.equal(ids_u, ids_s)),
                "id_slots_differing_all_within_score_ties": tied,
            }
        rows.append(row)
        parts = []
        for b, r in row["forward"].items():
            ids = (
                "="
                if r["ids_equal"]
                else f"tie-perm×{r['id_slots_differing_all_within_score_ties']}"
            )
            parts.append(f"{b}:scores={'=' if r['scores_equal'] else '≠'},ids={ids}")
        fw = " ".join(parts)
        print(
            f"N={n:>6} n_lists={n_lists:>4}: perm_equal={row['perm_equal']!s:5} "
            f"(diff {row['perm_diff_positions']:>6}) padded_equal={row['padded_equal']!s:5} "
            f"unstable_id_ordered={row['unstable_is_id_ordered_within_clusters']!s:5} | {fw}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
