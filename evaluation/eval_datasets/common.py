"""Helpers shared by the dataset ETL modules: attribute synthesis, hash sampling, plumbing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def synthesize_qa_narrow(
    target_first: list[int],
    narrow_t: torch.Tensor,
    n_clauses: int,
) -> torch.Tensor:
    """Build per-user `query_attrs_narrow` of shape `[n_users, n_clauses]`.

    For each clause `c`, take the first non-pad value of
    `narrow_t[target - 1, c, :]` (i.e. the most-frequent or first-listed
    attribute the held-out target carries). Users with `target <= 0`
    (no held-out item) get all `-1`.

    ``narrow_t`` is the ``[N, n_clauses, A]`` 0-indexed dense item-attr
    tensor (row ``i`` = item_id ``i + 1``). ``target_first[u]`` is a
    1-indexed item_id (matching ``item_id_map.json``).
    """
    n_users = len(target_first)
    qa = narrow_t.new_full((n_users, n_clauses), -1).long()
    for u, tgt in enumerate(target_first):
        if tgt <= 0:
            continue
        for c in range(n_clauses):
            row = narrow_t[tgt - 1, c]
            for v in row.tolist():
                if v != -1:
                    qa[u, c] = v
                    break
    return qa


def sample_rare_biased_wide(
    target_first: list[int],
    wide_t: torch.Tensor,
    wide_global_freq: list[int] | np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sample 1-shelf rare and 2-shelf (common+rare) wide attrs per user.

    Returns `(qa_wide_1, qa_wide_2, n_drop_1, n_drop_2)`.

    1-shelf: ``p ∝ 1/sqrt(global_freq)`` — biases toward rare shelves so the
    filter is selective.
    2-shelf: one common (``p ∝ sqrt(freq)``) and one rare (``p ∝ 1/sqrt(freq)``
    among the remaining bag entries). Both `-1` if the bag is empty / has
    only a single entry / target is missing.
    """
    rng = np.random.default_rng(seed)
    n_users = len(target_first)
    freq_np = np.asarray(wide_global_freq, dtype=np.int64)
    qa_wide_1 = np.full((n_users,), -1, dtype=np.int64)
    qa_wide_2 = np.full((n_users, 2), -1, dtype=np.int64)
    n_drop_1 = 0
    n_drop_2 = 0
    wide_np = wide_t.numpy()
    # ``wide_t`` is ``[N, 1, BAG]`` 0-indexed dense; ``tgt`` is a 1-indexed
    # item_id, so index at ``tgt - 1``.
    for u, tgt in enumerate(target_first):
        if tgt <= 0:
            n_drop_1 += 1
            n_drop_2 += 1
            continue
        bag = wide_np[tgt - 1, 0]
        bag = bag[bag != -1]
        if len(bag) == 0:
            n_drop_1 += 1
            n_drop_2 += 1
            continue
        freq = freq_np[bag].astype(np.float64)
        w_rare = 1.0 / np.sqrt(np.maximum(freq, 1.0))
        w_rare = w_rare / w_rare.sum()
        qa_wide_1[u] = int(rng.choice(bag, p=w_rare))
        if len(bag) < 2:
            n_drop_2 += 1
            continue
        w_common = np.sqrt(freq)
        w_common = w_common / w_common.sum()
        common_idx = int(rng.choice(len(bag), p=w_common))
        common_id = int(bag[common_idx])
        rest = np.array([i for i in range(len(bag)) if i != common_idx])
        if len(rest) == 0:
            n_drop_2 += 1
            continue
        w_rare2 = 1.0 / np.sqrt(np.maximum(freq[rest], 1.0))
        w_rare2 = w_rare2 / w_rare2.sum()
        rare_id = int(bag[rest[int(rng.choice(len(rest), p=w_rare2))]])
        qa_wide_2[u, 0] = common_id
        qa_wide_2[u, 1] = rare_id
    return qa_wide_1, qa_wide_2, n_drop_1, n_drop_2


def pmid_hash(pmids: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64's finaliser over the ids, keyed by ``seed`` — a fixed, order-free
    pseudo-random rank per id (uint64). Used by ``select_pmids``."""
    with np.errstate(over="ignore"):
        z = np.asarray(pmids, dtype=np.uint64) + np.uint64(seed + 1) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def select_pmids(pmids: np.ndarray, keep_items: int | None, seed: int = 0) -> np.ndarray:
    """``[len(pmids)]`` bool: the ``keep_items`` ids with the smallest ``pmid_hash`` (ties on
    the id itself), or everything when ``keep_items`` is ``None`` / not smaller than the
    catalog. The choice depends only on the id set and the seed — not on shard order, chunking
    or which shards have been fetched — so a resumed or re-sharded run keeps the same items,
    spread uniformly over the catalog instead of being its oldest prefix."""
    n = int(pmids.shape[0])
    if keep_items is None or keep_items >= n:
        return np.ones(n, dtype=bool)
    if keep_items <= 0:
        raise ValueError("keep_items must be positive")
    order = np.lexsort((pmids, pmid_hash(pmids, seed)))
    keep = np.zeros(n, dtype=bool)
    keep[order[:keep_items]] = True
    return keep


def parse_ranges(spec: str | None, n: int) -> list[int]:
    """``"0-3,7"`` → ``[0, 1, 2, 3, 7]``, sorted and de-duplicated; ``None`` → ``range(n)``."""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        if lo:
            out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def merge_prep_log(output: Path, key: str, payload: dict) -> None:
    """Set ``key`` in ``output/prep_log.json``, keeping the other steps' entries."""
    path = output / "prep_log.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2))


def file_hexdigest(path: Path, algorithm: str) -> str:
    """Hex digest of a file, streamed (``hashlib.file_digest`` needs 3.11)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "file_hexdigest",
    "merge_prep_log",
    "parse_ranges",
    "pmid_hash",
    "sample_rare_biased_wide",
    "select_pmids",
    "synthesize_qa_narrow",
]
