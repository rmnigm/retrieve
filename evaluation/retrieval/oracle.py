"""Exact filtered oracle for harness v2 (H §2.2, §8.2 I) and the resume key (§8.2 B).

Brute-force ``q @ E^T`` under the *exact* mask (``ExactAttributeFilter`` even on bloom
cells — bloom's false positives must never leak into ground truth), top-``k_max`` with
``-inf`` ties written as ``-1`` so short-filled rows never score against the algos' ``-1``
sentinel. Skipped rows (no live clause) stay all ``-1``. Indices are 0-indexed positions in
the pad-row-dropped ``item_embs``.

**Blob v4** — one dict at ``<gt_dir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt``:

| key | value |
|---|---|
| ``version`` | ``4`` |
| ``topk`` | ``[U, k_gt]`` int64, ``-1`` padded |
| ``pass_counts`` | ``[U]`` int64: items passing the exact mask; ``-1`` on skipped rows |
| ``pass_rate`` | mean over kept rows of ``pass_counts / n_items`` (H §2.2, LiNR's axis) |
| ``targets_in_filter`` | ``[U, T]`` bool: held-out target ``t`` of user ``u`` passes the mask |
| ``target_in_filter`` | ``[U]`` bool: any held-out target passes (``n_queries_heldout = sum``) |
| ``n_items``, ``n_queries``, ``n_kept``, ``k_gt``, ``sweep``, ``clauses`` | the build's shape |
| ``fingerprint`` | the content hash below |
| ``code_version``, ``harness_commit``, ``torch``, ``created`` | provenance of the build |

The **fingerprint** hashes shapes, dtypes and a fixed 64-row linspace sample of
``item_embs``, ``queries``, ``targets`` and the sweep's ``qa``, the *full bytes* of the item
side of the predicate — ``item_attrs`` and ``clause_is_reverse``, via ``attrs_digest``
(``data.load_inputs`` computes it once per ``(dataset, dim)``; the ETL edits that tensor
piecemeal, so a row sample is not enough) — plus ``clauses`` and ``k_gt``. Same-shape content
changes — another dim off the same ``data_dir``, regenerated attrs, a retrained checkpoint, a
changed ``users_limit`` — change the file name, so a stale blob is never *read*; the hash in
the name is what makes the blob a portable artifact (§8.2 I). Bloom pass rates are not
cached: they depend on ``m_bits`` / ``k_hash`` and cost one mask pass (``pass_counts``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm import tqdm

from retrieval.bench import _git, atomic_write, code_version
from retrieve.interfaces import FilterModule

BLOB_VERSION = 4
_FP_SAMPLE_ROWS = 64


def attrs_digest(item_attrs: torch.Tensor | None, clause_is_reverse: torch.Tensor | None) -> str:
    """sha256 over the full bytes (shape, dtype, content) of ``item_attrs`` and
    ``clause_is_reverse`` — the item side of the predicate. ≈ 1 s per GB; computed once per
    ``(dataset, dim)`` by ``data.load_inputs`` and passed down to ``fingerprint``."""
    h = hashlib.sha256()
    for t in (item_attrs, clause_is_reverse):
        if t is None:
            h.update(b"none")
            continue
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprint(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor | None,
    qa_sweep: torch.Tensor | None,
    clauses: tuple[int, ...] | None,
    k_gt: int,
    *,
    attrs_digest: str,
) -> str:
    """Deterministic content hash: shapes + dtypes + a fixed row sample of the four query-side
    tensors (<10 ms), the precomputed ``attrs_digest``, ``clauses`` and ``k_gt``."""
    h = hashlib.sha256()
    for t in (item_embs, queries, targets, qa_sweep):
        if t is None:
            h.update(b"none")
            continue
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        idx = torch.linspace(0, t.shape[0] - 1, steps=min(_FP_SAMPLE_ROWS, t.shape[0])).long()
        h.update(t[idx].detach().float().cpu().contiguous().numpy().tobytes())
    h.update(attrs_digest.encode())
    h.update(repr((None if clauses is None else tuple(clauses), int(k_gt))).encode())
    return h.hexdigest()


def _batches(n_rows: int, skip_mask: torch.Tensor | None, batch_size: int, desc: str):
    keep = ~skip_mask if skip_mask is not None else torch.ones(n_rows, dtype=torch.bool)
    keep_idx = keep.nonzero(as_tuple=False).reshape(-1)
    rng = range(0, keep_idx.numel(), batch_size)
    for s in tqdm(rng, desc=desc, leave=False, disable=not sys.stderr.isatty()):
        yield keep_idx[s : s + batch_size]


@torch.inference_mode()
def pass_counts(
    filter_mod: FilterModule,
    qa_sweep: torch.Tensor,
    skip_mask: torch.Tensor | None,
    *,
    batch_size: int = 64,
    device: torch.device,
) -> torch.Tensor:
    """``[U]`` int64 items passing ``filter_mod`` per query (``-1`` on skipped rows). With a
    ``BloomFilter`` this is the bloom pass count (H §2.2 ``bloom_fp_rate``)."""
    out = torch.full((qa_sweep.shape[0],), -1, dtype=torch.long)
    for idx in _batches(qa_sweep.shape[0], skip_mask, batch_size, "pass_counts"):
        mask = filter_mod.evaluate_mask(qa_sweep[idx].to(device, non_blocking=True))
        out[idx] = mask.sum(dim=1).cpu()
    return out


def pass_rate(counts: torch.Tensor, n_items: int) -> float:
    """Mean of ``counts / n_items`` over kept rows (``counts >= 0``); ``nan`` if none."""
    kept = counts[counts >= 0]
    return (kept.double() / n_items).mean().item() if kept.numel() else float("nan")


def bloom_fp_rate(bloom_counts: torch.Tensor, exact_counts: torch.Tensor, n_items: int) -> float:
    """Mean per-query false-positive rate ``(bloom − exact) / (N − exact)`` over kept rows
    with at least one negative; ``nan`` if none (bloom ⊇ exact, so this is ≥ 0)."""
    keep = (exact_counts >= 0) & (bloom_counts >= 0) & (exact_counts < n_items)
    fp = (bloom_counts[keep] - exact_counts[keep]).double()
    neg = (n_items - exact_counts[keep]).double()
    return (fp / neg).mean().item() if keep.any() else float("nan")


@torch.inference_mode()
def compute(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    filter_mod: FilterModule | None,
    k_gt: int,
    *,
    targets: torch.Tensor | None = None,
    batch_size: int = 64,
    device: torch.device,
) -> dict[str, Any]:
    """The v4 content fields (``topk``, ``pass_counts``, ``targets_in_filter``,
    ``target_in_filter``, ``pass_rate``) for one sweep. ``filter_mod`` must be exact; with
    ``None`` (or no ``qa_sweep``) every item passes. ``targets`` is ``[U, T]`` ``-1``-padded."""
    n_users, n_items = queries.shape[0], int(item_embs.shape[0])
    k_eff = min(k_gt, n_items)
    topk = torch.full((n_users, k_gt), -1, dtype=torch.long)
    counts = torch.full((n_users,), -1, dtype=torch.long)
    t_shape = targets.shape if targets is not None else (n_users, 0)
    tif = torch.zeros(t_shape, dtype=torch.bool)
    item_embs_t = item_embs.t().contiguous()
    for idx in _batches(n_users, skip_mask, batch_size, "oracle"):
        q = queries[idx].to(device, non_blocking=True)
        scores = q @ item_embs_t
        if filter_mod is not None and qa_sweep is not None:
            mask = filter_mod.evaluate_mask(qa_sweep[idx].to(device, non_blocking=True))
            scores = scores.masked_fill(~mask, float("-inf"))
            counts[idx] = mask.sum(dim=1).cpu()
        else:
            mask = None
            counts[idx] = n_items
        top = torch.topk(scores, k_eff, dim=1)
        # Fewer than k_eff survivors: the tail ties at -inf and topk picks the lowest ids.
        topk[idx, :k_eff] = torch.where(torch.isfinite(top.values), top.indices, -1).cpu()
        if targets is not None:
            t = targets[idx].to(device, non_blocking=True)
            valid = t != -1
            passes = valid if mask is None else valid & mask.gather(1, t.clamp(min=0))
            tif[idx] = passes.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "topk": topk,
        "pass_counts": counts,
        "pass_rate": pass_rate(counts, n_items),
        "targets_in_filter": tif,
        "target_in_filter": tif.any(dim=1),
    }


def blob_path(gt_dir: Path, sweep: str, fp: str) -> Path:
    return gt_dir / f"oracle_v4_{sweep}_{fp[:16]}.pt"


def load_or_build(
    gt_dir: Path,
    sweep: str,
    k_gt: int,
    *,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor | None,
    qa_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    clauses: tuple[int, ...] | None,
    filter_mod: FilterModule | None,
    attrs_digest: str,
    device: torch.device,
) -> dict[str, Any]:
    """The v4 blob for one sweep: read from ``blob_path`` when present, else build and
    save (atomically). The name carries the fingerprint, so a stale blob is simply never
    found; an unreadable file at the path (a crash mid-save) is rebuilt, not fatal."""
    fp = fingerprint(
        item_embs, queries, targets, qa_sweep, clauses, k_gt, attrs_digest=attrs_digest
    )
    path = blob_path(gt_dir, sweep, fp)
    if path.exists():
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — a torn / foreign file at the cache path
            logger.warning("  {}: unreadable ({}: {}); rebuilding", path, type(exc).__name__, exc)
            blob = None
        if (
            isinstance(blob, dict)
            and blob.get("version") == BLOB_VERSION
            and blob.get("fingerprint") == fp
        ):
            logger.info("  loaded oracle from cache: {}", path)
            return blob
        if blob is not None:
            logger.warning("  {}: not a v4 blob for this fingerprint; rebuilding", path)
    logger.info("  building exact oracle (k_gt={}) for sweep {}", k_gt, sweep)
    n_kept = int((~skip_mask).sum()) if skip_mask is not None else int(queries.shape[0])
    blob = {
        "version": BLOB_VERSION,
        **compute(
            item_embs,
            queries,
            qa_sweep,
            skip_mask,
            filter_mod,
            k_gt,
            targets=targets,
            device=device,
        ),
        "n_items": int(item_embs.shape[0]),
        "n_queries": int(queries.shape[0]),
        "n_kept": n_kept,
        "k_gt": int(k_gt),
        "sweep": sweep,
        "clauses": None if clauses is None else [int(c) for c in clauses],
        "fingerprint": fp,
        "code_version": code_version(),
        "harness_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "torch": str(torch.__version__),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write(path, lambda fh: torch.save(blob, fh))
    logger.info("  saved oracle → {} (pass_rate={:.4f})", path, blob["pass_rate"])
    return blob


# ----- resume key ---------------------------------------------------------------------


def resume_key(key: dict[str, Any], code_version: str) -> str:
    """The JSONL resume key (H §8.2 B): ``Job.key(params)`` plus the library's
    ``code_version``, as canonical JSON. A record's key is
    ``resume_key({k: rec[k] for k in KEY_FIELDS}, rec["env"]["code_version"])``."""
    return json.dumps({**key, "code_version": code_version}, sort_keys=True, separators=(",", ":"))


KEY_FIELDS = (
    "dataset",
    "dim",
    "suite",
    "filter_kind",
    "sweep",
    "algo",
    "backend",
    "params",
    "seed",
)


__all__ = [
    "BLOB_VERSION",
    "KEY_FIELDS",
    "attrs_digest",
    "blob_path",
    "bloom_fp_rate",
    "compute",
    "fingerprint",
    "load_or_build",
    "pass_counts",
    "pass_rate",
    "resume_key",
]
